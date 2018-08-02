
import json

import tensorflow as tf

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)


class RNN_Input:
	def __init__(self, tf_record_dir, max_seq_len):
#		dict_extra_infos = {
#			'idx2vec': dict_idx_vec,
#			'time_idx': dict_time_idx,
#		}
		with open('{}/extra_infos.dict'.format(tf_record_dir), 'r') as f_extra:
			self._dict_extra_infos = json.load(f_extra)

		self._idx_count = len(self._dict_extra_infos['idx2vec'])
		self._max_seq_len = max_seq_len

		self._tf_record_dir = tf_record_dir

	
	def get_tf_records_path(self, input_type='train'):
		if input_type == 'valid' or input_type == 'test':
			return '{}/{}.tfrecord'.format(self._tf_record_dir, input_type)
			
		return '{}/train.tfrecord'.format(self._tf_record_dir)


	def __del__(self):
		self._dict_extra_infos = None
	

	def idx2vec(self, idx):
		return self._dict_extra_infos['idx2vec'][str(idx)]


	def max_seq_len(self):
		return self._max_seq_len


	def idx_count(self):
		return self._idx_count


	def get_candidates(self, start_time=-1, end_time=-1, idx_count=0):
		if (start_time < 0) or (end_time < 0) or (idx_count <= 0):
			return []

		#	entry of : dict_extra_infos['time_idx']
		#	(timestamp) :
		#	{
		#		prev_time: (timestamp)
		#		next_time: (timestamp)
		#		'indices': { idx:count, ... }
		#	}

		# swap if needed
		if start_time > end_time:
			tmp_time = start_time
			start_time = end_time
			end_time = tmp_time

		cur_time = start_time

		dict_merged = {}

		while(cur_time < end_time):
			cur_time = self._dict_extra_infos['time_idx'][str(cur_time)]['next_time']
			for idx, count in self._dict_extra_infos['time_idx'][str(cur_time)]['indices'].items():
				dict_merged[idx] = dict_merged.get(idx, 0) + count

		steps = 0
		time_from_start = start_time
		time_from_end = end_time
		while(len(dict_merged.keys()) < idx_count):
			if time_from_start == None and time_from_end == None:
				break

			if steps % 3 == 0:
				if time_from_end == None:
					steps += 1
					continue

				cur_time = self._dict_extra_infos['time_idx'][str(time_from_end)]['next_time']
				time_from_end = cur_time
			else:
				if time_from_start == None:
					steps += 1
					continue

				cur_time = self._dict_extra_infos['time_idx'][str(time_from_start)]['prev_time']
				time_from_start = cur_time

			if cur_time == None:
				continue

			for idx, count in self._dict_extra_infos['time_idx'][str(cur_time)]['indices'].items():
				dict_merged[idx] = dict_merged.get(idx, 0) + count

		ret_sorted = sorted(dict_merged.items(), key=lambda x:x[1], reverse=True)
		return [int(idx) for idx, count in ret_sorted]


def main():
	hidden_layer_size = 1000
	rnn_layer_count = 3
	batch_size = 100

	options, args = parser.parse_args()

	if (options.input == None):
		return

	tf_record_dir = options.input
	rnn_input = RNN_Input(tf_record_dir=tf_record_dir, max_seq_len=20)
	embedding_dimension = len(rnn_input.idx2vec(0))
	max_seq_len = rnn_input.max_seq_len()
	idx_count = rnn_input.idx_count()

	print(embedding_dimension, max_seq_len, idx_count)
	print(rnn_input.get_tf_records_path(input_type='train'))

	with tf.name_scope('embeddings'):
		embeddings = tf.constant(
				[rnn_input.idx2vec(i) for i in range(idx_count)],
				dtype=tf.float32,
				name='embedding',
			)

	def parse_tf_records(example):
		record = tf.parse_single_example(example, features={
					'start_time': tf.FixedLenFeature([], tf.int64),
					'end_time': tf.FixedLenFeature([], tf.int64),
					'sequence_x': tf.VarLenFeature(tf.int64),
					'sequence_y': tf.VarLenFeature(tf.int64),
					'seq_len': tf.FixedLenFeature([], tf.int64),
				})

		return record['sequence_x'], record['sequence_y'], record['start_time'], record['end_time'], record['seq_len']

#	_tf_record_path = tf.placeholder(tf.string, [None], name='filenames')
	dataset = tf.data.TFRecordDataset('{}/{}.tfrecord'.format(tf_record_dir, 'test')) \
			  .map(parse_tf_records, num_parallel_calls=10) \
			  .batch(batch_size)

#	dataset = dataset.cache()
	iterator = dataset.make_initializable_iterator()
	_initializer = iterator.initializer

	xs, ys, start_time, end_time, seq_lens = iterator.get_next()

#	_xs = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
#	_ys = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_xs = tf.placeholder(tf.int32, shape=[None, None])
	_ys = tf.placeholder(tf.int32, shape=[None, None])
	_seqlens = tf.placeholder(tf.int32, shape=[None])

	with tf.name_scope('embeddings'):
#		embed_x = tf.nn.embedding_lookup(embeddings, _xs)
#		embed_y = tf.nn.embedding_lookup(embeddings, _ys)
		embed_x = tf.nn.embedding_lookup_sparse(embeddings, xs, None)
		embed_y = tf.nn.embedding_lookup_sparse(embeddings, ys, None)

	"""
	with tf.variable_scope('lstm'):
		lstm_cell = tf.contrib.rnn.MultiRNNCell(
				[tf.contrib.rnn.GRUCell(hidden_layer_size) for _ in range(rnn_layer_count)])

		rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell,
				embed_x, sequence_length=seq_lens, dtype=tf.float32)

		rnn_outputs_stratch = tf.reshape(rnn_outputs, [-1, hidden_layer_size])
		outputs = tf.layers.dense(rnn_outputs_stratch, embedding_dimension)

		ys_stratch = tf.reshape(embed_y, [-1, embedding_dimension])
		
		outputs_norm = tf.nn.l2_normalize(outputs,0)
		ys_norm = tf.nn.l2_normalize(ys_stratch,0)

		#loss = tf.losses.cosine_distance(ys_norm, outputs_norm, axis=1)
		loss = tf.losses.mean_squared_error(outputs, ys_stratch)

		train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

	with tf.variable_scope('metric'):
		rank_scores = tf.matmul(embeddings, tf.transpose(tf.reshape(outputs, [-1, embedding_dimension])))
		_, top_all = tf.nn.top_k(tf.transpose(rank_scores), embeddings.shape[0])
	"""

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.7

	print('{}/{}.tfrecord'.format(tf_record_dir, 'test'))

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		sess.run(_initializer)

		ret = sess.run(embed_x)

		print(ret.shape)


if __name__ == '__main__':
	main()
