
import os
import random
import json

import tensorflow as tf
import numpy as np

from optparse import OptionParser

from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)

dict_rnn_input = {}

def generate_batchs(input_type='train', batch_size=10):
	global dict_rnn_input

	total_len = len(dict_rnn_input['sequence'])
	if batch_size < 0:
		batch_size = total_len

	if input_type == 'train':
		idx_from = 0
		idx_to = int(total_len * 6 / 10)
	elif input_type == 'valid':
		idx_from = int(total_len * 6 / 10)
		idx_to = int(total_len * 8 / 10)
	else:
		idx_from = int(total_len * 8 / 10)
		idx_to = total_len

	batch_size = min(batch_size, idx_to - idx_from)

	data_idxs = list(range(idx_from, idx_to))
	np.random.shuffle(data_idxs)

#	dict_rnn_input['timestamp']
#	dict_rnn_input['seq_len']
#	dict_rnn_input['idx2url']
#	dict_rnn_input['sequence']

	sequence = np.matrix(np.array(dict_rnn_input['sequence'])[data_idxs][:batch_size].tolist())
	seq_len = np.array(dict_rnn_input['seq_len'])[data_idxs][:batch_size]

	input_x = sequence[:,:-1]
	input_y = sequence[:,1:]

	return input_x, input_y, seq_len-1


def main():
	global dict_rnn_input

	hidden_layer_size = 300
	rnn_layer_count = 3

	options, args = parser.parse_args()

	if (options.input == None) or (options.u2v_path == None):
		return

	rnn_input_path = options.input + '/rnn_input.json'
	url2vec_path = options.u2v_path

	write_log('Loading start')
	with open(rnn_input_path, 'r') as f_input:
		dict_rnn_input = json.load(f_input)

	# Padding.. better way??
	max_seq_len = max(dict_rnn_input['seq_len'])
	for seq_entry in dict_rnn_input['sequence']:
		pad_count = max_seq_len - len(seq_entry)
		if pad_count > 0:
			seq_entry += [0] * pad_count

	with open(url2vec_path, 'r') as f_u2v:
		dict_url2vec = json.load(f_u2v)
	write_log('Loading end')

	write_log('Generate embeddings : start')
	url_count = len(dict_rnn_input['idx2url'])
	embedding_dimension = len(dict_url2vec.items()[0][1])

	dict_url2vec['url_pad'] = [0.0]*embedding_dimension

	with tf.name_scope('embeddings'):
#		embeddings = tf.Variable(
		embeddings = tf.constant(
					[dict_url2vec[dict_rnn_input['idx2url'][str(i)]] for i in range(url_count)],
					dtype=tf.float32,
					name='embedding',
				)
	write_log('Generate embeddings : end')

#	_inputs = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
#	_ys = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_xs = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_ys = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_seqlens = tf.placeholder(tf.int32, shape=[None])

	with tf.name_scope('embeddings'):
		embed_x = tf.nn.embedding_lookup(embeddings, _xs)
		embed_y = tf.nn.embedding_lookup(embeddings, _ys)

	with tf.variable_scope('lstm'):
		lstm_cell = tf.contrib.rnn.MultiRNNCell(
#			[tf.contrib.rnn.BasicLSTMCell(hidden_layer_size) \
			[tf.contrib.rnn.GRUCell(hidden_layer_size) \
				for _ in range(rnn_layer_count)])

		rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell,
				embed_x, sequence_length=_seqlens, dtype=tf.float32)
		rnn_outputs_stratch = tf.reshape(rnn_outputs, [-1, hidden_layer_size])
		outputs = tf.layers.dense(rnn_outputs_stratch, embedding_dimension)

		ys_stratch = tf.reshape(embed_y, [-1, embedding_dimension])

		outputs_norm = tf.nn.l2_normalize(outputs,0)
		ys_norm = tf.nn.l2_normalize(ys_stratch,0)

#		loss = tf.losses.cosine_distance(ys_norm, outputs_norm, axis=1)
		loss = tf.losses.mean_squared_error(outputs, ys_stratch)
		train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

	with tf.variable_scope('metric'):
		acc, acc_op = tf.metrics.mean_cosine_distance(ys_norm, outputs_norm, 1)

		rank_scores = tf.matmul(embeddings, tf.transpose(tf.reshape(outputs, [-1, embedding_dimension])))
		_, top_all = tf.nn.top_k(tf.transpose(rank_scores), embeddings.shape[0])


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		for epoch in range(1000):
			train_x, train_y, train_seq_len = generate_batchs(input_type='train', batch_size=10000)

			sess.run(train_step, feed_dict={
					_xs: train_x,
					_ys: train_y,
					_seqlens: train_seq_len,
				})

			test_x, test_y, test_seq_len = generate_batchs(input_type='train', batch_size=100)

			test_loss, test_top_all = sess.run([loss, top_all], feed_dict={
					_xs: test_x,
					_ys: test_y,
					_seqlens: test_seq_len,
				})

			answers = tf.reshape(test_y, (-1, 1)).eval().tolist()
			rank_infos = tf.reshape(test_top_all, (-1, url_count)).eval().tolist()

			predict_total = 0
			predict_mrr = 0.0
			for batch_idx in range(len(test_seq_len)):
				for seq_idx in range(test_seq_len[batch_idx]):
					valid_idx = batch_idx * (max_seq_len-1) + seq_idx

					predict_total += 1
					hit_rank = rank_infos[valid_idx].index(answers[valid_idx][0]) + 1
					if hit_rank > 0:
						predict_mrr += 1.0/float(hit_rank)

			mrr_metric = predict_mrr / float(predict_total)

			write_log('epoch : {} - test loss:{} - mrr:{}'.format(epoch, test_loss, mrr_metric))

if __name__ == '__main__':
	main()
