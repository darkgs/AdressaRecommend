
import os
import random
import json

import tensorflow as tf
import numpy as np

from optparse import OptionParser

from ad_util import write_log
from ad_util import RNN_Input

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-m', '--mode', dest='mode', type='string', default='one_week')
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')

def main():

	hidden_layer_size = 1000
	rnn_layer_count = 3

	options, args = parser.parse_args()

	if (options.input == None) or (options.u2v_path == None):
		return

	# Pre-processed RNN input
	rnn_input_path = options.input + '/rnn_input.json'
	url2vec_path = options.u2v_path

	# Intermediate saved model
	mode = options.mode
	d2v_embed = options.d2v_embed
	saved_model_path = 'cache/{}/model/d2v_rnn/embed-{}/model.ckpt'.format(mode, d2v_embed)

	if not os.path.exists(os.path.dirname(saved_model_path)):
		os.system('mkdir -p ' + os.path.dirname(saved_model_path))

	# Statics to report
	report_path = 'report/{}/d2v_rnn/embed-{}.csv'.format(mode, d2v_embed)
	if not os.path.exists(os.path.dirname(report_path)):
		os.system('mkdir -p ' + os.path.dirname(report_path))
	
	# Change d2v file
	target_path = url2vec_path + '_' + d2v_embed
	if not os.path.exists(target_path):
		print('Can not find d2v file : {}'.format(target_path))
		return
	os.system('cp -f {} {}'.format(target_path, url2vec_path))

	write_log('Loading start')
	rnn_input = RNN_Input(rnn_input_path)

	with open(url2vec_path, 'r') as f_u2v:
		dict_url2vec = json.load(f_u2v)
	write_log('Loading end')

	write_log('Generate embeddings : start')
	url_count = rnn_input.url_count()
	max_seq_len = rnn_input.max_seq_len()
	embedding_dimension = len(dict_url2vec[next(iter(dict_url2vec))])

	dict_url2vec['url_pad'] = [0.0]*embedding_dimension
	with tf.name_scope('embeddings'):
		embeddings = tf.constant(
					[dict_url2vec[rnn_input.idx2url(i)] for i in range(url_count)],
					dtype=tf.float32,
					name='embedding',
				)
	write_log('Generate embeddings : end')

	_xs = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_ys = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_seqlens = tf.placeholder(tf.int32, shape=[None])

	with tf.name_scope('embeddings'):
		embed_x = tf.nn.embedding_lookup(embeddings, _xs)
		embed_y = tf.nn.embedding_lookup(embeddings, _ys)

	with tf.variable_scope('lstm'):
		lstm_cell = tf.contrib.rnn.MultiRNNCell(
#			[tf.contrib.rnn.BasicLSTMCell(hidden_layer_size) \
			[tf.contrib.rnn.GRUCell(hidden_layer_size) for _ in range(rnn_layer_count)])

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
		rank_scores = tf.matmul(embeddings, tf.transpose(tf.reshape(outputs, [-1, embedding_dimension])))
		_, top_all = tf.nn.top_k(tf.transpose(rank_scores), embeddings.shape[0])


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.gpu_options.per_process_gpu_memory_fraction = 0.7

#	saver_config = tf.ConfigProto()
#	saver_config.gpu_options.allow_growth = True
#	saver_config.gpu_options.per_process_gpu_memory_fraction = 0.2
#	saver = tf.train.Saver(config=saver_config)

	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		restored_step = 0
#		try:
#			saver.restore(sess, tf.train.latest_checkpoint(os.path.dirname(saved_model_path)))
#			ckpt = tf.train.get_checkpoint_state(os.path.dirname(saved_model_path))
#			restored_step =  int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
#			write_log('Restore model : {}, restored_step : {}'.format(saved_model_path, restored_step))
#		except tf.errors.NotFoundError as e:
#			write_log('Failed to restore model : {}'.format(saved_model_path))
#		except ValueError as e:
#			write_log('Failed to restore model : {}'.format(saved_model_path))
			
		def test_mrr_metric(input_type='test', sampling=50, repeat=1):
			ret_loss = []
			ret_mrr = []
			for i in range(repeat):
				test_x, test_y, test_seq_len, test_timestamps = rnn_input.generate_batchs(input_type=input_type, batch_size=sampling)
				test_loss, test_top_all = sess.run([loss, top_all], feed_dict={
						_xs: test_x,
						_ys: test_y,
						_seqlens: test_seq_len,
					})

				answers = tf.reshape(test_y, (-1, 1)).eval().tolist()
				rank_infos = tf.reshape(test_top_all, (-1, url_count)).eval().tolist()

				# Very slow!!
				predict_total = 0
				predict_mrr = 0.0
				for batch_idx in range(len(test_seq_len)):
					cand_start_time = test_timestamps[batch_idx][0]
					cand_end_time = test_timestamps[batch_idx][1]

					cand_indices = rnn_input.get_candidates(start_time=cand_start_time,
							end_time=cand_end_time, idx_count=100)

					for seq_idx in range(test_seq_len[batch_idx]):
						valid_idx = batch_idx * (max_seq_len-1) + seq_idx

						predict_total += 1
						hit_rank = 0
						for i in range(len(rank_infos[valid_idx])):
							if rank_infos[valid_idx][i] == answers[valid_idx][0]:
								hit_rank += 1
								break
							if rank_infos[valid_idx][i] in cand_indices:
								hit_rank += 1

							# MRR@20
							if hit_rank > 20:
								hit_rank = 0
								break

						if hit_rank > 0:
							predict_mrr += 1.0/float(hit_rank)

				mrr_metric = predict_mrr / float(predict_total)

				ret_loss.append(test_loss)
				ret_mrr.append(mrr_metric)

			loss_avg = sum(ret_loss)/float(repeat)
			mrr_avg = sum(ret_mrr)/float(repeat)

			return loss_avg, mrr_avg

#		_, prev_top_valid_mrr = test_mrr_metric('valid', 50, 10)
#		_, prev_top_test_mrr = test_mrr_metric('test', 50, 10)

#		write_log('Before train : valid mrr({}), test mrr({})'.format(prev_top_valid_mrr, prev_top_test_mrr))
		for epoch in range(500):
			global_step=restored_step+epoch

			# Train
			train_x, train_y, train_seq_len, train_timestamps = rnn_input.generate_batchs(input_type='train', batch_size=500)

			sess.run(train_step, feed_dict={
					_xs: train_x,
					_ys: train_y,
					_seqlens: train_seq_len,
				})

			print('epoch : {}'.format(epoch))

		for i in range(20):
			_, test_mrr = test_mrr_metric('test', 100, 1)
			print('{},{}'.format(i, test_mrr))

#			if (global_step > restored_step) and (global_step % 10 == 0):
#				write_log('global_step : {} - metric start'.format(global_step))
#				valid_loss, valid_mrr = test_mrr_metric('valid', 50, 10)
#				write_log('global_step : {} - valid loss:{} - mrr:{}'.format(global_step, valid_loss, valid_mrr))

#				_, test_mrr = test_mrr_metric('test', 50, 10)
#				if True:
#					with open(report_path, 'a') as report_f:
#						report_f.write('{},{},{}\n'.format(global_step, valid_mrr, test_mrr))

#				if (global_step > 49) and (test_mrr > prev_top_test_mrr):
#					write_log('global_step : {} - New Record {} -> {}'.format(global_step, prev_top_test_mrr, test_mrr))
#					prev_top_test_mrr = test_mrr;
#					save_path = saver.save(sess, saved_model_path, global_step=global_step )
#					write_log('Model saved : {}'.format(save_path))


if __name__ == '__main__':
	main()
