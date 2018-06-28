
import os
import json
import random
#import pathlib

from optparse import OptionParser

import tensorflow as tf
import numpy as np

from ad_util import write_log


parser = OptionParser()
parser.add_option('-m', '--mode', dest='mode', type='string', default=None)
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)
parser.add_option('-w', '--w2v_json', dest='w2v_json', type='string', default=None)
parser.add_option('-c', '--cache_path', dest='cache_path', type='string', default=None)

def generate_random_set(batch_size, time_steps, vector_dim):
	T = 100

	datas = np.empty((batch_size, time_steps, vector_dim), 'float32')
	for i in range(batch_size):
#		data = np.random.randint(0,T, time_steps).reshape(time_steps, 1) + np.ones(vector_dim)
#		data = (np.ones(time_steps) * i).reshape(time_steps, 1) + np.ones(vector_dim)
		data = np.random.randint(0,T, (time_steps, vector_dim))
		data = (data / 1.0 / T).astype('float32')
		datas[i] = data

	return datas

dataset_mode = None
w2v_path = None
per_time_path = None
per_user_path = None
cache_path = None


def generate_rnn_input():
	global w2v_path, per_time_path, per_user_path, cache_path

	rnn_input = {
		'train': {},
		'valid': {},
		'test': {},
	}

	write_log('Per_time Load : start')
	with open(per_time_path, 'r') as f_per_time:
		events_per_time = json.load(f_per_time)
	write_log('Per_time Load : end')

	train_time_limit = events_per_time[len(events_per_time)*7/10][0]

	write_log('Per_user Load : start')
	with open(per_user_path, 'r') as f_per_user:
		events_per_user = json.load(f_per_user)
	write_log('Per_user Load : end')

	write_log('w2v Load : start')
	with open(w2v_path, 'r') as f_w2v:
		dict_w2v = json.load(f_w2v)
	write_log('w2v Load : end')

	valid_url_list = dict_w2v.keys()

	write_log('Extract sequences : start')
	total_count = len(events_per_user.keys())
	count = 0
	for user_id, events in events_per_user.items():
		if i % 1000 == 0:
			write_log('processing {}/{}'.format(count,total_count))
		count += 1

		sequence = []
		for timestamp, url in events:
			if url not in valid_url_list:
				continue
			sequence.append(dict_w2v[url])

		input_type = 'train'
		if events[-1][0] > train_time_limit:
			if random.random() < 0.5:
				input_type = 'valid'
			else:
				input_type = 'test'

		batch_size = len(sequence)
		if batch_size < 2:
			continue

		if rnn_input[input_type].get(batch_size, None) == None:
			rnn_input[input_type][batch_size] = []
		rnn_input[input_type][batch_size].append(sequence)
	write_log('Extract sequences : end')

	with open(cache_path + '/rnn_input.json', 'w') as f_input:
		json.dump(rnn_input, f_input)

	for input_type, dict_input in rnn_input.items():
		print('==={}==='.format(input_type))
		for batch_size, sequences in dict_input.items():
			print('{} : {}'.format(batch_size, len(sequences)))


def main():
	global dataset_mode, w2v_path, per_time_path, per_user_path, cache_path

	options, args = parser.parse_args()
	if ((options.mode == None) or (options.data_path == None) or
			(options.w2v_json == None) or (options.cache_path == None)):
		return

	dataset_mode = options.mode
	w2v_path = options.w2v_json
	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'
	cache_path = options.cache_path

#	pathlib.Path(cache_path).mkdir(parents=True, exist_ok=True)
	os.system('mkdir -p ' + cache_path)

	generate_rnn_input()

	return 
	# Dataset
	batch_total = 1000
	time_steps = 10
	vector_dim = 100

	hidden_layer_size = 20

	input_train = generate_random_set(batch_size=batch_total*6/10,
			time_steps=time_steps, vector_dim=vector_dim)
	input_valid = generate_random_set(batch_size=batch_total*2/10,
			time_steps=time_steps, vector_dim=vector_dim)
	input_test = generate_random_set(batch_size=batch_total*2/10,
			time_steps=time_steps, vector_dim=vector_dim)

	# Graph
	_inputs = tf.placeholder(tf.float32,
			shape=[None, None, vector_dim], name='inputs')
#			shape=[None, None, vector_dim], name='inputs')

	y = tf.placeholder(tf.float32,
			shape=[None, vector_dim], name='output')

	rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
	outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

	Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, vector_dim],
				mean=0, stddev=.01))
	bl = tf.Variable(tf.truncated_normal([vector_dim], mean=0, stddev=.01))

	def get_linear_layer(vector):
		return tf.matmul(vector, Wl) + bl

	last_rnn_output = outputs[:,-1,:]
	final_output = get_linear_layer(last_rnn_output)

	softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
			labels = y)

	cross_entropy = tf.reduce_mean(softmax)
	train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

	normalize_output = tf.nn.l2_normalize(final_output,0)
	normalize_y = tf.nn.l2_normalize(y,0)
	cos_similarity = tf.reduce_sum(tf.multiply(normalize_output, normalize_y))

	# Session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for i in range(1000):
		batch_x, batch_y = input_train[:,:-1,:], input_train[:,-1,:]

		sess.run(train_step, feed_dict={
					_inputs: batch_x,
					y: batch_y,
				})

		acc = sess.run(cos_similarity, feed_dict={
					_inputs: input_valid[:,:-1,:],
					y: input_valid[:,-1,:],
				})

		loss = sess.run(cross_entropy, feed_dict={
					_inputs: input_valid[:,:-1,:],
					y: input_valid[:,-1,:],
				})
		
		if i % 100 == 0:
			write_log('Iter {} valid Loss:{:.6f}, Accuracy:{:.5f}'.format(i, loss, acc))

	write_log('Testing Accuracy : {}'.format(
		sess.run(cos_similarity, feed_dict={
				_inputs: input_test[:,:-1,:],
				y: input_test[:,-1,:],
			})
		))


if __name__ == '__main__':
	main()

