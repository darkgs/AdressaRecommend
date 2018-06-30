
import os
import json, time
import random
#import pathlib

from optparse import OptionParser

import tensorflow as tf
import numpy as np

from ad_util import get_files_under_path
from ad_util import write_log


parser = OptionParser()
parser.add_option('-i', '--input_path', dest='input_path', type='string', default=None)
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)

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

def generate_rnn_input_():
	global w2v_path, per_time_path, per_user_path, cache_path

	rnn_input = {
		'train': {},
		'valid': {},
		'test': {},
	}

	print('Per_time Load : start')
	with open(per_time_path, 'r') as f_per_time:
		events_per_time = json.load(f_per_time)
	print('Per_time Load : end')

	train_time_limit = events_per_time[len(events_per_time)*7/10][0]

	print('Per_user Load : start')
	with open(per_user_path, 'r') as f_per_user:
		events_per_user = json.load(f_per_user)
	print('Per_user Load : end')

	print('w2v Load : start')
	with open(w2v_path, 'r') as f_w2v:
		dict_w2v = json.load(f_w2v)
	print('w2v Load : end')

	print('Extract sequences : start')
	total_count = len(events_per_user.keys())
	count = 0
	for user_id, events in events_per_user.items():
		if count % 1000 == 0:
			print('processing {}/{}'.format(count,total_count))
		count += 1

		sequence = []
		for timestamp, url in events:
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
	print('Extract sequences : end')

	with open(cache_path + '/rnn_input.json', 'w') as f_input:
		json.dump(rnn_input, f_input)

	for input_type, dict_input in rnn_input.items():
		print('==={}==='.format(input_type))
		for batch_size, sequences in dict_input.items():
			print('{} : {}'.format(batch_size, len(sequences)))

per_time_path = None
per_user_path = None

def generate_rnn_input(seperated_input_path=None, max_seq_len=10):
	if seperated_input_path == None:
		return {}

	rnn_input = []
	seq_lengths = []

	for seperated_path in get_files_under_path(seperated_input_path):
		with open(seperated_path, 'r') as f_dict:
			seperated_dict = json.load(f_dict)

		for user_id, sequence in seperated_dict.items():
			len_seq = len(sequence)
			if len_seq > max_seq_len:
				sequence = sequence[len_seq-max_seq_len:]
			elif len_seq < max_seq_len:
				sequence += [[0.0]*100] * (max_seq_len - len_seq)

			rnn_input.append(sequence)
			seq_lengths.append(len_seq)

	return seq_lengths, rnn_input

def main():
	global per_time_path, per_user_path

	options, args = parser.parse_args()
	if (options.input_path == None) or (options.data_path == None):
		return

	seperated_input_path = options.input_path
	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'

	write_log('RNN Input merging : start')
	start_time = time.time()
	seq_lengths, rnn_input = generate_rnn_input(seperated_input_path, 20)
	write_log('RNN Input merging : end tooks {}'.format(time.time() - start_time))

	dict_rnn_input = {
		'seq_lengths': seq_lengths,
		'rnn_input': rnn_input,
	}
	with open('rnn_input.json', 'w') as f_input:
		json.dump(dict_rnn_input, f_input)

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
			print('Iter {} valid Loss:{:.6f}, Accuracy:{:.5f}'.format(i, loss, acc))

	print('Testing Accuracy : {}'.format(
		sess.run(cos_similarity, feed_dict={
				_inputs: input_test[:,:-1,:],
				y: input_test[:,-1,:],
			})
		))


if __name__ == '__main__':
	main()

