
import os
import json, time
import random

from optparse import OptionParser

import tensorflow as tf
import numpy as np

from ad_util import get_files_under_path
from ad_util import write_log


parser = OptionParser()
parser.add_option('-i', '--input_path', dest='input_path', type='string', default=None)
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)
parser.add_option('-r', '--rnn_input_path', dest='rnn_input_path', type='string', default=None)

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

per_time_path = None
per_user_path = None
dict_rnn_input = {}

def get_input_batch(input_type='train'):
	global dict_rnn_input
	
	total_length = len(dict_rnn_input['seq_lengths'])

	train_end_idx = int(total_length * 6 / 10)
	valid_end_idx = int(total_length * 8 / 10)

	if input_type == 'train':
		return dict_rnn_input['seq_lengths'][:train_end_idx], \
			dict_rnn_input['rnn_input'][:train_end_idx]
	elif input_type == 'valid':
		return dict_rnn_input['seq_lengths'][train_end_idx:valid_end_idx], \
			dict_rnn_input['rnn_input'][train_end_idx:valid_end_idx]
	else:
		return dict_rnn_input['seq_lengths'][valid_end_idx:], \
			dict_rnn_input['rnn_input'][valid_end_idx:]

def main():
	global per_time_path, per_user_path, dict_rnn_input

	options, args = parser.parse_args()
	if (options.input_path == None) or (options.data_path == None) or (options.rnn_input_path == None):
		return

	seperated_input_path = options.input_path
	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'
	rnn_input_path = options.rnn_input_path

	print('Loading rnn_input : start')
	with open(rnn_input_path + '/rnn_input.json', 'r') as f_input:
		dict_rnn_input = json.load(f_input)
	print('Loading rnn_input : end')

	seq_lengths_train, rnn_input = get_input_batch(input_type='train')
	input_train = np.array(rnn_input)
	rnn_input = None

	seq_lengths_valid, rnn_input = get_input_batch(input_type='valid')
	input_valid = np.array(rnn_input)
	rnn_input = None

	seq_lengths_test, rnn_input = get_input_batch(input_type='test')
	input_test = np.array(rnn_input)
	rnn_input = None

	# Dataset
	batch_total = 1000
	time_steps = 10
	vector_dim = 100

	hidden_layer_size = 20

#	input_train = generate_random_set(batch_size=batch_total*6/10,
#			time_steps=time_steps, vector_dim=vector_dim)
#	input_valid = generate_random_set(batch_size=batch_total*2/10,
#			time_steps=time_steps, vector_dim=vector_dim)
#	input_test = generate_random_set(batch_size=batch_total*2/10,
#			time_steps=time_steps, vector_dim=vector_dim)

	return

	# Graph
	_inputs = tf.placeholder(tf.float32,
			shape=[None, None, vector_dim], name='inputs')
#			shape=[None, None, vector_dim], name='inputs')

	y = tf.placeholder(tf.float32,
			shape=[None, None, vector_dim], name='output')

	rnn_cell = tf.contrib.rnn.LSTMCell(hidden_layer_size)
	outputs, _ = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)


	Wl = tf.Variable(tf.truncated_normal([hidden_layer_size, vector_dim],
				mean=0, stddev=.01))
	bl = tf.Variable(tf.truncated_normal([vector_dim], mean=0, stddev=.01))

	def get_linear_layer(vector):
		return tf.matmul(vector, Wl) + bl

	last_rnn_output = outputs
	final_output = get_linear_layer(last_rnn_output)

	softmax = tf.nn.softmax_cross_entropy_with_logits(logits=final_output,
			labels = y)

	cross_entropy = tf.reduce_mean(softmax)
	train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

	normalize_output = tf.nn.l2_normalize(final_output,0)
	normalize_y = tf.nn.l2_normalize(y,0)
	cos_similarity = tf.reduce_sum(tf.subtract(normalize_output, normalize_y))

	# Session
	sess = tf.InteractiveSession()
	sess.run(tf.global_variables_initializer())

	for i in range(1000):
		batch_x, batch_y = input_train[:,:-1,:], input_train[:,1:,:]

		sess.run(train_step, feed_dict={
					_inputs: batch_x,
					y: batch_y,
				})

		acc = sess.run(cos_similarity, feed_dict={
					_inputs: input_valid[:,:-1,:],
					y: input_valid[:,1:,:],
				})

		loss = sess.run(cross_entropy, feed_dict={
					_inputs: input_valid[:,:-1,:],
					y: input_valid[:,1:,:],
				})
		
		if i % 100 == 0:
			print('Iter {} valid Loss:{:.6f}, Accuracy:{:.5f}'.format(i, loss, acc))

	print('Testing Accuracy : {}'.format(
		sess.run(cos_similarity, feed_dict={
				_inputs: input_test[:,:-1,:],
				y: input_test[:,1:,:],
			})
		))


if __name__ == '__main__':
	main()

