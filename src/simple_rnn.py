
import os

import tensorflow as tf
import numpy as np

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

def main():
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
