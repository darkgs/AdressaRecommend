
import os
import random

import tensorflow as tf
import numpy as np

from ad_util import write_log

article_embeding_dimension = 5
sequences_size = 100
max_seq_len = 10
candidate_size = 10
batch_size = 10000

urls = ['url' + str(i) for i in range(256)]

def generate_test_url2idx_idx2url():
	global urls

	dict_url2idx  = {}

	cur_i = 0
	dict_url2idx['url_pad'] = 0
	for url in urls:
		dict_url2idx[url] = ++cur_i

	dict_idx2url = {idx:word for word, idx in dict_url2idx.items()}

	return dict_url2idx, dict_idx2url


def generate_test_sequences():
	global urls
	global sequences_size, max_seq_len
	global candidate_size

	sequences = []
	sequence_lens = []
	for batch in range(sequences_size):
		sequence = []
		sequence_len = random.randrange(5,max_seq_len+1)

		count_by = 0
		for i in range(max_seq_len):
			if i < sequence_len:
				count_by += 1
				sequence.append(urls[random.randrange(0,len(urls))])
			else:
				sequence.append('url_pad')

		sequences.append(sequence)
		sequence_lens.append(sequence_len)

	candidates = []
	for i in range(len(sequences)):
		candidate = [urls[random.randrange(0,len(urls))] for i in range(candidate_size-1)]
		candidate.append(sequences[i][sequence_lens[i]-1])

		candidates.append(candidate)

	return sequences, sequence_lens, candidates


def get_sentence_batch(batch_size, data_x, 
		data_candi, data_seqlens, dict_url2idx):
	instance_indices = list(range(len(data_x)))
	np.random.shuffle(instance_indices)
	batch = instance_indices[:batch_size]

	x = [[dict_url2idx[word] for word in data_x[i][:-1]] for i in batch]
	y = [[dict_url2idx[word] for word in data_x[i][1:]] for i in batch]
	seq_lens = [data_seqlens[i]-1 for i in batch]
	candi = [[dict_url2idx[word] for word in data_candi[i][:-1]] for i in batch]

	return x, y, seq_lens, candi


def main():
	global urls
	global article_embeding_dimension, sequences_size
	global max_seq_len, candidate_size
	global batch_size

	dict_url2idx, dict_idx2url = generate_test_url2idx_idx2url()
	sequences, sequence_lens, candidates = generate_test_sequences()

#	_inputs = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
#	_ys = tf.placeholder(tf.int32, shape=[None, max_seq_len-1])
	_inputs = tf.placeholder(tf.int32, shape=[None, None])
	_ys = tf.placeholder(tf.int32, shape=[None, None])
	_seqlens = tf.placeholder(tf.int32, shape=[None])

	hidden_layer_size = 250

	with tf.name_scope('embeddings'):
		embeddings = tf.Variable(
				tf.random_uniform([len(urls), article_embeding_dimension], -1.0, 1.0),
				name='embedding')
		embed = tf.nn.embedding_lookup(embeddings, _inputs)
		embed_y = tf.nn.embedding_lookup(embeddings, _ys)

	with tf.variable_scope('lstm'):
		lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_layer_size)
		rnn_outputs, states = tf.nn.dynamic_rnn(lstm_cell, embed, sequence_length=_seqlens, dtype=tf.float32)
		rnn_outputs_stratch = tf.reshape(rnn_outputs, [-1, hidden_layer_size])
		outputs = tf.layers.dense(rnn_outputs_stratch, article_embeding_dimension)

		outputs_norm = tf.nn.l2_normalize(outputs,0)
		ys_norm = tf.nn.l2_normalize(tf.reshape(embed_y, [-1, article_embeding_dimension]),0)

		cos_loss = tf.losses.cosine_distance(ys_norm, outputs_norm, axis=1)
		train_step = tf.train.AdamOptimizer(1e-3).minimize(cos_loss)

		acc, acc_op = tf.metrics.mean_cosine_distance(ys_norm, outputs_norm, 1)

	x_test, y_test, seq_lens_test, candi_test = get_sentence_batch(batch_size=100,
			data_x=sequences, data_candi=candidates,
			data_seqlens=sequence_lens, dict_url2idx=dict_url2idx)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())

		for epoch in range(100):
			x_batch, y_batch, seq_lens, candi = get_sentence_batch(batch_size=batch_size,
					data_x=sequences, data_candi=candidates,
					data_seqlens=sequence_lens, dict_url2idx=dict_url2idx)

			sess.run(train_step, feed_dict={
					_inputs:x_batch,
					_ys:y_batch,
					_seqlens:seq_lens,
				})

			valid_loss = sess.run(loss, feed_dict={
					_inputs:x_test,
					_ys:y_test,
					_seqlens:seq_lens_test,
				})

			print('epoch : {} - valid loss : {}'.format(epoch, valid_loss))

		sess.run(rnn_outputs, feed_dict={
				_inputs:x_batch,
				_ys:y_batch,
				_seqlens:seq_lens,
			})


if __name__ == '__main__':
	main()
