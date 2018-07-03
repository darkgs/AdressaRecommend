
import os
import random

import tensorflow as tf
import numpy as np

from ad_util import write_log

word_vector_dimension = 5
batch_size = 20
max_seq_len = 10

test_word_embeding = {
	'a': np.random.rand(word_vector_dimension),
	'b': np.random.rand(word_vector_dimension),
	'c': np.random.rand(word_vector_dimension),
	'd': np.random.rand(word_vector_dimension),
	'e': np.random.rand(word_vector_dimension),
}

vocas = ['a','b','c','d','e']

def generate_test_word2idx():
	global vocas

	dict_w2i = {}

	cur_i = 0
	for word in vocas:
		dict_w2i[word] = cur_i
		cur_i += 1

	return dict_w2i

def generate_test_idx2word(dict_w2i={}):
	return {idx:word for word, idx in dict_w2i.items()}

def generate_test_sentences():
	global vocas

	test_sentences = []
	test_seq_lens = []

	for i in range(batch_size):
		sentence_length = random.randrange(4,max_seq_len+1)
		test_sentence = []
		for j in range(sentence_length):
			test_sentence.append(vocas[random.randrange(0,5)])
		test_sentences.append(test_sentence)
		test_seq_lens.append(sentence_length)

	return test_sentences, test_seq_lens

def main():
	print(test_word_embeding['a'].shape)
	generate_test_sentences()


if __name__ == '__main__':
	main()

