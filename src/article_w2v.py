
import json, pickle
import time
import random

import numpy as np

import gensim
from optparse import OptionParser

from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-m', '--model_path', dest='model_path', type='string', default=None)
parser.add_option('-d', '--dataset', dest='dataset', type='string', default=None)
parser.add_option('-g', '--glob_pickle', dest='glob_pickle', type='string', default=None)

article_info_path = None
output_path = None
embedding_dimension = None
model_path = None

def generate_w2v_map():
	global article_info_path, output_path, embedding_dimension, model_path

	write_log('W2V Load article info : Start')
	with open(article_info_path, 'r') as f_art:
		article_info = json.load(f_art)
	write_log('W2V Load article info : End')

	write_log('W2V Generate labeled_sentences : Start')
	labeled_sentences = []
	for url, dict_info in article_info.items():
		sentence_header = dict_info.get('sentence_header', None)
		sentence_body = dict_info.get('sentence_body', None)

		if (sentence_header == None) or (sentence_body == None):
			continue

		words = []
		for sentence in sentence_header + sentence_body:
			for word in sentence.split(' '):
				words.append(word)

		labeled_sentence = gensim.models.doc2vec.LabeledSentence(words=words, tags=[url])
		labeled_sentences.append(labeled_sentence)
	write_log('W2V Generate labeled_sentences : End')

	w2v_model = gensim.models.Doc2Vec(alpha=.025, min_alpha=.001, min_count=1,
					vector_size=embedding_dimension, window=10, dm=0, dbow_words=1, workers=16, epochs=10)

	w2v_model.build_vocab(labeled_sentences)

	for epoch in range(20):
		start_time = time.time()
		write_log('W2V epoch {} : Start'.format(epoch))

		random.shuffle(labeled_sentences)
		w2v_model.train(labeled_sentences,
				total_examples=w2v_model.corpus_count,
				epochs=w2v_model.epochs)

		w2v_model.alpha -= 0.001
		w2v_model.min_alpha = w2v_model.alpha
		write_log('W2V epoch {} ends : tooks {}'.format(epoch, time.time() - start_time))

	w2v_model.save(model_path)

	dict_w2v = {}
	for url in  article_info.keys():
		dict_w2v[url] = w2v_model[url].tolist()
	dict_w2v['url_pad'] = [float(0)] * embedding_dimension

	write_log('W2V json dump : start')
	with open(output_path, 'w') as out_f:
		json.dump(dict_w2v, out_f)
	write_log('W2V json dump : end')

def generate_w2v_map_glob(pickle_path):
	global output_path, embedding_dimension

	with open(pickle_path, 'rb') as f_input:
		embedding_data = pickle.load(f_input)

	dict_w2v = {}
	for i in range(embedding_data.shape[0]):
		dict_w2v['url_{}'.format(i)] = embedding_data[i].tolist()

	dict_w2v['url_pad'] = [float(0)] * embedding_dimension

	with open(output_path, 'w') as out_f:
		json.dump(dict_w2v, out_f)

def main():
	global article_info_path, output_path, embedding_dimension, model_path

	options, args = parser.parse_args()
	if (options.output == None) or (options.input == None) or (options.model_path == None) \
						or (options.dataset == None) or (options.glob_pickle == None):
		return

	article_info_path = options.input
	output_path = options.output
	embedding_dimension = int(options.d2v_embed)
	model_path = options.model_path
	dataset = options.dataset
	glob_pickle_path = options.glob_pickle

	if dataset not in ['adressa', 'glob']:
		print('Wrong dataset name : {}'.format(dataset))
		return

	if dataset == 'glob' and embedding_dimension != 250:
		print('Glob only supports 250 dims')
		return

	if dataset == 'adressa':
		generate_w2v_map()
	else:
		generate_w2v_map_glob(glob_pickle_path)

if __name__ == '__main__':
	main()
