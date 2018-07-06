
import json
import time
import random

import gensim
from optparse import OptionParser

from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)

article_info_path = None
output_path = None

def generate_w2v_map():
	global article_info_path, output_path

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
					vector_size=200, window=10, dm=0, dbow_words=1, workers=16, epochs=10)

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

	dict_w2v = {}
	for url in  article_info.keys():
		dict_w2v[url] = w2v_model[url].tolist()

	write_log('W2V json dump : start')
	with open(output_path, 'w') as out_f:
		json.dump(dict_w2v, out_f)
	write_log('W2V json dump : end')

def main():
	global article_info_path, output_path

	options, args = parser.parse_args()
	if (options.output == None) or (options.input == None):
		return

	article_info_path = options.input
	output_path = options.output

	generate_w2v_map()

if __name__ == '__main__':
	main()
