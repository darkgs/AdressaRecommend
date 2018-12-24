
import json
import random

import numpy as np

from optparse import OptionParser

from ad_util import load_json

parser = OptionParser()
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-a', '--u2i_path', dest='u2i_path', type='string', default=None)

def generate_rnn_input(dict_url2info, dict_url2vec):
	# 38,155 items has category in the total 51,418
	# 44 category variation
	# dict_url2info contains only existing url in the sequence
	dict_datas = {}
	for url, dict_info in dict_url2info.items():
		category = dict_info.get("category0", None)

		if dict_url2vec.get(url, None) == None:
			continue

		if category == None:
			continue

		dict_datas[category] = dict_datas.get(category, [])
		dict_datas[category].append(url)

	train_datas = {}
	test_datas = {}

	for category, datas in dict_datas.items():
		if len(datas) < 20:
			train_datas[category] = datas
			continue

		random.shuffle(datas)

		test_datas[category] = datas[:len(datas)//10]
		train_datas[category] = datas[len(test_datas[category]):]

	def generate_triples(target_datas):
		# full combination of same categories
		# There are 173,025,765 combinations in the train set
		# There are 2,135,504 combinations in the test set
		url_triples = []
		for category, urls in target_datas.items():
			another_urls = []
			for k, v in target_datas.items():
				if category == k:
					continue
				another_urls += v

			if len(another_urls) <= 0:
				 continue

			for i in range(len(urls)):
				for j in range(i+1, len(urls)):
					for _ in range(10):
						url_triples.append((urls[i], urls[j], another_urls[random.randrange(len(another_urls))]))

		url_triples = np.random.permutation(np.array(url_triples))
		maximum_count = 20000000
		if len(url_triples) > maximum_count:
			url_triples = url_triples[:maximum_count]

		return url_triples.tolist()

	dict_rnn_input = {
		'train': generate_triples(train_datas),
		'test': generate_triples(test_datas),
	}

	return dict_rnn_input

def main():
	options, args = parser.parse_args()

	if (options.u2i_path == None) or (options.output == None):
		return

	output_file_path = options.output
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)
	url2info_path = options.u2i_path

	print('Loading url2vec : start')
	dict_url2vec = load_json(url2vec_path)
	print('Loading url2vec : end')

	print('Loading url2info : start')
	dict_url2info = load_json(url2info_path)
	print('Loading url2info : end')

	dict_rnn_input = generate_rnn_input(dict_url2info, dict_url2vec)

	with open(output_file_path, 'w') as f_out:
		json.dump(dict_rnn_input, f_out)

if __name__ == '__main__':
	main()

