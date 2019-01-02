
import os, sys
import json

from optparse import OptionParser

from multiprocessing.pool import ThreadPool

from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-u', '--url2id', dest='url2id', type='string', default=None)
parser.add_option('-d', '--dataset', dest='dataset', type='string', default=None)
parser.add_option('-g', '--glob_meta', dest='glob_meta', type='string', default=None)

contentdata_path = None
dict_article_info = {}

def extract_article_info(args):
	global contentdata_path, dict_article_info

	url, article_id = args

	data_path = contentdata_path + '/' + article_id 
	if not os.path.exists(data_path):
		return

	with open(data_path, 'r') as f_data:
		data_lines = f_data.readlines()

	category0 = None
	category1 = None

	for line in data_lines:
		if category0 != None and category1 != None:
			break

		line_json = json.loads(line.strip())
		if line_json == None:
			continue

		for dict_field in line_json.get('fields', []):
			if dict_field.get('field', '') == 'category0':
				category0 = dict_field.get('value', None)
			elif dict_field.get('field', '') == 'category1':
				splited_value = dict_field.get('value', '').split('|')
				category1 = splited_value[1] if len(splited_value) > 1 else None

	dict_article_info[url] = {
		'category0': category0,
		'category1': category1,
	}

def main():
	global contentdata_path, dict_article_info

	options, args = parser.parse_args()
	if (options.output == None) or (options.url2id == None) or (options.input == None) \
						or (options.dataset == None) or (options.glob_meta == None):
		return

	contentdata_path = options.input
	out_path = options.output
	url2id_path = options.url2id
	dataset = options.dataset
	glob_meta_path = options.glob_meta

	if dataset not in ['adressa', 'glob']:
		print('Wrong dataset name : {}'.format(dataset))
		return

	dict_article_info = {}

	if dataset == 'adressa':
		with open(url2id_path, 'r') as f_dict:
			dict_url2id = json.load(f_dict)

		write_log('Starting threads')

		with ThreadPool(8) as pool:
			pool.map(extract_article_info, dict_url2id.items())
		write_log('Thread works done')

	elif dataset == 'glob':
		with open(glob_meta_path, 'r') as f_meta:
			lines = f_meta.readlines()

		dict_header_idx = None
		for line in lines:
			line = line.strip()

			if dict_header_idx == None:
				dict_header_idx = {}
				for i, k in enumerate(line.split(',')):
					dict_header_idx[k] = i
				continue

			line_split = line.split(',')
			url = 'url_{}'.format(line_split[dict_header_idx['article_id']])
			category_id = 'cate_{}'.format(line_split[dict_header_idx['category_id']])

			dict_article_info[url] = {
				'category0': category_id,
			}

	write_log('Save to {}'.format(out_path))
	with open(out_path, 'w') as f_json:
		json.dump(dict_article_info, f_json)
	write_log('Done')


if __name__ == '__main__':
	main()
