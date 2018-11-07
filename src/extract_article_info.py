
import os, sys
import json

from optparse import OptionParser

from multiprocessing.pool import ThreadPool

from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-d', '--url2id', dest='url2id', type='string', default=None)

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
	if (options.output == None) or (options.url2id == None) or (options.input == None):
		return

	contentdata_path = options.input
	out_path = options.output
	url2id_path = options.url2id

	with open(url2id_path, 'r') as f_dict:
		dict_url2id = json.load(f_dict)

	write_log('Starting threads')
	dict_article_info = {}
	with ThreadPool(8) as pool:
		pool.map(extract_article_info, dict_url2id.items())
	write_log('Thread works done')

	write_log('Save to {}'.format(out_path))
	with open(out_path, 'w') as f_json:
		json.dump(dict_article_info, f_json)
	write_log('Done')


if __name__ == '__main__':
	main()
