
import sys
import json
reload(sys)
sys.setdefaultencoding('utf-8')

from optparse import OptionParser

import pymongo
from pymongo import MongoClient

from ad_util import write_log

parser = OptionParser()
parser.add_option('-o', '--output', dest='output', type='string', default=None)

out_dir = None

def extract_article_content():
	global out_dir

	write_log('Extract article content : Start')
	write_log('MongoDB query')
	collection = MongoClient('darkgs.tplinkdns.com', 27017).adressa.article_copy

	cursor = collection.find(
		{
			'$and': [
				{
					'proper': True
				},
				{
					'sentence_header': {'$exists': True}
				},
				{
					'sentence_body': {'$exists': True}
				},
			]
		}
	)
	write_log('MongoDB query End')

	article_content = {}

	total_count = cursor.count()
	count = 0
	for doc in cursor:
		if count % 1000 == 0:
			write_log('Extracting : {}/{}'.format(count, total_count))
		count += 1

		url = doc.get('url', None)
		sentence_header = doc.get('sentence_header', [])
		sentence_body = doc.get('sentence_body', [])
		words_header = doc.get('words_header', [])
		words_body = doc.get('words_body', [])

		if url == None:
			continue

		article_content[url] = {
			'sentence_header': sentence_header,
			'sentence_body': sentence_body,
			'words_header': words_header,
			'words_body': words_body,
		}

	write_log('Save to Json : start')
	with open(out_dir, 'w') as f_json:
		json.dump(article_content, f_json)
	write_log('Save to Json : end')


def main():
	global out_dir

	options, args = parser.parse_args()
	if options.output == None:
		return

	out_dir = options.output

	extract_article_content()


if __name__ == '__main__':
	main()
