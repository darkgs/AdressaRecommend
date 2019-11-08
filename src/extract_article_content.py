
import os, sys
import re

import json
import pickle

from optparse import OptionParser

import pymongo
from pymongo import MongoClient

from ad_util import write_log
from ad_util import find_best_url

parser = OptionParser()
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-d', '--dataset', dest='dataset', type='string', default=None)
parser.add_option('-g', '--glob_pickle', dest='glob_pickle', type='string', default=None)
parser.add_option('-a', '--adressa_content', dest='adressa_content', type='string', default=None)

out_dir = None

def extract_article_content(content_dir):

    target_files = []

    for file_name in os.listdir(content_dir):
        file_path = os.path.join(content_dir, file_name)

        if not os.path.isfile(file_path):
            continue

        target_files.append(file_path)

    output = {}
    for file_idx, file_path in enumerate(target_files):
        lines = []
        with open(file_path, 'r') as f_con:
            lines = [line.strip() for line in f_con.readlines() if len(line.strip()) > 0]

        for line in lines:
            try:
                dict_cont = json.loads(line)
            except:
                print('Error: {}'.format(line))
                continue

            dict_data = {}

            for field in dict_cont.get('fields', []):
                field_name = field.get('field', None)
                field_value = field.get('value', None)

                if not field_name or not field_value:
                    continue

                if field_name not in ['url', 'cannonicalUrl', 'referrerUrl', 
                        'title', 'body',
                        'category0', 'category1']:
                    continue

                dict_data[field_name] = field_value

            # find the best URL
            best_url = find_best_url(dict_data)
            if not best_url:
                continue

            for key in ['url', 'cannonicalUrl', 'referrerUrl']:
                dict_data.pop(key, None)

            # preprocess title & body
            if ('title' not in dict_data) or ('body' not in dict_data):
                continue

            def preprocess_sentence(sentences):
                new_sentences = []
                regex_remove = re.compile('[\'|\"|,|\-|\\.| |\?|«|»|:|!|–|@|\\(|\\)|−]+')
                for sentence in sentences:
                    sentence = re.sub(regex_remove, ' ', sentence)
                    new_sentences.append(sentence.strip())
                return new_sentences

            dict_data['sentence_header'] = preprocess_sentence([dict_data['title']])
            dict_data['sentence_body'] = preprocess_sentence(dict_data['body'])

            for key in ['title', 'body']:
                dict_data.pop(key, None)

            output[best_url] = dict_data

    write_log('Save to Json : start')
    with open(out_dir, 'w') as f_json:
        json.dump(output, f_json)
    write_log('Save to Json : end')


def extract_article_content_glob(pickle_path):
    global out_dir

    with open(pickle_path, 'rb') as f_input:
        article_id_count = pickle.load(f_input).shape[0]

    article_content = {}

    for i in range(article_id_count):
        article_content['url_{}'.format(i)] = {
            'sentence_header': [],
            'sentence_body': [],
            'words_header': [],
            'words_body': [],
        }

    with open(out_dir, 'w') as f_json:
        json.dump(article_content, f_json)

def main():
    global out_dir

    options, args = parser.parse_args()
    if (options.output == None) or (options.dataset == None) or \
            (options.glob_pickle == None) or (options.adressa_content == None):
        return

    out_dir = options.output
    dataset = options.dataset
    glob_pickle_path = options.glob_pickle
    adressa_content_path = options.adressa_content

    if dataset not in ['adressa', 'glob']:
        print('Wrong dataset name : {}'.format(dataset))
        return

    if dataset == 'adressa':
        extract_article_content(adressa_content_path)
    else:
        extract_article_content_glob(glob_pickle_path)


if __name__ == '__main__':
    main()
