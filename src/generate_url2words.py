

import json, pickle
import time
import random

import numpy as np

from optparse import OptionParser

#from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-g', '--word2vec', dest='word2vec', type='string', default=None)

article_info_path = None
output_path = None
embedding_dimension = None
dict_word2vec = None


def write_log(log):
    print(log)


def generate_url2words():
    global article_info_path, output_path, dict_word2vec

    write_log('url2words Load article info : Start')
    with open(article_info_path, 'r') as f_art:
        article_info = json.load(f_art)
    write_log('url2words Load article info : End')

    write_log('url2words Generate the set of words : Start')
    words = set([])
    for url, dict_info in article_info.items():
        sentence_header = dict_info.get('sentence_header', None)
        sentence_body = dict_info.get('sentence_body', None)

        if (sentence_header == None) or (sentence_body == None):
            continue

        #for sentence in sentence_header + sentence_body:
        for sentence in sentence_header:
            for word in sentence.split(' '):
                words.update([word])
    write_log('url2words Generate the set of words - {}: End'.format(len(words)))

    dict_word2idx = {word:idx for idx, word in enumerate(words)}

    dict_word_idx2vec = {idx: np.array(dict_word2vec[word]) for word, idx in dict_word2idx.items()}

    dict_url2words = {}
    dict_url2words['url2word_idx'] = {}
    dict_url2words['word_idx2vec'] = dict_word_idx2vec

    write_log('url2words Generate url2words : Start')
    for url, dict_info in article_info.items():
        sentence_header = dict_info.get('sentence_header', None)
        sentence_body = dict_info.get('sentence_body', None)

        if (sentence_header == None) or (sentence_body == None):
            continue

        words_seq = []
        for sentence in sentence_header:
            words_seq += [dict_word2idx[word] for word in sentence.split(' ')]

        dict_url2words['url2word_idx'][url] = words_seq
    write_log('url2words Generate url2words : End')

    with open(output_path, 'wb') as f_out:
        pickle.dump(dict_url2words, f_out)


def main():
    global article_info_path, output_path, \
            embedding_dimension, dict_word2vec

    options, args = parser.parse_args()
    if (options.output == None) or (options.input == None) or \
            (options.d2v_embed == None) or (options.word2vec == None):
        return

    article_info_path = options.input
    output_path = options.output
    embedding_dimension = options.d2v_embed

    with open(options.word2vec, 'rb') as f_w2v:
        dict_word2vec = pickle.load(f_w2v)

    generate_url2words()


if __name__ == '__main__':
    main()

