

import json, pickle
import time
import random

import numpy as np
from glove import Corpus, Glove

from optparse import OptionParser

#from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)

article_info_path = None
output_path = None


def write_log(log):
    print(log)


def generate_glove_corpus():
    global article_info_path, output_path

    write_log('GloVe Load article info : Start')
    with open(article_info_path, 'r') as f_art:
        article_info = json.load(f_art)
    write_log('GloVe Load article info : End')

    write_log('GloVe Generate sentences : Start')
    sentences = []
    for url, dict_info in article_info.items():
        sentence_header = dict_info.get('sentence_header', None)
        sentence_body = dict_info.get('sentence_body', None)

        if (sentence_header == None) or (sentence_body == None):
            continue

        words = []
        for sentence in sentence_header + sentence_body:
            for word in sentence.split(' '):
                words.append(word)

        sentences.append(words)
    write_log('GloVe Generate sentences : End')

    write_log('GloVe Generate corpus : Start')
    corpus = Corpus()
    corpus.fit(sentences, window=10)
    write_log('GloVe Generate corpus : End')

    corpus.save(output_path)


def main():
    global article_info_path, output_path

    options, args = parser.parse_args()
    if (options.output == None) or (options.input == None):
        return

    article_info_path = options.input
    output_path = options.output

    generate_glove_corpus()


if __name__ == '__main__':
    main()

