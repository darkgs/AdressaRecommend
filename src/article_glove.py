

import json, pickle
import time
import random

import pickle
import numpy as np
from glove import Corpus, Glove

from optparse import OptionParser

#from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-m', '--model_path', dest='model_path', type='string', default=None)
parser.add_option('-d', '--dataset', dest='dataset', type='string', default=None)
parser.add_option('-c', '--corpus_path', dest='corpus_path', type='string', default=None)

article_info_path = None
output_path = None
embedding_dimension = None
model_path = None


def write_log(log):
    print(log)


def generate_glove_map():
    global article_info_path, output_path, embedding_dimension, \
            model_path, corpus_path

    write_log('GloVe Load article info : Start')
    with open(article_info_path, 'r') as f_art:
        article_info = json.load(f_art)
    write_log('GloVe Load article info : End')

    write_log('GloVe Generate set of words : Start')
    words = set([])
    for url, dict_info in article_info.items():
        sentence_header = dict_info.get('sentence_header', None)
        sentence_body = dict_info.get('sentence_body', None)

        if (sentence_header == None) or (sentence_body == None):
            continue

        for sentence in sentence_header + sentence_body:
            for word in sentence.split(' '):
                words.update([word])

    write_log('GloVe Generate set of words - {}  : End'.format(len(words)))

    write_log('GloVe Load corpus from {}: Start'.format(corpus_path))
    corpus = Corpus.load(corpus_path)
    write_log('GloVe Load corpus : End')

    write_log('GloVe learning : Start')
    glove = Glove(no_components=embedding_dimension, learning_rate=0.05)
    glove.fit(corpus.matrix, epochs=1, no_threads=4, verbose=True)
    glove.add_dictionary(corpus.dictionary)
    write_log('GloVe learning : End')

    dict_a2g = {}
    for word in words:
        word_vector = np.array(glove.word_vectors[glove.dictionary[word]])
        assert(word_vector.shape[0] == embedding_dimension)
        dict_a2g[word] = word_vector

    write_log('GloVe result dump : Start')
    with open(output_path, 'wb') as f_out:
        pickle.dump(dict_a2g, f_out)
    write_log('GloVe result dump : End')

def main():
    global article_info_path, output_path, \
            embedding_dimension, model_path, corpus_path

    options, args = parser.parse_args()
    if (options.output == None) or (options.input == None) or \
            (options.model_path == None) or (options.dataset == None) or \
            (options.corpus_path == None):
        return

    article_info_path = options.input
    output_path = options.output
    embedding_dimension = int(options.d2v_embed)
    model_path = options.model_path
    dataset = options.dataset
    corpus_path = options.corpus_path

    if dataset not in ['adressa']:
        print('Wrong dataset name : {}'.format(dataset))
        return

    generate_glove_map()


if __name__ == '__main__':
    main()

