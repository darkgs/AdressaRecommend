
import os, sys
import time
import pickle

from optparse import OptionParser

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from adressa_dataset import AdressaRec
from dataset.selections import SelectRec

from ad_util import load_json
from ad_util import option2str

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)
parser.add_option('-s', action="store_true", dest='save_model', default=False)
parser.add_option('-z', action="store_true", dest='search_mode', default=False)

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=1)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=1)

parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)
parser.add_option('-g', '--word_embed_path', dest='word_embed_path', type='string', default=None)

parser.add_option('-a', '--word_dim', dest='word_dim', type='int', default=1000)
parser.add_option('-b', '--num_prev_watch', dest='num_prev_watch', type='int', default=5)


class SimpleAVGModel(nn.Module):
    def __init__(self, options):
        super(__class__, self).__init__()

        embed_dim = int(options.d2v_embed)
        num_prev_watch = options.num_prev_watch

        self.mlp = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: [batch, num_prev_watch, embed_size]

        step = x
        step =torch.mean(step, 1, keepdim=False)
        step = self.mlp(step)

        # output: [batch, embed_size]
        return step


def main():
    options, args = parser.parse_args()

    if (options.input == None) or (options.d2v_embed == None) or \
            (options.u2v_path == None) or (options.ws_path == None) or \
            (options.word_embed_path == None):
        return

    path_rec_input = '{}/torch_rnn_input.dict'.format(options.input)
    embedding_dimension = int(options.d2v_embed)
    path_url2vec = '{}_{}'.format(options.u2v_path, embedding_dimension)

    sr = SelectRec(path_rec_input, path_url2vec, SimpleAVGModel, options)
    sr.do_train(total_epoch=1)


if __name__ == '__main__':
    main()

