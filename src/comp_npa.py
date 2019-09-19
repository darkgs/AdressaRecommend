
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

parser.add_option('-a', '--num_words', dest='num_words', type='int', default=20)
parser.add_option('-b', '--num_prev_watch', dest='num_prev_watch', type='int', default=5)


class SimpleAVGModel(nn.Module):
    def __init__(self, options):
        super(__class__, self).__init__()

        embed_dim = int(options.d2v_embed)
        num_prev_watch = options.num_prev_watch

        self.mlp = nn.Linear(embed_dim, embed_dim)

        self.news_cnn = nn.Conv2d(1, 300, [3, embed_dim], stride=1, padding=[1, 0])
        self.cnn_relu = nn.ReLU()

    def news_encoder(self, words):
        step = words

        ##### cnn layer
        # step: [batch, num_words, embed_dim]
        step = torch.unsqueeze(step, dim=1)
        # step: [batch, 1, num_words, embed_dim]
        step = self.cnn_relu(self.news_cnn(step))
        # step: [batch, 300, num_words, 1]
        step = torch.transpose(torch.squeeze(step, dim=3), 2, 1)
        # step: [batch, num_words, 300]

        ##### personal attention
        step = torch.mean(step, 1, keepdim=False)

        return step

    def forward(self, x, y, z):
        # x: [batch, num_prev_watch, word_count, word_embed_size]
        prev_watches = []
        for i in range(x.shape[1]):
            prev_watches.append(self.news_encoder(x[:,i,:,:]))
        prev_watches = torch.stack(prev_watches, dim=1)

        prev_watches = torch.mean(prev_watches, 1, keepdim=False)
        prev_watches = self.mlp(prev_watches)

        # y: [batch, word_count, word_embed_size]
        target_embed = self.news_encoder(y)

        # z: [batch, num_candidate, word_count, word_embed_size]
        candidate_embed = []
        for i in range(z.shape[1]):
            candidate_embed.append(self.news_encoder(z[:,i,:,:]))
        candidate_embed = torch.stack(candidate_embed, dim=1)

        # output: [batch, embed_size]
        return prev_watches, target_embed, candidate_embed


class SingleLSTMModel(nn.Module):
    def __init__(self, embed_size, cate_dim, args):
        super(SingleLSTMModel, self).__init__()

        hidden_size = args.hidden_size
        num_layers = args.num_layers

        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, x, _, __, seq_lens):
        batch_size = x.size(0)
        embed_size = x.size(2)

        x = pack(x, seq_lens, batch_first=True)
        outputs, _ = self.rnn(x)
        outputs, _ = unpack(outputs, batch_first=True)
        outputs = self.linear(outputs)

        return outputs

#        outputs = outputs.view(-1, embed_size)
#        outputs = self.bn(outputs)
#        outputs = outputs.view(batch_size, -1, embed_size)
#
#        return outputs


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
    return

    torch_input_path = options.input
    embedding_dimension = int(options.d2v_embed)
    url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)
    ws_path = options.ws_path
    search_mode = options.search_mode
    model_ws_path = '{}/model/{}'.format(ws_path, option2str(options))

    if not os.path.exists(ws_path):
        os.system('mkdir -p {}'.format(ws_path))

#os.system('rm -rf {}'.format(model_ws_path))
    os.system('mkdir -p {}'.format(model_ws_path))

    print('Loading url2vec : start')
    dict_url2vec = load_json(url2vec_path)
    print('Loading url2vec : end')

    print('Loading glove : start')
    with open(options.glove, 'rb') as f_glove:
        dict_glove = pickle.load(f_glove)
    print('Loading glove : end')

    predictor = AdressaRec(SingleLSTMModel, ws_path, torch_input_path, dict_url2vec, options,
            dict_glove=dict_glove)

    best_hit_5, best_auc_20, best_mrr_20 = predictor.do_train()


if __name__ == '__main__':
    main()

