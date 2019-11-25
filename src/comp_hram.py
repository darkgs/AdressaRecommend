
import time
import os
import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from optparse import OptionParser

from adressa_dataset import AdressaRec
from ad_util import load_json
from ad_util import option2str

from ad_util import weights_init

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
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=1000)
parser.add_option('-b', '--num_layers', dest='num_layers', type='int', default=1)
parser.add_option('-c', '--embedding_dim', dest='embedding_dim', type='int', default=300)


class HRAMModel(nn.Module):
    def __init__(self, embed_size, cate_dim, args):
        super(HRAMModel, self).__init__()

        #hidden_size = args.hidden_size
        hidden_size = int(args.d2v_embed)
        num_layers = args.num_layers

        user_size = args.user_size
        article_size = args.article_size
        article_pad_idx = args.article_pad_idx

        assert((hidden_size % 2 == 0) and 'rnn hidden should be divisible by 2')

        self.rnn = nn.LSTM(embed_size, int(hidden_size/2), num_layers, 
                batch_first=True, bidirectional=True)
        self.rnn_mlp = nn.Linear(hidden_size, 64)

        self.embed_user = nn.Embedding(user_size, args.embedding_dim)
        self.embed_article = nn.Embedding(article_size, args.embedding_dim, padding_idx = article_pad_idx)

        self.embed_mlp = nn.Sequential(
                nn.Linear(args.embedding_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU())
        self.embed_mlp.apply(weights_init)

        self.attn = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False))
        self.attn.apply(weights_init)

        self.last_mlp = nn.Sequential(
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid())
                #nn.Linear(64, 1), nn.ReLU())
        self.last_mlp.apply(weights_init)

        self._rnn_hidden_size = hidden_size

    def forward(self, x1, _, cate, seq_lens, user_ids, sample_ids, sample_d2v_embeds):
        batch_size = x1.size(0)
        max_seq_length = x1.size(1)
        embed_size = x1.size(2)

        # generallized matrix factorization
        user_embed = self.embed_user(user_ids)
        sample_embed = self.embed_article(sample_ids)

        embed_step =  torch.unsqueeze(torch.unsqueeze(user_embed, dim=1), dim=1).expand(*sample_embed.shape) * sample_embed
        embed_step = self.embed_mlp(embed_step)     # [batch, seq, #sample, 64]

        # sequence embedding
        x1 = pack(x1, seq_lens, batch_first=True)
        x1, _ = self.rnn(x1)

        rnn_step = torch.zeros([max_seq_length, batch_size, self._rnn_hidden_size],
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        sequence_lenths = x1.batch_sizes.cpu().numpy()
        cursor = 0
        prev_x1s = []
        for step in range(sequence_lenths.shape[0]):
            sequence_lenth = sequence_lenths[step]

            x1_step = x1.data[cursor:cursor+sequence_lenth]

            prev_x1s.append(x1_step)
            prev_x1s = [prev_x1[:sequence_lenth] for prev_x1 in prev_x1s]

            prev_hs = torch.stack(prev_x1s, dim=1)

            attn_score = []
            for prev in range(prev_hs.size(1)):
                attn_input = prev_hs[:,prev,:]
                attn_score.append(self.attn(attn_input))
            attn_score = torch.softmax(torch.stack(attn_score, dim=1), dim=1)

            x_step = torch.squeeze(torch.bmm(torch.transpose(attn_score, 1, 2), prev_hs), dim=1)
            #x_step = torch.mean(prev_hs, dim=1, keepdim=False)
            #x_step = x1_step

            rnn_step[step][:sequence_lenth] = x_step

            cursor += sequence_lenth

        rnn_step = torch.transpose(rnn_step, 0, 1)
        rnn_step = torch.unsqueeze(rnn_step, dim=2).expand([-1,-1,sample_d2v_embeds.size(2),-1])
        rnn_step = rnn_step * sample_d2v_embeds
        rnn_step = self.rnn_mlp(rnn_step)

        step = torch.cat([embed_step, rnn_step], dim=3)
        step = torch.squeeze(self.last_mlp(step), dim=3)

        return step


def main():
    options, args = parser.parse_args()

    if (options.input == None) or (options.d2v_embed == None) or \
                       (options.u2v_path == None) or (options.ws_path == None):
        return

    torch_input_path = options.input
    embedding_dimension = int(options.d2v_embed)
    url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)
    ws_path = options.ws_path
    search_mode = options.search_mode
    model_ws_path = '{}/model/{}'.format(ws_path, option2str(options))

    if not os.path.exists(ws_path):
        os.system('mkdir -p {}'.format(ws_path))

#    os.system('rm -rf {}'.format(model_ws_path))
    os.system('mkdir -p {}'.format(model_ws_path))

    # Save best result with param name
    param_search_path = ws_path + '/param_search'
    if not os.path.exists(param_search_path):
        os.system('mkdir -p {}'.format(param_search_path))
    param_search_file_path = '{}/{}'.format(param_search_path, option2str(options))

    if search_mode and os.path.exists(param_search_file_path):
        print('Param search mode already exist : {}'.format(param_search_file_path))
        return

    print('Loading url2vec : start')
    dict_url2vec = load_json(url2vec_path)
    print('Loading url2vec : end')

    predictor = AdressaRec(HRAMModel, model_ws_path, torch_input_path, dict_url2vec, options, hram_mode=True)

    best_hit_5, best_auc_20, best_mrr_20 = predictor.do_train()

    if search_mode:
        with open(param_search_file_path, 'w') as f_out:
            f_out.write(str(best_hit_5) + '\n')
            f_out.write(str(best_auc_20) + '\n')
            f_out.write(str(best_mrr_20) + '\n')


if __name__ == '__main__':
    main()

