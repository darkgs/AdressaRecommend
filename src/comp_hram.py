
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
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=1024)
parser.add_option('-b', '--num_layers', dest='num_layers', type='int', default=1)
parser.add_option('-c', '--embedding_dim', dest='embedding_dim', type='int', default=300)


class HRAMModel(nn.Module):
    def __init__(self, embed_size, cate_dim, args):
        super(HRAMModel, self).__init__()

        hidden_size = args.hidden_size
        num_layers = args.num_layers

        user_size = args.user_size
        article_size = args.article_size
        article_pad_idx = args.article_pad_idx

        assert((hidden_size % 2 == 0) and 'rnn hidden should be divisible by 2')

        self.rnn = nn.LSTM(embed_size, int(hidden_size/2), num_layers, 
                batch_first=True, bidirectional=True)
        self.rnn_mlp = nn.Linear(hidden_size, embed_size)

        self.embed_user = nn.Embedding(user_size, args.embedding_dim)
        self.embed_article = nn.Embedding(article_size, args.embedding_dim, padding_idx = article_pad_idx)

        #self._W_attn = nn.Parameter(torch.zeros([hidden_size*2, 1], dtype=torch.float32), requires_grad=True)
        #self._b_attn = nn.Parameter(torch.zeros([1], dtype=torch.float32), requires_grad=True)

        #self._mha = nn.MultiheadAttention(embed_size, 20 if (embed_size % 20) == 0 else 10)
        #self._mlp_mha = nn.Linear(embed_size, hidden_size)

        #nn.init.xavier_normal_(self._W_attn.data)

        self._rnn_hidden_size = hidden_size

    def forward(self, x1, _, y, seq_lens, user_ids, article_ids):
        batch_size = x1.size(0)
        max_seq_length = x1.size(1)
        embed_size = x1.size(2)

        outputs = torch.zeros([max_seq_length, batch_size, embed_size],
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # user embedding
#        print('user embed', user_ids.shape, self.embed_user(user_ids).shape)
#        print('article embed', article_ids.shape, self.embed_article(article_ids).shape)

        # sequence embedding
        x1 = pack(x1, seq_lens, batch_first=True)
        x1, _ = self.rnn(x1)

        sequence_lenths = x1.batch_sizes.cpu().numpy()
        cursor = 0
        prev_x1s = []
        for step in range(sequence_lenths.shape[0]):
            sequence_lenth = sequence_lenths[step]

            x1_step = x1.data[cursor:cursor+sequence_lenth]
            x_step = self.rnn_mlp(x1_step)
            outputs[step][:sequence_lenth] = x_step
            continue

            prev_x1s.append(x1_step)

            prev_x1s = [prev_x1[:sequence_lenth] for prev_x1 in prev_x1s]

            prev_hs = torch.stack(prev_x1s, dim=1)
            attn_score = []
            for prev in range(prev_hs.size(1)):
                attn_input = torch.cat((prev_hs[:,prev,:], x2_step), dim=1)
                attn_score.append(torch.matmul(attn_input, self._W_attn) + self._b_attn)
            attn_score = torch.softmax(torch.stack(attn_score, dim=1), dim=1)
            x1_step = torch.squeeze(torch.bmm(torch.transpose(attn_score, 1, 2), prev_hs), dim=1)
            
            x_step = torch.cat((x1_step, x2_step), dim=1)
            x_step = self.mlp(x_step)

            outputs[step][:sequence_lenth] = x_step

            cursor += sequence_lenth

        outputs = torch.transpose(outputs, 0, 1)

        return outputs


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

    predictor = AdressaRec(HRAMModel, model_ws_path, torch_input_path, dict_url2vec, options)

    best_hit_5, best_auc_20, best_mrr_20 = predictor.do_train()

    if search_mode:
        with open(param_search_file_path, 'w') as f_out:
            f_out.write(str(best_hit_5) + '\n')
            f_out.write(str(best_auc_20) + '\n')
            f_out.write(str(best_mrr_20) + '\n')


if __name__ == '__main__':
    main()

