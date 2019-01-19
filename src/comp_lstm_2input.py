
import os, sys
import time

from optparse import OptionParser

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from adressa_dataset import AdressaRec

from ad_util import load_json
from ad_util import option2str

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)
parser.add_option('-s', action="store_true", dest='save_model', default=False)
parser.add_option('-z', action="store_true", dest='search_mode', default=False)

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=5)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=3)

parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)
#parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=786)
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=1440)


class SingleLSTM(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super(SingleLSTM, self).__init__()

		self._W_f = nn.Parameter(torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32), requires_grad=True)
		self._b_f = nn.Parameter(torch.zeros([hidden_size], dtype=torch.float32), requires_grad=True)

		self._W_i = nn.Parameter(torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32), requires_grad=True)
		self._b_i = nn.Parameter(torch.zeros([hidden_size], dtype=torch.float32), requires_grad=True)

		self._W_c = nn.Parameter(torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32), requires_grad=True)
		self._b_c = nn.Parameter(torch.zeros([hidden_size], dtype=torch.float32), requires_grad=True)

		self._W_o = nn.Parameter(torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32), requires_grad=True)
		self._b_o = nn.Parameter(torch.zeros([hidden_size], dtype=torch.float32), requires_grad=True)

		nn.init.xavier_normal_(self._W_f.data)
		nn.init.xavier_normal_(self._W_i.data)
		nn.init.xavier_normal_(self._W_c.data)
		nn.init.xavier_normal_(self._W_o.data)

	def forward(self, x, states):
		h_t, c_t = states

		# forget gate
		f_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x], 1), self._W_f) + self._b_f)

		# input gate
		i_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x], 1), self._W_i) + self._b_i)

		# cell candidate
		c_tilda = torch.tanh(torch.matmul(torch.cat([h_t, x], 1), self._W_c) + self._b_c)

		# new cell state
		c_t = f_t * c_t + i_t * c_tilda

		# out gate
		o_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x], 1), self._W_o) + self._b_o)
		
		# new hidden state
		h_t = o_t * torch.tanh(c_t)

		return h_t, (h_t, c_t)

#	def to(self, device):
#		ret = super(SingleLSTM, self).to(device)
#
#		self._W_f = self._W_f.to(device)
#		self._b_f = self._b_f.to(device)
#
#		self._W_i = self._W_i.to(device)
#		self._b_i = self._b_i.to(device)
#
#		self._W_c = self._W_c.to(device)
#		self._b_c = self._b_c.to(device)
#
#		self._W_o = self._W_o.to(device)
#		self._b_o = self._b_o.to(device)
#
#		return ret

class SingleLSTMModel(nn.Module):
	def __init__(self, embed_size, cate_dim, args):
		super(SingleLSTMModel, self).__init__()

		self._hidden_size = args.hidden_size
		attn_count = args.trendy_count + args.recency_count

		self.lstm = SingleLSTM(embed_size*2, self._hidden_size)
		self.linear = nn.Linear(self._hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

		self._W_attn = nn.Parameter(torch.zeros([self._hidden_size + embed_size * 2, 1]), requires_grad=True)
		nn.init.xavier_normal_(self._W_attn.data)

	def to(self, device):
		ret = super(SingleLSTMModel, self).to(device)

		self.lstm = self.lstm.to(device)
#self._W_attn = self._W_attn.to(device)

		self._device = device
		return ret

	def get_start_states(self, batch_size, hidden_size):
		h0 = torch.zeros(batch_size, hidden_size).to(self._device)
		c_0 = torch.zeros(batch_size, hidden_size).to(self._device)
		return (h0, c_0)

	def forward(self, x1, x2, _, seq_lens):
		batch_size = x1.size(0)
		max_seq_length = x1.size(1)
		embed_size = x1.size(2)

		x1 = pack(x1, seq_lens, batch_first=True)
		x2 = pack(x2, seq_lens, batch_first=True)
		outputs = torch.zeros([max_seq_length, batch_size, self._hidden_size])

		cursor = 0
		sequence_lenths = x1.batch_sizes.cpu().numpy()
		hx, cx = self.get_start_states(batch_size, self._hidden_size)
		for step in range(sequence_lenths.shape[0]):
			sequence_lenth = sequence_lenths[step]

			x1_step = x1.data[cursor:cursor+sequence_lenth]
			x2_step = x2.data[cursor:cursor+sequence_lenth]

			# Attention
			attn_count = x2_step.size(1)
			attn_scores = []
			for i in range(attn_count):
				attn_scores.append(torch.matmul(torch.cat([hx[:sequence_lenth], \
								x1_step, x2_step[:,i,:]], 1), self._W_attn))
			attn_scores = torch.softmax(torch.cat(attn_scores, dim=1), dim=1)
			attn_scores = torch.unsqueeze(attn_scores, dim=1)
			x2_step = torch.squeeze(torch.bmm(attn_scores, x2_step), dim=1)

			x_step = torch.cat([x1_step, x2_step], 1)

			hx, (_, cx) = self.lstm(x_step, \
					(hx[:sequence_lenth], cx[:sequence_lenth]))
			outputs[step][:sequence_lenth] = hx

			cursor += sequence_lenth

		outputs = torch.transpose(outputs, 1, 0).to(self._device)
		outputs = self.linear(outputs)

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

#os.system('rm -rf {}'.format(model_ws_path))
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

	predictor = AdressaRec(SingleLSTMModel, ws_path, torch_input_path, dict_url2vec, options)
	best_hit_5, best_auc_20, best_mrr_20 = predictor.do_train()

	if search_mode:
		with open(param_search_file_path, 'w') as f_out:
			f_out.write(str(best_hit_5) + '\n')
			f_out.write(str(best_auc_20) + '\n')
			f_out.write(str(best_mrr_20) + '\n')


if __name__ == '__main__':
	main()

