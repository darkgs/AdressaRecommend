
import os, sys

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
parser.add_option('-c', '--u2i_path', dest='u2i_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)
parser.add_option('-s', action="store_true", dest='save_model', default=False)
parser.add_option('-z', action="store_true", dest='search_mode', default=False)

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=1)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=1)

parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=896)
parser.add_option('-d', '--decay_rate', dest='decay_rate', type='float', default=0.5)

class NaverModel(nn.Module):
	def __init__(self, embed_size, cate_dim, args):
		super(NaverModel, self).__init__()

		self._hidden_size = args.hidden_size
		self._cate_dim = cate_dim
		self._cate_decay_rate = args.decay_rate

		self.lstm = nn.LSTMCell(embed_size, self._hidden_size)
		self.linear = nn.Linear(self._hidden_size+self._cate_dim, embed_size)
#self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def get_start_states(self, batch_size, hidden_size):
		h0 = torch.zeros(batch_size, hidden_size).to(self._device)
		c_0 = torch.zeros(batch_size, hidden_size).to(self._device)
		return (h0, c_0)

	def to(self, device):
		ret = super(NaverModel, self).to(device)

		self._device = device
		return ret

	def forward(self, x, _, cate, seq_lens):
		batch_size = x.size(0)
		max_seq_length = x.size(1)
		embed_size = x.size(2)

		x = pack(x, seq_lens, batch_first=True)
		cate = pack(cate, seq_lens, batch_first=True)
		outputs = torch.zeros([max_seq_length, batch_size, self._hidden_size+self._cate_dim])

		cursor = 0
		sequence_lenths = x.batch_sizes.cpu().numpy()
		hx, cx = self.get_start_states(batch_size, self._hidden_size)

		for step in range(sequence_lenths.shape[0]):
			sequence_lenth = sequence_lenths[step]

			x_step = x.data[cursor:cursor+sequence_lenth]
			cate_step = cate.data[cursor:cursor+sequence_lenth]
			if step > 0:
				cate_step = prev_cate[:sequence_lenth] * self._cate_decay_rate + \
							cate_step * (1.0 - self._cate_decay_rate)

			hx, cx = self.lstm(x_step, (hx[:sequence_lenth], cx[:sequence_lenth]))
			outputs[step][:sequence_lenth] = torch.cat([hx, cate_step], 1)

			cursor += sequence_lenth
			prev_cate = cate_step

		outputs = torch.transpose(outputs, 1, 0).to(self._device)
		outputs = self.linear(outputs)

#		outputs, _ = self.rnn(x)
#		outputs, _ = unpack(outputs, batch_first=True)
#		outputs = self.linear(outputs)

		return outputs

#		outputs = outputs.view(-1, embed_size)
#		outputs = self.bn(outputs)
#		outputs = outputs.view(batch_size, -1, embed_size)
#
#		return outputs


def main():
	options, args = parser.parse_args()

	if (options.input == None) or (options.d2v_embed == None) or \
					   (options.u2v_path == None) or (options.u2i_path == None) or \
					   (options.ws_path == None):
		return

	torch_input_path = options.input
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)
	url2info_path = options.u2i_path
	ws_path = options.ws_path
	search_mode = options.search_mode
	model_ws_path = '{}/model/{}'.format(ws_path, option2str(options))

	if not os.path.exists(ws_path):
		os.system('mkdir -p {}'.format(ws_path))

	os.system('rm -rf {}'.format(model_ws_path))
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

	print('Loading url2info : start')
	dict_url2info = load_json(url2info_path)
	print('Loading url2info : end')

	predictor = AdressaRec(NaverModel, ws_path, torch_input_path, 
			dict_url2vec, options, dict_url2info=dict_url2info)
	best_hit_5, best_auc_10, best_auc_20, best_mrr_5, best_mrr_20 = predictor.do_train()

	if search_mode:
		with open(param_search_file_path, 'w') as f_out:
			f_out.write(str(best_hit_5) + '\n')
			f_out.write(str(best_auc_10) + '\n')
			f_out.write(str(best_auc_20) + '\n')
			f_out.write(str(best_mrr_5) + '\n')
			f_out.write(str(best_mrr_20) + '\n')


if __name__ == '__main__':
	main()

