
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

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=5)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=3)

parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=1440)
parser.add_option('-d', '--x2_dropout_rate', dest='x2_dropout_rate', type='float', default=0.3)


class MultiCellLSTM(nn.Module):
	def __init__(self, embed_size, hidden_size, attn_count, x2_dropout_rate):
		super(MultiCellLSTM, self).__init__()

		self._x2_dropout_rate = x2_dropout_rate

		self._W1_f = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b1_f = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)
		self._W2_f = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b2_f = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W1_i = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b1_i = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)
		self._W2_i = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b2_i = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W1_c = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b1_c = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)
		self._W2_c = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b2_c = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W1_o = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b1_o = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)
		self._W2_o = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b2_o = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W_attn = torch.zeros([hidden_size + embed_size * 2, 1])

		nn.init.xavier_normal_(self._W1_f.data)
		nn.init.xavier_normal_(self._W1_i.data)
		nn.init.xavier_normal_(self._W1_c.data)
		nn.init.xavier_normal_(self._W1_o.data)

		nn.init.xavier_normal_(self._W2_f.data)
		nn.init.xavier_normal_(self._W2_i.data)
		nn.init.xavier_normal_(self._W2_c.data)
		nn.init.xavier_normal_(self._W2_o.data)

		nn.init.xavier_normal_(self._W_attn.data)

		self.training = True

	def train(self, mode=True):
		self.training = mode

	def eval(self):
		self.train(False)

	def to(self, device):
		ret = super(MultiCellLSTM, self).to(device)

		self._W1_f = self._W1_f.to(device)
		self._b1_f = self._b1_f.to(device)
		self._W2_f = self._W2_f.to(device)
		self._b2_f = self._b2_f.to(device)

		self._W1_i = self._W1_i.to(device)
		self._b1_i = self._b1_i.to(device)
		self._W2_i = self._W2_i.to(device)
		self._b2_i = self._b2_i.to(device)

		self._W1_c = self._W1_c.to(device)
		self._b1_c = self._b1_c.to(device)
		self._W2_c = self._W2_c.to(device)
		self._b2_c = self._b2_c.to(device)

		self._W1_o = self._W1_o.to(device)
		self._b1_o = self._b1_o.to(device)
		self._W2_o = self._W2_o.to(device)
		self._b2_o = self._b2_o.to(device)

		self._W_attn = self._W_attn.to(device)

		return ret

	def forward(self, x1, x2, states):
		h_t, c1_t, c2_t = states

		# Attention
		attn_count = x2.size(1)
		attn_scores = []
		for i in range(attn_count):
			attn_scores.append(torch.matmul(torch.cat([h_t, x1, x2[:,i,:]], 1), self._W_attn))
		attn_scores = torch.softmax(torch.cat(attn_scores, dim=1), dim=1)
		attn_scores = torch.unsqueeze(attn_scores, dim=1)
		x2 = torch.squeeze(torch.bmm(attn_scores, x2), dim=1)

		#x2 = x1

		# forget gate
		f1_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x1], 1), self._W1_f) + self._b1_f)
		f2_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x2], 1), self._W2_f) + self._b2_f)

		# input gate
		i1_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x1], 1), self._W1_i) + self._b1_i)
		i2_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x2], 1), self._W2_i) + self._b2_i)

		# cell candidate
		c1_tilda = torch.tanh(torch.matmul(torch.cat([h_t, x1], 1), self._W1_c) + self._b1_c)
		c2_tilda = torch.tanh(torch.matmul(torch.cat([h_t, x2], 1), self._W2_c) + self._b2_c)

		# new cell state
		c1_t = f1_t * c1_t + i1_t * c1_tilda
		c2_t = f2_t * c2_t + i2_t * c2_tilda

		# out gate without sigmoid
		o1_t = torch.matmul(torch.cat([h_t, x1], 1), self._W1_o) + self._b1_o
		o2_t = torch.matmul(torch.cat([h_t, x2], 1), self._W2_o) + self._b2_o

		do_softmax = True
		if do_softmax:
			softmax_sum = torch.exp(o1_t) + torch.exp(o2_t)
			if self.training and random.random() < self._x2_dropout_rate:
				h_t = torch.sigmoid(o1_t) * torch.tanh(c1_t)
			else:
				h_t = torch.exp(o1_t) * torch.tanh(c1_t) / softmax_sum \
					  + torch.exp(o2_t) * torch.tanh(c2_t) / softmax_sum
		else:
			o1_t = torch.sigmoid(o1_t)
			o2_t = torch.sigmoid(o2_t)

			if self.training and random.random() < self._x2_dropout_rate:
				h_t = o1_t * torch.tanh(c1_t)
			else:
				h_t = torch.tanh(o1_t * torch.tanh(c1_t) + o2_t * torch.tanh(c2_t))

		return h_t, (h_t, c1_t, c2_t)


class MultiCellModel(nn.Module):
	def __init__(self, embed_size, cate_dim, args):
		super(MultiCellModel, self).__init__()

		self._hidden_size = args.hidden_size
		attn = args.trendy_count + args.recency_count

		self.lstm = MultiCellLSTM(embed_size, self._hidden_size, attn, args.x2_dropout_rate)
#self.dropout = nn.Dropout(0.3)
		self.linear = nn.Linear(self._hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def train(self, mode=True):
		super(MultiCellModel, self).train(mode)
		self.lstm.train(mode)

	def eval(self):
		super(MultiCellModel, self).eval()
		self.lstm.eval()

	def to(self, device):
		ret = super(MultiCellModel, self).to(device)

		self.lstm = self.lstm.to(device)
		self._device = device

		return ret

	def get_start_states(self, batch_size, hidden_size):
		h0 = torch.zeros(batch_size, hidden_size).to(self._device)
		c1_0 = torch.zeros(batch_size, hidden_size).to(self._device)
		c2_0 = torch.zeros(batch_size, hidden_size).to(self._device)
		return (h0, c1_0, c2_0)

	def forward(self, x1, x2, _, seq_lens):
		batch_size = x1.size(0)
		max_seq_length = x1.size(1)
		embed_size = x1.size(2)

		x1 = pack(x1, seq_lens, batch_first=True)
		x2 = pack(x2, seq_lens, batch_first=True)
		outputs = torch.zeros([max_seq_length, batch_size, self._hidden_size])

		cursor = 0
		sequence_lenths = x1.batch_sizes.cpu().numpy()
		hx, cx1, cx2 = self.get_start_states(batch_size, self._hidden_size)
		for step in range(sequence_lenths.shape[0]):
			sequence_lenth = sequence_lenths[step]

			x1_step = x1.data[cursor:cursor+sequence_lenth]
			x2_step = x2.data[cursor:cursor+sequence_lenth]

			hx, (_, cx1, cx2) = self.lstm(x1_step, x2_step, \
					(hx[:sequence_lenth], cx1[:sequence_lenth], cx2[:sequence_lenth]))
			outputs[step][:sequence_lenth] = hx

			cursor += sequence_lenth

		outputs = torch.transpose(outputs, 1, 0).to(self._device)
#		if self.training:
#			outputs = self.dropout(outputs)

		outputs = self.linear(outputs)

		return outputs

#		outputs = outputs.view(-1, embed_size)
#		outputs = self.bn(outputs)
#		outputs = outputs.view(batch_size, -1, embed_size)
#
#		outputs, _ = unpack(pack(outputs, seq_lens, batch_first=True), batch_first=True)
#		return outputs


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

	predictor = AdressaRec(MultiCellModel, model_ws_path, torch_input_path, dict_url2vec, options)
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

