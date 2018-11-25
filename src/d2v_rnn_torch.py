
import os
import json
import itertools
import time

import random

import numpy as np
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from adressa_dataset import AdressaRNNInput

from optparse import OptionParser

from ad_util import weights_init
from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)


dict_url_vec = {}
def load_url2vec(url2vec_path=None):
	global dict_url_vec

	dict_url_vec = {}
	if url2vec_path == None:
		return

	with open(url2vec_path, 'r') as f_u2v:
		dict_url_vec = json.load(f_u2v)


class RNNRecommender(nn.Module):

	def __init__(self, embed_size, hidden_size, num_layers):
		super(RNNRecommender, self).__init__()

#		self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

		self._embed_size = embed_size
		self._hidden_size = hidden_size

	def forward(self, x, seq_lens):
		batch_size = x.size(0)

		x = pack(x, seq_lens, batch_first=True)
		outputs, _ = self.rnn(x)
		outputs, _ = unpack(outputs, batch_first=True)
		outputs = self.linear(outputs)

		outputs = outputs.view(-1, self._embed_size)
		outputs = self.bn(outputs)
		outputs = outputs.view(batch_size, -1, self._embed_size)

		return outputs


def main():
	global dict_url_vec

	options, args = parser.parse_args()
	if (options.input == None) or (options.d2v_embed == None) or (options.u2v_path == None):
		return

	torch_input_path = options.input
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = options.u2v_path

	print('Loading url2vec : start')
	load_url2vec(url2vec_path=url2vec_path)
	print('Loading url2vec : end')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	rnn_input_json_path = '{}/torch_rnn_input.dict'.format(torch_input_path)

	rnn_input = AdressaRNNInput(rnn_input_json_path, dict_url_vec)
	def adressa_collate(batch):
		batch.sort(key=lambda x: x[2], reverse=True)

		seq_x_b = []	# batch_size * max_seq * embedding_d
		seq_y_b = []	# batch_size * max_seq * embedding_d
		seq_len_b = []	# batch_size * 1
		timestamp_starts = []
		timestamp_ends = []
		idx_x_b = []
		idx_y_b = []
		for seq_x, seq_y, seq_len, idx_x, idx_y, \
			timestamp_start, timestamp_end in batch:
			seq_x_b.append(seq_x)
			seq_y_b.append(seq_y)
			seq_len_b.append(seq_len)
			timestamp_starts.append(timestamp_start)
			timestamp_ends.append(timestamp_end)
			idx_x_b.append(idx_x)
			idx_y_b.append(idx_y)

		return torch.FloatTensor(seq_x_b), torch.FloatTensor(seq_y_b), \
			torch.IntTensor(seq_len_b), timestamp_starts, timestamp_ends, \
			idx_x_b, idx_y_b

	trainloader = torch.utils.data.DataLoader(rnn_input.get_dataset(data_type='train'),
			batch_size=128, shuffle=True, num_workers=16,
			collate_fn=adressa_collate)

	testloader = torch.utils.data.DataLoader(rnn_input.get_dataset(data_type='test'),
			batch_size=32, shuffle=True, num_workers=4,
			collate_fn=adressa_collate)

	embed_size = embedding_dimension
	hidden_size = 512
	num_layers = 3

	model = RNNRecommender(embed_size, hidden_size, num_layers).to(device)
	model.apply(weights_init)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
#	optimizer = optim.SGD(model.parameters(), lr=0.01)

	def train():
		model.train()
		for i, data in enumerate(trainloader, 0):
			time_check = time.time()
			input_x_s, input_y_s, seq_lens, _, _, _, _ = data
			input_x_s = input_x_s.to(device)
			input_y_s = input_y_s.to(device)

#			packed_x_s = pack_padded_sequence(input_x_s, seq_lens, batch_first=True)
#			packed_y_s = pack_padded_sequence(input_y_s, seq_lens, batch_first=True)

#			normalized_y_s = torch.Tensor(preprocessing.normalize(packed_y_s[0], norm='l2')).to(device)
			model.zero_grad()
			optimizer.zero_grad()

			outputs = model(input_x_s, seq_lens)
			unpacked_y_s, _ = unpack(pack(input_y_s, seq_lens, batch_first=True), batch_first=True)
#			outputs = outputs.reshape(unpacked_y_s.shape)
			loss = criterion(outputs, unpacked_y_s)

			loss.backward()
			optimizer.step()

	def test_mrr_20():
		model.eval()

		predict_count = 0
		predict_mrr = 0.0
		for i, data in enumerate(testloader, 0):
#			if random.randrange(0,10) != 0:
#				continue
			input_x_s, input_y_s, seq_lens, timestamp_starts, timestamp_ends, indices_x, indices_y = data
			input_x_s = input_x_s.to(device)
			input_y_s = input_y_s.to(device)
			input_y_s = input_y_s.cpu().numpy()

			with torch.no_grad():
				outputs = model(input_x_s, seq_lens)
				
			outputs = outputs.cpu().numpy()

			batch_size = seq_lens.size(0)
			seq_lens = seq_lens.cpu().numpy()

			for batch in range(batch_size):
				cand_indices = rnn_input.get_candidates(start_time=timestamp_starts[batch],
						end_time=timestamp_ends[batch], idx_count=100)
				cand_embed = [rnn_input.idx2vec(idx) for idx in cand_indices]
				cand_matrix = np.matrix(cand_embed)

				for seq_idx in range(seq_lens[batch]):

					next_idx = indices_y[batch][seq_idx]
					if next_idx not in cand_indices:
						continue

					pred_vector = outputs[batch][seq_idx]
					cand_eval = np.asarray(np.dot(cand_matrix, pred_vector).T).tolist()

					infered_values = [(cand_indices[i], evaluated[0]) for i, evaluated in enumerate(cand_eval)]
					infered_values.sort(key=lambda x:x[1], reverse=True)
					# MRR@20
					rank = -1
					for infered_idx in range(20):
						if next_idx == infered_values[infered_idx][0]:
							rank = infered_idx + 1

					if rank > 0:
						predict_mrr += 1.0/float(rank)
					predict_count += 1

		return predict_mrr / float(predict_count) if predict_count > 0 else 0.0


	for epoch in range(1000):
		start_time = time.time()
#		write_log('train epoch {} start'.format(epoch))
		print('train epoch {} start'.format(epoch))
		train()
		print('train epoch {} mrr({}) tooks {}'.format(epoch, test_mrr_20(), time.time()-start_time))
#		if epoch % 10 == 0:
#			write_log('train epoch {} mrr({}) tooks {}'.format(epoch, test_mrr_20(), time.time()-start_time))
#		else:
#			write_log('train epoch {} tooks {}'.format(epoch, time.time()-start_time))


if __name__ == '__main__':
	main()

