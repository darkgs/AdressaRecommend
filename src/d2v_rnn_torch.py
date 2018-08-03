
import os
import json
import itertools
import time

import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

from torch.utils.data.dataset import Dataset  # For custom datasets

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)

class AdressaDataset(Dataset):
	def __init__(self, dict_dataset):

		self._dict_dataset = dict_dataset
		self._data_len = len(self._dict_dataset)

	def __getitem__(self, index):
		return self._dict_dataset[index]

	def __len__(self):
		return self._data_len

class RNNInputTorch(object):
	def __init__(self, rnn_input_json_path):
		with open(rnn_input_json_path, 'r') as f_rnn_input:
			self._dict_rnn_input = json.load(f_rnn_input)

		self._dataset = {}

	def get_dataset(self, data_type='test'):
		if data_type not in ['train', 'valid', 'test']:
			data_type = 'test'

		max_seq = 20
		if self._dataset.get(data_type, None) == None:
			print('get_dataset : {}'.format(data_type))
			def pad_sequence(sequence):
				len_diff = (max_seq+1) - len(sequence)

				if len_diff < 0:
					return sequence[:(max_seq+1)]
				elif len_diff == 0:
					return sequence

				padded_sequence = sequence.copy()
				padded_sequence += [self.get_pad_idx()] * len_diff

				return padded_sequence

			datas = []
			for timestamp_start, timestamp_end, sequence in self._dict_rnn_input['dataset'][data_type]:
				pad_indices = [idx for idx in pad_sequence(sequence)]
				pad_seq = [self.idx2vec(idx) for idx in pad_indices]

				seq_len = min(len(sequence), max_seq)
				seq_x = pad_seq[:-1]
				seq_y = pad_seq[1:]

				idx_x = pad_indices[:-1]
				idx_y = pad_indices[1:]
			
				datas.append(
					(seq_x, seq_y, seq_len, idx_x, idx_y, timestamp_start, timestamp_end)
				)


			self._dataset[data_type] = AdressaDataset(datas)

		return self._dataset[data_type]

	def idx2vec(self, idx):
		return self._dict_rnn_input['idx2vec'][str(idx)]

	def get_pad_idx(self):
		return self._dict_rnn_input['pad_idx']

	def get_embed_dimension(self):
		return self._dict_rnn_input['embedding_dimension']

	def get_candidates(self, start_time=-1, end_time=-1, idx_count=0):
		if (start_time < 0) or (end_time < 0) or (idx_count <= 0):
			return []

		#	entry of : dict_rnn_input['time_idx']
		#	(timestamp) :
		#	{
		#		prev_time: (timestamp)
		#		next_time: (timestamp)
		#		'indices': { idx:count, ... }
		#	}

		# swap if needed
		if start_time > end_time:
			tmp_time = start_time
			start_time = end_time
			end_time = tmp_time

		cur_time = start_time

		dict_merged = {}
		while(cur_time < end_time):
			cur_time = self._dict_rnn_input['time_idx'][str(cur_time)]['next_time']
			for idx, count in self._dict_rnn_input['time_idx'][str(cur_time)]['indices'].items():
				dict_merged[idx] = dict_merged.get(idx, 0) + count

		steps = 0
		time_from_start = start_time
		time_from_end = end_time
		while(len(dict_merged.keys()) < idx_count):
			if time_from_start == None and time_from_end == None:
				break

			if steps % 3 == 0:
				if time_from_end == None:
					steps += 1
					continue
				cur_time = self._dict_rnn_input['time_idx'][str(time_from_end)]['next_time']
				time_from_end = cur_time
			else:
				if time_from_start == None:
					steps += 1
					continue
				cur_time = self._dict_rnn_input['time_idx'][str(time_from_start)]['prev_time']
				time_from_start = cur_time

			if cur_time == None:
				continue

			for idx, count in self._dict_rnn_input['time_idx'][str(cur_time)]['indices'].items():
				dict_merged[idx] = dict_merged.get(idx, 0) + count

		ret_sorted = sorted(dict_merged.items(), key=lambda x:x[1], reverse=True)
		if len(ret_sorted) > idx_count:
			ret_sorted = ret_sorted[:idx_count]
		return list(map(lambda x: int(x[0]), ret_sorted))


class RNNRecommender(nn.Module):

	def __init__(self, embed_size, hidden_size, num_layers):
		super(RNNRecommender, self).__init__()

		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Sequential(
			nn.Linear(hidden_size, embed_size),
			nn.BatchNorm1d(embed_size, momentum=0.01),
		)

	def forward(self, x):
		outputs, _ = self.lstm(x)
		outputs = self.linear(outputs[0])

		return outputs


def main():
	options, args = parser.parse_args()
	if (options.input == None):
		return

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	rnn_input_json_path = '{}/torch_rnn_input.dict'.format(options.input)

	rnn_input = RNNInputTorch(rnn_input_json_path)
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
			batch_size=1024, shuffle=True, num_workers=4,
			collate_fn=adressa_collate)

	testloader = torch.utils.data.DataLoader(rnn_input.get_dataset(data_type='test'),
			batch_size=32, shuffle=True, num_workers=4,
			collate_fn=adressa_collate)

	embed_size = rnn_input.get_embed_dimension()
	hidden_size = 1024
	num_layers = 3

	model = RNNRecommender(embed_size, hidden_size, num_layers).to(device)

	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	def train():
		model.train()
		optimizer.zero_grad()
		for i, data in enumerate(trainloader, 0):
			time_check = time.time()
			input_x_s, input_y_s, seq_lens, _, _, _, _ = data
			input_x_s = input_x_s.to(device)
			input_y_s = input_y_s.to(device)

			packed_x_s = pack_padded_sequence(input_x_s, seq_lens, batch_first=True)
			packed_y_s = pack_padded_sequence(input_y_s, seq_lens, batch_first=True)

			normalized_y_s = torch.Tensor(preprocessing.normalize(packed_y_s[0], norm='l2')).to(device)

			outputs = model(packed_x_s)

			loss = criterion(outputs, normalized_y_s)

			model.zero_grad()
			loss.backward()
			optimizer.step()

	def test_mrr_20():
		model.eval()

		predict_count = 0
		predict_mrr = 0.0
		total_iter = len(testloader)
		for i, data in enumerate(testloader, 0):
			input_x_s, input_y_s, seq_lens, timestamp_starts, timestamp_ends, indices_x, indices_y = data
			input_x_s = input_x_s.to(device)
			input_y_s = input_y_s.to(device)

			packed_x_s = pack_padded_sequence(input_x_s, seq_lens, batch_first=True)
			packed_y_s = pack_padded_sequence(input_y_s, seq_lens, batch_first=True)

			with torch.no_grad():
				outputs = model(packed_x_s).cpu().numpy()

			batch_size = seq_lens.size(0)
			seq_lens = seq_lens.cpu().numpy()

			packed_idx = 0
			for batch in range(batch_size):
				cand_indices = rnn_input.get_candidates(start_time=timestamp_starts[batch],
						end_time=timestamp_ends[batch], idx_count=100)
				cand_embed = [rnn_input.idx2vec(idx) for idx in cand_indices]

				cand_matrix = np.matrix(cand_embed)
				for seq_idx in range(seq_lens[batch]):
					next_idx = indices_y[batch][seq_idx]
					if next_idx not in cand_indices:
						packed_idx += 1
						continue

					pred_vector = outputs[packed_idx]
					cand_eval = np.asarray(np.dot(cand_matrix, pred_vector).T)

					infered_values = [(cand_indices[i], evaluated) for i, evaluated in enumerate(cand_eval)]
					infered_values.sort(key=lambda x:x[1])
					# MRR@20
					rank = -1
					for infered_idx in range(20):
						if next_idx == infered_values[infered_idx][0]:
							rank = infered_idx + 1

					if rank > 0:
						predict_mrr += 1.0/float(rank)
					predict_count += 1

					packed_idx += 1

		return predict_mrr / float(predict_count)


	for epoch in range(100):
		start_time = time.time()
		print('train epoch {} start'.format(epoch))
		train()
		print('train epoch {} tooks {}'.format(epoch, time.time()-start_time))
#		print('train epoch {} mrr({}) tooks {}'.format(epoch, test_mrr_20(), time.time()-start_time))


if __name__ == '__main__':
	main()

