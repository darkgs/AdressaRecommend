
import os
import json

import torch
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

		if self._dataset.get(data_type, None) == None:
			self._dataset[data_type] = AdressaDataset(self._dict_rnn_input['dataset'][data_type])

		return self._dataset[data_type]

	def idx2vec(self, idx):
		return self._dict_rnn_input['idx2vec'][str(idx)]

	def get_pad_idx(self):
		return self._dict_rnn_input['pad_idx']


def main():
	options, args = parser.parse_args()
	if (options.input == None):
		return

	rnn_input_json_path = '{}/torch_rnn_input.dict'.format(options.input)

	rnn_input = RNNInputTorch(rnn_input_json_path)
	def adressa_collate(batch):
		max_seq = 20
		def pad_sequence(sequence):
			len_diff = (max_seq+1) - len(sequence)

			if len_diff < 0:
				return sequence[:(max_seq+1)]
			elif len_diff == 0:
				return sequence

			padded_sequence = sequence.copy()
			padded_sequence += [rnn_input.get_pad_idx()] * len_diff

			return padded_sequence

		seq_x = []	# batch_size * max_seq * embedding_d
		seq_y = []	# batch_size * max_seq * embedding_d
		seq_len = []	# batch_size * 1

		for timestamp_start, timestamp_end, sequence in batch:
			pad_seq = [rnn_input.idx2vec(idx) for idx in pad_sequence(sequence)]
			seq_len.append(len(sequence))
			seq_x.append(pad_seq[:-1])
			seq_y.append(pad_seq[1:])

		return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), torch.IntTensor(seq_len)

	trainloader = torch.utils.data.DataLoader(rnn_input.get_dataset(data_type='train'),
			batch_size=4, shuffle=True, num_workers=4,
			collate_fn=adressa_collate)

	for i, data in enumerate(trainloader, 0):
		input_x_s, input_y_s, seq_lens = data
		print(input_x_s.shape, input_y_s.shape, seq_lens.shape)
		break

if __name__ == '__main__':
	main()

