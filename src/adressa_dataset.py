
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from torch.utils.data.dataset import Dataset  # For custom datasets

import numpy as np

from ad_util import load_json
from ad_util import weights_init

class AdressaDataset(Dataset):
	def __init__(self, dict_dataset):

		self._dict_dataset = dict_dataset
		self._data_len = len(self._dict_dataset)

	def __getitem__(self, index):
		return self._dict_dataset[index]

	def __len__(self):
		return self._data_len

class AdressaRNNInput(object):
	def __init__(self, rnn_input_json_path, dict_url2vec):
		self._dict_url2vec = dict_url2vec

		self._dict_rnn_input = load_json(rnn_input_json_path)

		self._dataset = {}

	def get_dataset(self, data_type='test'):
		if data_type not in ['train', 'valid', 'test']:
			data_type = 'test'

		max_seq = 20
		if self._dataset.get(data_type, None) == None:
			def pad_sequence(sequence, padding):
				len_diff = max_seq - len(sequence)

				if len_diff < 0:
					return sequence[:max_seq]
				elif len_diff == 0:
					return sequence

				padded_sequence = sequence.copy()
				padded_sequence += [padding] * len_diff

				return padded_sequence

			datas = []

			for timestamp_start, timestamp_end, sequence, time_sequence in \
					self._dict_rnn_input['dataset'][data_type]:
				pad_indices = [idx for idx in pad_sequence(sequence, self.get_pad_idx())]
				pad_time_indices = [idx for idx in pad_sequence(time_sequence, -1)]
#				pad_seq = [normalize([self.idx2vec(idx)], norm='l2')[0] for idx in pad_indices]
				pad_seq = [self.idx2vec(idx) for idx in pad_indices]

				seq_len = min(len(sequence), max_seq) - 1
				seq_x = pad_seq[:-1]
				seq_y = pad_seq[1:]

				idx_x = pad_indices[:-1]
				idx_y = pad_indices[1:]
				
				seq_trendy = [[self.idx2vec(idx) for idx, count in \
						 self.get_trendy(timestamp, 5, self.get_pad_idx())] \
						 for timestamp in pad_time_indices]
				seq_trendy = seq_trendy[:-1]
			
				datas.append(
					(seq_x, seq_y, seq_len, idx_x, idx_y, seq_trendy, \
					 timestamp_start, timestamp_end)
				)


			self._dataset[data_type] = AdressaDataset(datas)

		return self._dataset[data_type]

	def idx2vec(self, idx):
		return self._dict_url2vec[self._dict_rnn_input['idx2url'][str(idx)]]

	def get_pad_idx(self):
		return self._dict_rnn_input['pad_idx']

	def get_trendy(self, cur_time=-1, topk=10, padding=0):
		trendy_list = self._dict_rnn_input['trendy_idx'].get(str(cur_time), None)

		if trendy_list == None:
			return [[padding, 0]] * topk

		assert(len(trendy_list) >= topk)

		return trendy_list[:topk]

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


def adressa_collate(batch):
	batch.sort(key=lambda x: x[2], reverse=True)

	seq_x = [x for x, _, _, _, _, _, _, _ in batch]
	seq_y = [x for _, x, _, _, _, _, _, _ in batch]
	seq_len = [x for _, _, x, _, _, _, _, _ in batch]
	x_indices = [x for _, _, _, x, _, _, _, _ in batch]
	y_indices = [x for _, _, _, _, x, _, _, _ in batch]
	seq_trendy = [x for _, _, _, _, _, x, _, _ in batch]
	timestamp_starts = [x for _, _, _, _, _, _, x, _ in batch]
	timestamp_ends = [x for _, _, _, _, _, _, _, x in batch]

	return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), torch.FloatTensor(seq_trendy), \
		torch.IntTensor(seq_len), timestamp_starts, timestamp_ends, \
		x_indices, y_indices


class AdressaRec(object):
	def __init__(self, model_class, ws_path, torch_input_path, dict_url2vec):
		super(AdressaRec, self).__init__()

		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._ws_path = ws_path

		dim_article = len(next(iter(dict_url2vec.values())))
		learning_rate = 1e-3

		dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)
		self._rnn_input = AdressaRNNInput(dict_rnn_input_path, dict_url2vec)

		self._train_dataloader, self._test_dataloader = \
								self.get_dataloader(dict_url2vec)

		self._model = model_class(dim_article).to(self._device)
		self._model.apply(weights_init)

#self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
#self._criterion = nn.MSELoss()
#self._criterion = nn.CrossEntropyLoss()
		self._criterion = nn.BCELoss()
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

		self._saved_model_path = self._ws_path + '/predictor.pth.tar'

	def get_dataloader(self, dict_url2vec):
		train_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='train'),
				batch_size=256, shuffle=True, num_workers=16,
				collate_fn=adressa_collate)

		test_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='test'),
				batch_size=128, shuffle=True, num_workers=16,
				collate_fn=adressa_collate)

		return train_dataloader, test_dataloader

	def do_train(self, total_epoch=1000, early_stop=10):

		start_epoch, best_test_loss = self.load_model()
		best_mrr = -1.0

		if start_epoch < total_epoch:
			endure = 0
			start_time = time.time()
			for epoch in range(start_epoch, total_epoch):
				if endure > early_stop:
					print('Early stop!')
					break

				train_loss = self.train()
				test_loss = self.test()
				mrr_20 = self.test_mrr_20()
				best_mrr = max(best_mrr, mrr_20)
				best_test_loss = min(best_test_loss, test_loss)

				print('epoch {} - train loss({:.8f}) test loss({:.8f}) test mrr_20({:.4f}) best mrr({:.4f}) tooks {:.2f}'.format(
					epoch, train_loss, test_loss, mrr_20, best_mrr, time.time() - start_time))

				if best_mrr == mrr_20:
					self.save_model(epoch, test_loss)
					print('Model saved! - best mrr({})'.format(best_mrr))

				if best_test_loss == test_loss:
					endure = 0
				else:
					endure += 1

	def train(self):
		self._model.train()
		train_loss = 0.0
		batch_count = len(self._train_dataloader)

		for batch_idx, train_input in enumerate(self._train_dataloader):
			input_x_s, input_y_s, input_trendy, seq_lens, \
				timestamp_starts, timestamp_ends, indices_x, indices_y = train_input
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_trendy = input_trendy.to(self._device)

			self._model.zero_grad()
			self._optimizer.zero_grad()

#outputs = self._model(input_x_s, input_trendy, seq_lens)
#unpacked_y_s, _ = unpack(pack(input_y_s, seq_lens, batch_first=True), batch_first=True)

#loss = self._criterion(outputs, unpacked_y_s)

			outputs = self._model(input_x_s, input_trendy, seq_lens)
			packed_outputs = pack(outputs, seq_lens, batch_first=True).data
			packed_y_s = pack(input_y_s, seq_lens, batch_first=True).data

#loss = self._criterion(F.softmax(packed_outputs, dim=1), \
#F.softmax(packed_y_s, dim=1).long())
			loss = self._criterion(F.softmax(packed_outputs, dim=1), \
					F.softmax(packed_y_s, dim=1))
			loss.backward()
			self._optimizer.step()

			train_loss += loss.item()

		return train_loss / batch_count

	def test(self):
		self._model.eval()

		test_loss = 0.0
		batch_count = len(self._test_dataloader)

		for batch_idx, test_input in enumerate(self._test_dataloader):
			input_x_s, input_y_s, input_trendy, seq_lens, _, _, _, _ = test_input
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_trendy = input_trendy.to(self._device)

#outputs = self._model(input_x_s, input_trendy, seq_lens)
#unpacked_y_s, _ = unpack(pack(input_y_s, seq_lens, batch_first=True), batch_first=True)
#loss = self._criterion(outputs, unpacked_y_s)

			outputs = self._model(input_x_s, input_trendy, seq_lens)
			packed_outputs = pack(outputs, seq_lens, batch_first=True).data
			packed_y_s = pack(input_y_s, seq_lens, batch_first=True).data

			loss = self._criterion(F.softmax(packed_outputs, dim=1), F.softmax(packed_y_s, dim=1))

			test_loss += loss.item()

		return test_loss / batch_count

	def test_mrr_20(self):

		self._model.eval()

		predict_count = 0
		predict_mrr = 0.0
		for i, data in enumerate(self._test_dataloader, 0):
			input_x_s, input_y_s, input_tendy, seq_lens, \
				timestamp_starts, timestamp_ends, _, indices_y = data
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_tendy = input_tendy.to(self._device)
			input_y_s = input_y_s.cpu().numpy()

			with torch.no_grad():
				outputs = self._model(input_x_s, input_tendy, seq_lens)

			outputs = torch.tanh(outputs)
			outputs = outputs.cpu().numpy()

			batch_size = seq_lens.size(0)
			seq_lens = seq_lens.cpu().numpy()

			for batch in range(batch_size):
				cand_indices = self._rnn_input.get_candidates(start_time=timestamp_starts[batch],
						end_time=timestamp_ends[batch], idx_count=100)
				cand_embed = [self._rnn_input.idx2vec(idx) for idx in cand_indices]
				cand_matrix = np.matrix(cand_embed)
				cand_matrix = np.tanh(cand_matrix)
#cand_matrix = np_sigmoid(np.matrix(cand_embed))

				for seq_idx in range(seq_lens[batch]):

					next_idx = indices_y[batch][seq_idx]
					if next_idx not in cand_indices:
						continue

					pred_vector = outputs[batch][seq_idx]
#pred_vector = np_sigmoid(outputs[batch][seq_idx])
					cand_eval = np.asarray(np.dot(cand_matrix, pred_vector).T).tolist()

					infered_values = [(cand_indices[i], evaluated[0]) \
										for i, evaluated in enumerate(cand_eval)]
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

	def save_model(self, epoch, test_loss):
		dict_states = {
			'epoch': epoch,
			'test_loss': test_loss,
			'model': self._model.state_dict(),
			'optimizer': self._optimizer.state_dict(),
		}

		torch.save(dict_states, self._saved_model_path)

	def load_model(self):
		return 0, sys.float_info.max

		if not os.path.exists(self._saved_model_path):
			return 0, sys.float_info.max

		dict_states = torch.load(self._saved_model_path)
		self._model.load_state_dict(dict_states['model'])
		self._optimizer.load_state_dict(dict_states['optimizer'])

		return dict_states['epoch'], dict_states['test_loss']

