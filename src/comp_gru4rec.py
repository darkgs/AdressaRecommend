
import os, sys

from optparse import OptionParser

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from d2v_rnn_torch import AdressaDataset
from d2v_rnn_torch import RNNInputTorch as AdressaRNNInput

from ad_util import weights_init
from ad_util import load_json

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)

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

class GRU4RecModel(nn.Module):
	def __init__(self, embed_size, hidden_size, num_layers):
		super(GRU4RecModel, self).__init__()

		self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
		self.linear = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, x, seq_lens):
		batch_size = x.size(0)
		embed_size = x.size(2)

		x = pack(x, seq_lens, batch_first=True)
		outputs, _ = self.rnn(x)
		outputs, _ = unpack(outputs, batch_first=True)
		outputs = self.linear(outputs)

		outputs = outputs.view(-1, embed_size)
		outputs = self.bn(outputs)
		outputs = outputs.view(batch_size, -1, embed_size)

		return outputs

class GRU4Rec(object):
	def __init__(self, ws_path, torch_input_path, dict_url2vec):
		super(GRU4Rec, self).__init__()

		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._ws_path = ws_path

		dim_article = len(next(iter(dict_url2vec.values())))
		hidden_size = 512
		num_layers = 3
		learning_rate = 0.01

		dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)
		self._rnn_input = AdressaRNNInput(dict_rnn_input_path, dict_url2vec)

		self._train_dataloader, self._test_dataloader = \
								self.get_dataloader(dict_url2vec)

		self._model = GRU4RecModel(dim_article, hidden_size, num_layers).to(self._device)
		self._model.apply(weights_init)

#self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
		self._criterion = nn.MSELoss()
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

		self._saved_model_path = self._ws_path + '/perdictor.pth.tar'

	def get_dataloader(self, dict_url2vec):
		train_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='train'),
				batch_size=128, shuffle=True, num_workers=16,
				collate_fn=adressa_collate)

		test_dataloader = torch.utils.data.DataLoader(self._rnn_input.get_dataset(data_type='test'),
				batch_size=32, shuffle=True, num_workers=4,
				collate_fn=adressa_collate)

		return train_dataloader, test_dataloader

	def train(self):
		self._model.train()
		train_loss = 0.0
		batch_count = len(self._train_dataloader)

		for batch_idx, train_input in enumerate(self._train_dataloader):
			input_x_s, input_y_s, seq_lens, _, _, _, _ = train_input
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)

			self._model.zero_grad()
			self._optimizer.zero_grad()

			outputs = self._model(input_x_s, seq_lens)
			unpacked_y_s, _ = unpack(pack(input_y_s, seq_lens, batch_first=True), batch_first=True)

#loss = F.binary_cross_entropy(torch.sigmoid(outputs), torch.sigmoid(unpacked_y_s))
			loss = self._criterion(outputs, unpacked_y_s)
			loss.backward()
			self._optimizer.step()

			train_loss += loss.item()

		return train_loss / batch_count

	def test(self):
		self._model.eval()

		test_loss = 0.0
		batch_count = len(self._test_dataloader)

		for batch_idx, test_input in enumerate(self._test_dataloader):
			input_x_s, input_y_s, seq_lens, _, _, _, _ = test_input
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)

			outputs = self._model(input_x_s, seq_lens)
			unpacked_y_s, _ = unpack(pack(input_y_s, seq_lens, batch_first=True), batch_first=True)

#loss = F.binary_cross_entropy(torch.sigmoid(outputs), torch.sigmoid(unpacked_y_s))
			loss = self._criterion(outputs, unpacked_y_s)
			test_loss += loss.item()

		return test_loss / batch_count

	def test_mrr_20(self):
		def np_sigmoid(x):
			return 1/(1+np.exp(-x))

		self._model.eval()

		predict_count = 0
		predict_mrr = 0.0
		for i, data in enumerate(self._test_dataloader, 0):
			input_x_s, input_y_s, seq_lens, \
				timestamp_starts, timestamp_ends, indices_x, indices_y = data
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_y_s = input_y_s.cpu().numpy()

			with torch.no_grad():
				outputs = self._model(input_x_s, seq_lens)

			outputs = outputs.cpu().numpy()

			batch_size = seq_lens.size(0)
			seq_lens = seq_lens.cpu().numpy()

			for batch in range(batch_size):
				cand_indices = self._rnn_input.get_candidates(start_time=timestamp_starts[batch],
						end_time=timestamp_ends[batch], idx_count=100)
				cand_embed = [self._rnn_input.idx2vec(idx) for idx in cand_indices]
				cand_matrix = np.matrix(cand_embed)
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
		if not os.path.exists(self._saved_model_path):
			return 0, sys.float_info.max

		dict_states = torch.load(self._saved_model_path)
		self._model.load_state_dict(dict_states['model'])
		self._optimizer.load_state_dict(dict_states['optimizer'])

		return dict_states['epoch'], dict_states['test_loss']


def main():
	options, args = parser.parse_args()

	if (options.input == None) or (options.d2v_embed == None) or \
					   (options.u2v_path == None) or (options.ws_path == None):
		return

	torch_input_path = options.input
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = options.u2v_path
	ws_path = options.ws_path

#os.system('rm -rf {}'.format(ws_path))
	os.system('mkdir -p {}'.format(ws_path))

	print('Loading url2vec : start')
	dict_url2vec = load_json(url2vec_path)
	print('Loading url2vec : end')

	predictor = GRU4Rec(ws_path, torch_input_path, dict_url2vec)

	start_epoch, best_test_loss = predictor.load_model()
	total_epoch = 1000
	if start_epoch < total_epoch:
		endure = 0
		for epoch in range(start_epoch, total_epoch):
			if endure > 3:
				print('Early stop!')
				break

			train_loss = predictor.train()
			test_loss = predictor.test()
			mrr_20 = predictor.test_mrr_20()

			print('epoch {} - train loss({}) test loss({}) test mrr_20({})'.format(
						epoch, train_loss, test_loss, mrr_20))

			if epoch % 5 == 0:
				if test_loss < best_test_loss:
					best_test_loss = test_loss
					endure = 0
					predictor.save_model(epoch, test_loss)
					print('Model saved! - test loss({})'.format(test_loss))
				else:
					endure += 1

	print('mrr 20 : {}'.format(predictor.test_mrr_20()))

if __name__ == '__main__':
	main()

