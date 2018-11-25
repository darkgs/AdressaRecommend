
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

	seq_x = [x for x, _, _, _, _, _, _, _ in batch]
	seq_y = [x for _, x, _, _, _, _, _, _ in batch]
	seq_len = [x for _, _, x, _, _, _, _, _ in batch]
	x_indices = [x for _, _, _, x, _, _, _, _ in batch]
	y_indices = [x for _, _, _, _, x, _, _, _ in batch]
#seq_trendy = [np.mean(np.array(x), axis=1).tolist() for _, _, _, _, _, x, _, _ in batch]
	seq_trendy = [x for _, _, _, _, _, x, _, _ in batch]
	timestamp_starts = [x for _, _, _, _, _, _, x, _ in batch]
	timestamp_ends = [x for _, _, _, _, _, _, _, x in batch]

	return torch.FloatTensor(seq_x), torch.FloatTensor(seq_y), torch.FloatTensor(seq_trendy), \
		torch.IntTensor(seq_len), timestamp_starts, timestamp_ends, \
		x_indices, y_indices

class MultiCellLSTM(nn.Module):
	def __init__(self, embed_size, hidden_size, attn_count):
		super(MultiCellLSTM, self).__init__()

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

		self._W_attn = torch.zeros([hidden_size+embed_size, attn_count])

		nn.init.xavier_normal_(self._W1_f.data)
		nn.init.xavier_normal_(self._W1_i.data)
		nn.init.xavier_normal_(self._W1_c.data)
		nn.init.xavier_normal_(self._W1_o.data)

		nn.init.xavier_normal_(self._W2_f.data)
		nn.init.xavier_normal_(self._W2_i.data)
		nn.init.xavier_normal_(self._W2_c.data)
		nn.init.xavier_normal_(self._W2_o.data)

		nn.init.xavier_normal_(self._W_attn.data)

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
		attn_score = torch.softmax(torch.matmul(torch.cat([h_t, x1], 1), self._W_attn), dim=1)
		attn_score = torch.unsqueeze(attn_score, dim=1)
		x2 = torch.squeeze(torch.bmm(attn_score, x2), dim=1)

		x2 = x1

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

		# out gate
		o1_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x1], 1), self._W1_o) + self._b1_o)
		o2_t = torch.sigmoid(torch.matmul(torch.cat([h_t, x2], 1), self._W2_o) + self._b2_o)
		
		# new hidden state
		h_t = torch.tanh(o1_t * torch.tanh(c1_t) + o2_t * torch.tanh(c2_t))

		return h_t, (h_t, c1_t, c2_t)

	
class MyLSTM(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super(MyLSTM, self).__init__()

		self._W_f = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b_f = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W_i = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b_i = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W_c = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b_c = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		self._W_o = torch.zeros([hidden_size+embed_size, hidden_size], dtype=torch.float32, requires_grad=True)
		self._b_o = torch.zeros([hidden_size], dtype=torch.float32, requires_grad=True)

		nn.init.xavier_normal_(self._W_f.data)
		nn.init.xavier_normal_(self._W_i.data)
		nn.init.xavier_normal_(self._W_c.data)
		nn.init.xavier_normal_(self._W_o.data)

	def to(self, device):
		ret = super(MyLSTM, self).to(device)

		self._W_f = self._W_f.to(device)
		self._b_f = self._b_f.to(device)

		self._W_i = self._W_i.to(device)
		self._b_i = self._b_i.to(device)

		self._W_c = self._W_c.to(device)
		self._b_c = self._b_c.to(device)

		self._W_o = self._W_o.to(device)
		self._b_o = self._b_o.to(device)

		return ret

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


class MultiCellModel(nn.Module):
	def __init__(self, embed_size, hidden_size):
		super(MultiCellModel, self).__init__()

		self._hidden_size = hidden_size

		self.lstm = MultiCellLSTM(embed_size, hidden_size, 5)
		self.linear = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

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

	def forward(self, x1, x2, seq_lens):
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

		outputs = self.linear(outputs)

		outputs = outputs.view(-1, embed_size)
		outputs = self.bn(outputs)
		outputs = outputs.view(batch_size, -1, embed_size)

		outputs, _ = unpack(pack(outputs, seq_lens, batch_first=True), batch_first=True)
		return outputs


class MultiCellRec(object):
	def __init__(self, ws_path, torch_input_path, dict_url2vec):
		super(MultiCellRec, self).__init__()

		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._ws_path = ws_path

		dim_article = len(next(iter(dict_url2vec.values())))
		hidden_size = 256
		learning_rate = 1e-3

		dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)
		self._rnn_input = AdressaRNNInput(dict_rnn_input_path, dict_url2vec)

		self._train_dataloader, self._test_dataloader = \
								self.get_dataloader(dict_url2vec)

		self._model = MultiCellModel(dim_article, hidden_size).to(self._device)
		self._model.apply(weights_init)

#self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
		self._criterion = nn.MSELoss()
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)

		self._saved_model_path = self._ws_path + '/predictor.pth.tar'

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
			input_x_s, input_y_s, input_trendy, seq_lens, \
				timestamp_starts, timestamp_ends, indices_x, indices_y = train_input
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_trendy = input_trendy.to(self._device)

			self._model.zero_grad()
			self._optimizer.zero_grad()

			outputs = self._model(input_x_s, input_trendy, seq_lens)
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
			input_x_s, input_y_s, input_trendy, seq_lens, _, _, _, _ = test_input
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_trendy = input_trendy.to(self._device)

			outputs = self._model(input_x_s, input_trendy, seq_lens)
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
			input_x_s, input_y_s, input_tendy, seq_lens, \
				timestamp_starts, timestamp_ends, _, indices_y = data
			input_x_s = input_x_s.to(self._device)
			input_y_s = input_y_s.to(self._device)
			input_tendy = input_tendy.to(self._device)
			input_y_s = input_y_s.cpu().numpy()

			with torch.no_grad():
				outputs = self._model(input_x_s, input_tendy, seq_lens)

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
		return 0, sys.float_info.max

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

	predictor = MultiCellRec(ws_path, torch_input_path, dict_url2vec)

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

			continue

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

