
import os, sys
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from torch.utils.data.dataset import Dataset  # For custom datasets

from optparse import OptionParser

from d2v_rnn_torch import AdressaDataset
from d2v_rnn_torch import RNNInputTorch as AdressaRNNInput
from comp_gru4rec import adressa_collate

from ad_util import weights_init
from ad_util import load_json

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-a', '--u2i_path', dest='u2i_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)

class ArticleModel(nn.Module):
	def __init__(self, dim_article, dim_h, corruption_rate=0.3):
		super(ArticleModel, self).__init__()

		self._p = corruption_rate

		self._encode_w = torch.zeros((dim_article, dim_h), requires_grad=True)
		nn.init.xavier_uniform_(self._encode_w, gain=nn.init.calculate_gain('relu'))
		self._encode_b = torch.zeros((dim_h,), requires_grad=True)

		self._decode = torch.nn.Linear(dim_h, dim_article)

	def to(self, *args, **kwargs):
		ret = super(ArticleModel, self).to(*args, **kwargs)

		device, dtype, non_blocking = torch._C._nn._parse_to(*args, **kwargs)

		self._encode_w = self._encode_w.to(device)
		self._encode_b = self._encode_b.to(device)
		return ret


	def forward(self, x0, x1, x2):
		if self.training:
			# TODO stocastic corruption distribution
			x0 = x0
			x1 = x1
			x2 = x2
		else:
			# constant decay
			x0 = (1.0 - self._p) * x0
			x1 = (1.0 - self._p) * x1
			x2 = (1.0 - self._p) * x2

		h0 = torch.matmul(x0, self._encode_w) + self._encode_b
		h0 = torch.sigmoid(h0) - torch.sigmoid(self._encode_b)
		h1 = torch.matmul(x1, self._encode_w) + self._encode_b
		h1 = torch.sigmoid(h1) - torch.sigmoid(self._encode_b)
		h2 = torch.matmul(x2, self._encode_w) + self._encode_b
		h2 = torch.sigmoid(h2) - torch.sigmoid(self._encode_b)

		y0 = torch.sigmoid(self._decode(h0))
		y1 = torch.sigmoid(self._decode(h1))
		y2 = torch.sigmoid(self._decode(h2))
		return [h0, h1, h2], [y0, y1, y2]

	def inference(self, x):
		h = torch.matmul(x, self._encode_w) + self._encode_b
		h = torch.sigmoid(h) - torch.sigmoid(self._encode_b)
		y = torch.sigmoid(self._decode(h))
		return y


class ArticleRepresentationDataset(Dataset):
	def __init__(self, dataset):
		self._dataset = dataset

	def __getitem__(self, index):
		return self._dataset[index]

	def __len__(self):
		return len(self._dataset)


class ArticleRepresentation(object):
	def __init__(self, dict_url2vec, dict_url2info, ws_path):
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._dict_url2vec = dict_url2vec
		self._dict_url2info = dict_url2info
		self._ws_path = ws_path

		self._dim_article = len(next(iter(dict_url2vec.values())))
		self._dim_h = self._dim_article // 2
		learning_rate = 0.01

		# Generate dataloader
		self._train_dataloader, self._test_dataloader = self.get_dataloader()
		
		# Model
		self._vae = ArticleModel(self._dim_article, self._dim_h).to(self._device)
		self._vae.apply(weights_init)
#self._optimizer = torch.optim.SGD(self._vae.parameters(), lr=learning_rate, momentum=0.9)
		self._optimizer = torch.optim.Adam(self._vae.parameters(), lr=0.001)

		self._saved_model_path = self._ws_path + '/article_representation_vae.pth.tar'

	def get_dataloader(self):
		saved_rnn_input_path = self._ws_path + '/rnn_input_article.json'

		if not os.path.exists(saved_rnn_input_path):
			print("rnn input : Newly generating!")
			dict_rnn_input = self.generate_rnn_input()
			with open(saved_rnn_input_path, 'w') as f_input:
				json.dump(dict_rnn_input, f_input)
			print("rnn input : Newly generating! - end")
		else:
			print("rnn input : Load from file!")
			with open(saved_rnn_input_path, 'r') as f_input:
				dict_rnn_input = json.load(f_input)
			print("rnn input : Load from file! - end")

		def article_collate(batch):
			# batch * (url0, url1, url_diff)
			return torch.FloatTensor([ (self._dict_url2vec[url0], self._dict_url2vec[url1], 
					self._dict_url2vec[url_diff]) for url0, url1, url_diff in batch ])

		train_dataset = ArticleRepresentationDataset(dict_rnn_input['train'])
		test_dataset = ArticleRepresentationDataset(dict_rnn_input['test'])

		train_dataloader = torch.utils.data.DataLoader(train_dataset,
						batch_size=128, shuffle=True, num_workers=16,
						collate_fn=article_collate)

		test_dataloader = torch.utils.data.DataLoader(test_dataset,
						batch_size=32, shuffle=True, num_workers=4,
						collate_fn=article_collate)


		return train_dataloader, test_dataloader

	def generate_rnn_input(self):
		stat = True
		# 38,155 items has category in the total 51,418
		# 44 category variation
		# dict_url2info contains only existing url in the sequence
		dict_datas = {}
		if stat:
			total_datas = 0
			cate_datas = 0

		for url, dict_info in self._dict_url2info.items():
			category = dict_info.get("category0", None)

			if self._dict_url2vec.get(url, None) == None:
				if stat:
					total_datas += 1
				continue

			if category == None:
				continue

			dict_datas[category] = dict_datas.get(category, [])
			dict_datas[category].append(url)

			if stat:
				cate_datas += 1

		if stat:
			print('has category / total : {}/{}'.format(cate_datas, total_datas))

		train_datas = {}
		test_datas = {}

		for category, datas in dict_datas.items():
			if len(datas) < 20:
				train_datas[category] = datas
				continue

			random.shuffle(datas)

			test_datas[category] = datas[:len(datas)//10]
			train_datas[category] = datas[len(test_datas[category]):]

		def generate_triples(target_datas):
			if stat:
				sim_combs = 0
				for category, urls in target_datas.items():
					sim_combs += len(urls) * (len(urls) - 1) // 2
				print('Total similar combinations : {}'.format(sim_combs))

			# full combination of same categories
			# There are 172,991,409 combinations in the train set
			# There are 2,131,705 combinations in the test set
			url_triples = []
			for category, urls in target_datas.items():
				another_urls = []
				for cate, urls in target_datas.items():
					if category == cate:
						continue
					another_urls += urls

				if len(another_urls) <= 0:
				 	continue

				for i in range(len(urls)):
					for j in range(i, len(urls)):
						for _ in range(1):
							url_triples.append((urls[i], urls[j], another_urls[random.randrange(len(another_urls))]))

			return url_triples

		dict_rnn_input = {
			'train': generate_triples(train_datas),
			'test': generate_triples(test_datas),
		}

		return dict_rnn_input

	# alpha is a hyperparameter for balancing
	def loss_f(self, x, h, y, alpha=0.3):
		loss = 0.0
		for i in range(3):
			loss += F.binary_cross_entropy(y[i], torch.sigmoid(x[i]))

		# log(1+exp(h0.*h2−h0.*h1))
		loss += alpha * torch.mean(torch.log(
					1+torch.exp(torch.sum(h[0]*h[2], dim=1) - torch.sum(h[0]*h[1], dim=1))
					)
				)
		return loss

	def train(self):
		self._vae.train()
		train_loss = 0.0
		batch_count = len(self._train_dataloader)
		
		for batch_idx, train_input in enumerate(self._train_dataloader):
			train_input = train_input.to(self._device)
			x = [train_input[:,0,:], train_input[:,1,:], train_input[:,2,:]]

			h, y = self._vae(*x)

			self._optimizer.zero_grad()
			self._vae.zero_grad()

			loss = self.loss_f(x, h, y)
			loss.backward()
			self._optimizer.step()

			train_loss += loss.item()

		return train_loss / batch_count

	def test(self):
		self._vae.eval()

		test_loss = 0.0
		batch_count = len(self._test_dataloader)
		
		for batch_idx, test_input in enumerate(self._test_dataloader):
			test_input = test_input.to(self._device)
			x = [test_input[:,0,:], test_input[:,1,:], test_input[:,2,:]]

			h, y = self._vae(*x)

			loss = self.loss_f(x, h, y)

			test_loss += loss.item()
		return test_loss / batch_count

	def generate_article_vector(self, x):
		self._vae.eval()
		x = x.to(self._device)

		return self._vae.inference(x)

	def save_model(self, epoch, test_loss):
		dict_states = {
			'epoch': epoch,
			'test_loss': test_loss,
			'vae': self._vae.state_dict(),
			'optimizer': self._optimizer.state_dict(),
		}

		torch.save(dict_states, self._saved_model_path)

	def load_model(self):
		if not os.path.exists(self._saved_model_path):
			return 0, sys.float_info.max
		dict_states = torch.load(self._saved_model_path)
		self._vae.load_state_dict(dict_states['vae'])
		self._optimizer.load_state_dict(dict_states['optimizer'])

		return dict_states['epoch'], dict_states['test_loss']

class  UserRepresentationModel(nn.Module):
	def __init__(self, embed_size, hidden_size, num_layers):
		super(UserRepresentationModel, self).__init__()

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

class UserRepresentation(object):
	def __init__(self, ws_path, torch_input_path, dict_url2vec):
		super(UserRepresentation, self).__init__()

		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._ws_path = ws_path

		dim_article = len(next(iter(dict_url2vec.values())))
		hidden_size = 512
		num_layers = 2
		learning_rate = 0.01

		dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)
		self._rnn_input = AdressaRNNInput(dict_rnn_input_path, dict_url2vec)

		self._train_dataloader, self._test_dataloader = \
								self.get_dataloader(dict_url2vec)

		self._model = UserRepresentationModel(dim_article, \
									hidden_size, num_layers).to(self._device)
		self._model.apply(weights_init)

#self._optimizer = torch.optim.SGD(self._model.parameters(), lr=learning_rate, momentum=0.9)
		self._criterion = nn.MSELoss()
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)

		self._saved_model_path = self._ws_path + '/user_representation.pth.tar'

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
#cand_matrix = np.matrix(cand_embed)
				cand_matrix = np_sigmoid(np.matrix(cand_embed))

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
					   (options.u2v_path == None) or (options.u2i_path == None) or \
						(options.ws_path == None):
		return

	torch_input_path = options.input
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = options.u2v_path
	url2info_path = options.u2i_path
	ws_path = options.ws_path

#os.system('rm -rf {}'.format(ws_path))
	os.system('mkdir -p {}'.format(ws_path))

	print('Loading url2vec : start')
	dict_url2vec = load_json(url2vec_path)
	print('Loading url2vec : end')

	print('Loading url2info : start')
	dict_url2info = load_json(url2info_path)
	print('Loading url2info : end')

	# Generate article_representation model
	article_representation = ArticleRepresentation(dict_url2vec, dict_url2info, ws_path)

	start_epoch, best_test_loss = article_representation.load_model()
	total_epoch = 2000
#if start_epoch < total_epoch:
	if start_epoch < 1:
		endure = 0
		for epoch in range(start_epoch, total_epoch):
			if endure > 10:
				print('Early stop!')
				break

			train_loss = article_representation.train()
			test_loss = article_representation.test()

			print('epoch {} - train loss({}) test loss({})'.format(epoch, train_loss, test_loss))

			if epoch % 5 == 0:
				if test_loss < best_test_loss:
					best_test_loss = test_loss
					endure = 0
					article_representation.save_model(epoch, test_loss)
					print('Model saved! - test loss({})'.format(test_loss))
				else:
					endure += 1

	print('Exchange url2vec - start')
	for url, vec in dict_url2vec.items():
		vec_t = torch.FloatTensor(vec)
		y = article_representation.generate_article_vector(vec_t)
		dict_url2vec[url] = y.tolist()
	print('Exchange url2vec - end')

	# Generate user_representation model
	user_representation = UserRepresentation(ws_path, torch_input_path, dict_url2vec)

	start_epoch, best_test_loss = user_representation.load_model()
	total_epoch = 1000
	if start_epoch < total_epoch:
		endure = 0
		for epoch in range(start_epoch, total_epoch):
			if endure > 3:
				print('Early stop!')
				break

			train_loss = user_representation.train()
			test_loss = user_representation.test()
			mrr_20 = user_representation.test_mrr_20()

			print('epoch {} - train loss({}) test loss({}) test mrr_20({})'.format(
						epoch, train_loss, test_loss, mrr_20))

			if epoch % 5 == 0:
				if test_loss < best_test_loss:
					best_test_loss = test_loss
					endure = 0
					user_representation.save_model(epoch, test_loss)
					print('Model saved! - test loss({})'.format(test_loss))
				else:
					endure += 1


if __name__ == '__main__':
	main()
