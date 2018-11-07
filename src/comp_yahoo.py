
import os
import json
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset  # For custom datasets

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-a', '--u2i_path', dest='u2i_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)

def load_json(dict_path=None):
	dict_ret = {}

	if dict_path == None:
		return

	with open(dict_path, 'r') as f_dict:
		dict_ret = json.load(f_dict)

	return dict_ret


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


class ArticleRepresentationDataset(Dataset):
	def __init__(self, dataset):
		self._dataset = dataset

	def __getitem__(self, index):
		return self._dataset[index]

	def __len__(self):
		return len(self._dataset)


class ArticleRepresentation():
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

		self._optimizer = torch.optim.SGD(self._vae.parameters(), lr=learning_rate, momentum=0.9)

	def get_dataloader(self):
		saved_rnn_input_path = self._ws_path + '/rnn_input_article.json'

		if not os.path.exists(saved_rnn_input_path):
			print("Newly generating!")
			dict_rnn_input = self.generate_rnn_input()
			with open(saved_rnn_input_path, 'w') as f_input:
				json.dump(dict_rnn_input, f_input)
		else:
			print("Load from file!")
			with open(saved_rnn_input_path, 'r') as f_input:
				dict_rnn_input = json.load(f_input)

		def article_collate(batch):
			# batch * (url0, url1, url_diff)
			return torch.FloatTensor([ (self._dict_url2vec[url0], self._dict_url2vec[url1], 
					self._dict_url2vec[url_diff]) for url0, url1, url_diff in batch ])

		train_dataset = ArticleRepresentationDataset(dict_rnn_input['train'])
		test_dataset = ArticleRepresentationDataset(dict_rnn_input['test'])

		self._trainloader = torch.utils.data.DataLoader(train_dataset,
						batch_size=128, shuffle=True, num_workers=16,
						collate_fn=article_collate)

		self._testloader = torch.utils.data.DataLoader(test_dataset,
						batch_size=32, shuffle=True, num_workers=4,
						collate_fn=article_collate)


		return dict_rnn_input['train'], dict_rnn_input['test']

	def generate_rnn_input(self):
		stat = False
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
						for _ in range(10):
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
		batch_count = len(self._trainloader)
		
		for batch_idx, train_input in enumerate(self._trainloader):
			train_input = train_input.to(self._device)
			x = [train_input[:,0,:], train_input[:,1,:], train_input[:,2,:]]

			h, y = self._vae(*x)

			self._optimizer.zero_grad()
			loss = self.loss_f(x, h, y)
			loss.backward()
			self._optimizer.step()

			train_loss += loss.item()

		return train_loss / batch_count

	def test(self):
		self._vae.eval()

		test_loss = 0.0
		batch_count = len(self._testloader)
		
		for batch_idx, test_input in enumerate(self._testloader):
			test_input = test_input.to(self._device)
			x = [test_input[:,0,:], test_input[:,1,:], test_input[:,2,:]]

			h, y = self._vae(*x)

			loss = self.loss_f(x, h, y)

			test_loss += loss.item()
		return test_loss / batch_count


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

	article_representation = ArticleRepresentation(dict_url2vec, dict_url2info, ws_path)

	for epoch in range(100):
		train_loss = article_representation.train()
		test_loss = article_representation.test()

		print('epoch {} - train loss({}) test loss({})'.format(epoch, train_loss, test_loss))


if __name__ == '__main__':
	main()
