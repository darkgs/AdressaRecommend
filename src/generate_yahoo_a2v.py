
import os, sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset  # For custom datasets

from optparse import OptionParser

from ad_util import load_json
from ad_util import weights_init

parser = OptionParser()
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)

class ArticleModel(nn.Module):
	def __init__(self, dim_article, dim_h, corruption_rate=0.1):
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
	def __init__(self, dict_url2vec, dict_rnn_input, output_path, embedding_dimension):
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._dict_url2vec = dict_url2vec
		self._dict_rnn_input = dict_rnn_input

		self._dim_article = embedding_dimension
		self._dim_h = self._dim_article // 2
		learning_rate = 1e-3

		# Generate dataloader
		self._train_dataloader, self._test_dataloader = self.get_dataloader()
		
		# Model
		self._vae = ArticleModel(self._dim_article, self._dim_h).to(self._device)
		self._vae.apply(weights_init)
#self._optimizer = torch.optim.SGD(self._vae.parameters(), lr=learning_rate, momentum=0.9)
		self._optimizer = torch.optim.Adam(self._vae.parameters(), lr=learning_rate)

		self._saved_model_path = output_path

	def get_dataloader(self):
		def article_collate(batch):
			# batch * (url0, url1, url_diff)
			return torch.FloatTensor([ (self._dict_url2vec[url0], self._dict_url2vec[url1], 
					self._dict_url2vec[url_diff]) for url0, url1, url_diff in batch ])

		train_dataset = ArticleRepresentationDataset(self._dict_rnn_input['train'])
		test_dataset = ArticleRepresentationDataset(self._dict_rnn_input['test'])

		train_dataloader = torch.utils.data.DataLoader(train_dataset,
						batch_size=128, shuffle=True, num_workers=16,
						collate_fn=article_collate)

		test_dataloader = torch.utils.data.DataLoader(test_dataset,
						batch_size=32, shuffle=True, num_workers=4,
						collate_fn=article_collate)


		return train_dataloader, test_dataloader


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

	def do_train(self, total_epoch=1000, early_stop=10):
		start_epoch, best_test_loss = self.load_model()

		if start_epoch >= total_epoch:
			return best_test_loss

		endure = 0
		for epoch in range(start_epoch, total_epoch):
			start_time = time.time()
			if endure > early_stop:
				print('Early stop!')
				break

			train_loss = self.train()
			test_loss = self.test()

			print('epoch {} - train loss({:.8f}) test loss({:.8f}) tooks {:.2f}'.format(
					epoch, train_loss, test_loss, time.time() - start_time))

			if best_test_loss > test_loss:
				best_test_loss = test_loss
				endure = 0

				self.save_model(epoch, test_loss)
				print('Model saved! - best test loss({:.8f})'.format(best_test_loss))
			else:
				endure += 1

		return best_test_loss

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

		
def main():
	options, args = parser.parse_args()

	if (options.d2v_embed == None) or (options.u2v_path == None) \
						   or (options.input == None) or (options.output == None):
		return

	output_file_path = options.output
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)

	rnn_input_path = options.input

	print('Loading url2vec : start')
	dict_url2vec = load_json(url2vec_path)
	print('Loading url2vec : end')

	print('Loading a2v rnn input : start')
	dict_rnn_input = load_json(rnn_input_path)
	print('Loading a2v rnn input : end')

	ar = ArticleRepresentation(dict_url2vec, dict_rnn_input, output_file_path, embedding_dimension)
	ar.do_train()


if __name__ == '__main__':
	main()

