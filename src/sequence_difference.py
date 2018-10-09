
import os
import random
import time
import math

import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset  # For custom datasets

from optparse import OptionParser

from ad_util import weights_init
from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-o', '--output_dir_path', dest='output_dir_path', type='string', default=None)

class SequenceGraph(object):

	def __init__(self, rnn_input_json_path, url2vec_path,
			embedding_dimension, raw_dataset_path, do_stat=False):

		self._embedding_dimension = embedding_dimension

		if not do_stat and os.path.exists(raw_dataset_path):
			return

		with open(rnn_input_json_path, 'r') as f_rnn_input:
		    self._dict_rnn_input = json.load(f_rnn_input)

		with open(url2vec_path, 'r') as f_u2v:
		    self._dict_url2vec = json.load(f_u2v)

		self._graph = {}
		train_sequences = self._dict_rnn_input['dataset']['train']
		# Initilization
		indices_set = set([])
		for _, _, sequence in train_sequences:
			for idx in sequence:
				indices_set.update([idx])
				if self._graph.get(str(idx), None) != None:
					continue
				self._graph[str(idx)] = {
					'prev': {},
					'next': {},
				}
		self._indices = list(indices_set)

		# count the adjust nodes
		for _, _, sequence in train_sequences:
			key_prev_idx = None
			for idx in sequence:
				key_idx = str(idx)
				if key_prev_idx != None:
					self._graph[key_idx]['prev'][key_prev_idx] = self._graph[key_idx]['prev'].get(key_prev_idx, 0) + 1
					self._graph[key_prev_idx]['next'][key_idx] = self._graph[key_prev_idx]['next'].get(key_idx, 0) + 1

				key_prev_idx = key_idx

		def do_stat_for_graph(graph):
			prev_sum = 0
			prev_pow2_sum = 0
			prev_max = 0
			next_sum = 0
			next_pow2_sum = 0
			next_max = 0
			total_count = 0
			for idx, data in graph.items():
				total_count += 1
				prev_count = 0
				for _, w in data['prev'].items():
					prev_count += w
				prev_sum += prev_count
				prev_pow2_sum += prev_count ** 2
				prev_max = max(prev_max, prev_count)

				next_count = 0
				for _, w in data['next'].items():
					next_count += w
				next_sum += next_count
				next_pow2_sum += next_count ** 2
				next_max = max(next_max, next_count)

			prev_avg = float(prev_sum)/float(total_count)
			prev_pow2_avg = float(prev_pow2_sum)/float(total_count)
			prev_var = prev_pow2_avg - (prev_avg ** 2)

			next_avg = float(next_sum)/float(total_count)
			next_pow2_avg = float(next_pow2_sum)/float(total_count)
			next_var = next_pow2_avg - (next_avg ** 2)

			print('Total count : {}'.format(total_count))
			print('Prev : avg({}) var({}) std({}) max({})'.format(prev_avg, prev_var, math.sqrt(prev_var), prev_max))
			print('Next : avg({}) var({}) std({}) max({})'.format(next_avg, next_var, math.sqrt(next_var), next_max))

		if do_stat:
			print('== Full Graph ===')
			do_stat_for_graph(self._graph)

		def extract_top_k(dict_data, k):
			sorted_list = sorted([(n_id, n_count) for n_id, n_count in dict_data.items()], key=lambda x: int(x[1]))
			count_sum = 0
			for n_id, n_count in sorted_list:
				count_sum += n_count
			if len(sorted_list) > k:
				sorted_list = sorted_list[-k:]

			sorted_list = [(n_id, float(n_count)/float(count_sum)) for n_id, n_count, in sorted_list]
			
			return sorted_list

		topk = 10
		self._graph_topk = {}
		# Top n of adjacent nodes and normalization
		for node_id, dict_adjacent in self._graph.items():
			self._graph_topk[node_id] = {}
			self._graph_topk[node_id]['prev'] = extract_top_k(self._graph[node_id]['prev'], topk)
			self._graph_topk[node_id]['next'] = extract_top_k(self._graph[node_id]['next'], topk)

		def do_stat_for_topk(graph):
			prev_sum = 0.0
			next_sum = 0.0
			total_count = 0
			for idx, data in graph.items():
				total_count += 1
				for _, w in data['prev']:
					prev_sum += w
				for _, w in data['next']:
					next_sum += w

			print('Prev cover({})'.format(prev_sum/float(total_count)))
			print('Next cover({})'.format(next_sum/float(total_count)))

		if do_stat:
			print('== TOP{} Graph ==='.format(topk))
			do_stat_for_topk(self._graph_topk)

	def idx2vec(self, idx):
		key_idx = idx if type(idx) is str else str(idx)
		return self._dict_url2vec[self._dict_rnn_input['idx2url'][key_idx]]

	def test_diff(self):
		idx_0, idx_1 = self.random_idx_pair()

		return self.difference(idx_0, idx_1)

	def random_idx_pair(self):
		idx_0 = self._indices[random.randrange(len(self._indices))]
		idx_1 = self._indices[random.randrange(len(self._indices))]

		idx_0_check = len(self._graph_topk[str(idx_0)]['prev']) > 0 and len(self._graph_topk[str(idx_0)]['next']) > 0
		idx_1_check = len(self._graph_topk[str(idx_1)]['prev']) > 0 and len(self._graph_topk[str(idx_1)]['next']) > 0

		return (idx_0, idx_1) if (idx_0 != idx_1 and idx_0_check and idx_1_check) else self.random_idx_pair()

	def difference(self, idx_0, idx_1):
		def weighted_sum_of_difference(datas_0, datas_1):
			ret = 0.0
			for idx_0, w_0 in datas_0:
				for idx_1, w_1 in datas_1:
					mse = np.square(np.subtract(self.idx2vec(idx_0), self.idx2vec(idx_1))).mean()
					ret += mse * w_0 * w_1

			return ret

		prev_diff = weighted_sum_of_difference(self._graph_topk[str(idx_0)]['prev'], 
				self._graph_topk[str(idx_1)]['prev'])

		next_diff = weighted_sum_of_difference(self._graph_topk[str(idx_0)]['next'], 
				self._graph_topk[str(idx_1)]['next'])

		return prev_diff + next_diff

	def generate_dataset(self, raw_dataset_path):
		if os.path.exists(raw_dataset_path):
			return

		diff_of_docs = []
		raw_dataset_path_tmp = raw_dataset_path + '_tmp'
		if os.path.exists(raw_dataset_path_tmp):
			with open(raw_dataset_path_tmp, 'r') as f_data:
				dict_dataset = json.load(f_data)

			diff_of_docs += dict_dataset['diff_of_docs']

		# 10min for /1000
		target_data_size = int(len(self._indices) **2 / 1000 * 3)

		while(len(diff_of_docs) < target_data_size):
			start_time = time.time()
			diff_of_docs += [(idx_0, idx_1, self.difference(idx_0, idx_1)) \
				for idx_0, idx_1 in [self.random_idx_pair() for _ in range(200000)]]

			dict_dataset = {
				'diff_of_docs': diff_of_docs,
			}

			with open(raw_dataset_path_tmp + '_t', 'w') as f_data:
				json.dump(dict_dataset, f_data)
			os.system('mv {} {}'.format(raw_dataset_path_tmp + '_t', raw_dataset_path_tmp))

			print('Generate datasets {}/{} tooks {}'.format(len(diff_of_docs), target_data_size, time.time() - start_time))

#		if os.path.exists(raw_dataset_path_tmp):
#			os.system('rm {}'.format(raw_dataset_path_tmp))

		dict_dataset = {
			'idx2vec': {idx:self.idx2vec(idx) \
				for idx, _ in self._dict_rnn_input['idx2url'].items()},
			'diff_of_docs': diff_of_docs,
		}

		with open(raw_dataset_path, 'w') as f_data:
			json.dump(dict_dataset, f_data)


class ArticleDataset(Dataset):
	def __init__(self, dict_dataset):
		self._dict_dataset = dict_dataset
		self._data_len = len(self._dict_dataset)

	def __getitem__(self, index):
		return self._dict_dataset[index]

	def __len__(self):
		return self._data_len


class ArticleInputTorch(object):
	def __init__(self, raw_dataset_path):
		with open(raw_dataset_path, 'r') as f_data:
			raw_dataset = json.load(f_data)
			diff_of_docs = [ (raw_dataset['idx2vec'][str(idx_0)], 
					raw_dataset['idx2vec'][str(idx_1)], diff) \
				for idx_0, idx_1, diff in raw_dataset['diff_of_docs'] ]

		total_dataset_count = len(diff_of_docs)

		division_infos = [
			('train', 0, int(total_dataset_count * 8 / 10)),
			('valid', int(total_dataset_count * 8 / 10), int(total_dataset_count * 9 / 10)),
			('test', int(total_dataset_count * 9 / 10), total_dataset_count),
		]

		self._dataset = {}
		for dataset_name, idx_st, idx_ed in division_infos:
			self._dataset[dataset_name] = ArticleDataset(diff_of_docs[idx_st:idx_ed])

	def get_dataset(self, data_type='test'):
		return self._dataset[data_type]

	@staticmethod
	def collate(batch):
		vec_0s = []
		vec_1s = []
		diffs = []
		for vec_0, vec_1, diff in batch:
			vec_0s.append(vec_0)
			vec_1s.append(vec_1)
			diffs.append(diff)

		return torch.FloatTensor(vec_0s), torch.FloatTensor(vec_1s), \
			torch.FloatTensor(diffs)

class ArticleEncoderModel(nn.Module):
	def __init__(self, embed_size, output_size):
		super(ArticleEncoderModel, self).__init__()

		hidden_size = int(embed_size * 3 / 2)

		self.linear = nn.Sequential(
			nn.Linear(embed_size, hidden_size),
#			nn.BatchNorm1d(hidden_size),
#			nn.ReLU(),
			nn.Linear(hidden_size, embed_size),
			nn.BatchNorm1d(embed_size),
			nn.ReLU(),
			nn.Linear(embed_size, output_size),
			nn.BatchNorm1d(output_size),
			nn.ReLU(),
		)
		self.drop = nn.Dropout(0.5)

	def forward(self, x, y):
		x = self.linear(x)

		y = self.linear(y)

		if self.training:
			x = self.drop(x)
			y = self.drop(y)

		return x, y


class ArticleEncoder(object):
	def __init__(self, torch_input, embed_size, output_size):
		# generate DataLoaders
		self._trainloader = torch.utils.data.DataLoader(torch_input.get_dataset(data_type='train'),
				batch_size=256, shuffle=True, num_workers=16,
				collate_fn=ArticleInputTorch.collate)

		self._validloader = torch.utils.data.DataLoader(torch_input.get_dataset(data_type='valid'),
				batch_size=256, shuffle=False, num_workers=16,
				collate_fn=ArticleInputTorch.collate)

		self._testloader = torch.utils.data.DataLoader(torch_input.get_dataset(data_type='test'),
				batch_size=128, shuffle=False, num_workers=4,
				collate_fn=ArticleInputTorch.collate)

		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self._model = ArticleEncoderModel(embed_size, output_size).to(self._device)
		self._model.apply(weights_init)

		self._criterion = nn.MSELoss()
		self._optimizer = optim.Adam(self._model.parameters(), lr=0.001)

	def train(self):
		model = self._model
		device = self._device
		dataloader = self._trainloader
		criterion = self._criterion
		optimizer = self._optimizer

		model.train()
		loss_sum = 0.0
		for i, data in enumerate(dataloader, 0):
			model.zero_grad()
			optimizer.zero_grad()

			vec_0s, vec_1s, diff = data
			vec_0s = vec_0s.to(device)
			vec_1s = vec_1s.to(device)
			diff = diff.to(device)

			vec_0s_embed, vec_1s_embed = model(vec_0s, vec_1s)
			mse = torch.pow(torch.sub(vec_0s_embed, vec_1s_embed), 2).mean(dim=1)
			loss = criterion(torch.norm(mse, dim=0), torch.norm(diff, dim=0))
			loss_sum += loss.item()

			loss.backward()
			optimizer.step()

		return loss_sum / float(len(dataloader.dataset))

	def valid(self):
		model = self._model
		device = self._device
		dataloader = self._validloader
		criterion = self._criterion

		model.eval()
		loss_sum = 0.0
		for i, data in enumerate(dataloader, 0):
			model.zero_grad()
			vec_0s, vec_1s, diff = data
			vec_0s = vec_0s.to(device)
			vec_1s = vec_1s.to(device)
			diff = diff.to(device)

			vec_0s_embed, vec_1s_embed = model(vec_0s, vec_1s)
			mse = torch.pow(torch.sub(vec_0s_embed, vec_1s_embed), 2).mean(dim=1)

			loss = criterion(torch.norm(mse, dim=0), torch.norm(diff, dim=0))

			loss_sum += loss.item()

		return loss_sum / float(len(dataloader.dataset))

	def save(self, epoch=0, model_path=None):
		if model_path == None:
			return

		states = {
			'epoch': epoch,
			'model': self._model.state_dict(),
			'optimizer': self._optimizer.state_dict(),
		}
		
		torch.save(states, model_path)
		write_log('Model saved! - {}'.format(model_path))

	def load(self, epoch=None, model_path=None):
		if model_path == None or not os.path.exists(model_path):
			return 0

		states = torch.load(model_path)

		self._model.load_state_dict(states['model'])
		self._optimizer.load_state_dict(states['optimizer'])	

		write_log('Model loaded!! - {}'.format(model_path))

		return states['epoch']

def main():
	options, args = parser.parse_args()
	if (options.input == None) or (options.d2v_embed == None) or \
                (options.u2v_path == None) or (options.output_dir_path == None):
	    return

	torch_input_path = options.input
	rnn_input_json_path = '{}/torch_rnn_input.dict'.format(torch_input_path)

	embedding_dimension = int(options.d2v_embed)
	url2vec_path = options.u2v_path
	
	output_dir_path = options.output_dir_path

	if not os.path.exists(output_dir_path):
		os.system('mkdir -p {}'.format(output_dir_path))

	raw_dataset_path = '{}/raw_dataset.json'.format(output_dir_path)
	model_path = '{}/article_encoder.pth.tar'.format(output_dir_path)

	sg = SequenceGraph(rnn_input_json_path, url2vec_path,
			embedding_dimension, raw_dataset_path)
	sg.generate_dataset(raw_dataset_path)

	torch_input = ArticleInputTorch(raw_dataset_path)
	
	new_article_dimension = 256
	ae = ArticleEncoder(torch_input, embedding_dimension, new_article_dimension)

	start_epoch = ae.load(model_path)
	for epoch in range(start_epoch, 100):
		train_loss = ae.train()
		valid_loss = ae.valid()
#		write_log('epoch {} : train loss({}), valid loss({})'.format(epoch, train_loss, valid_loss))
		print('epoch {} : train loss({}), valid loss({})'.format(epoch, train_loss, valid_loss))

		if epoch > 0 and epoch % 10 == 0:
			ae.save(epoch, '{}_e{}'.format(model_path, epoch))


if __name__ == '__main__':
	main()

