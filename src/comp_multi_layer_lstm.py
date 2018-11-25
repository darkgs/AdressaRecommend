
import os, sys

from optparse import OptionParser

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from adressa_dataset import AdressaRec

from ad_util import load_json

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)


class MultiLSTMModel(nn.Module):
	def __init__(self, embed_size):
		super(MultiLSTMModel, self).__init__()

		hidden_size = 384
		num_layers = 2

		self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
		self.linear = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, x, _, seq_lens):
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


def main():
	options, args = parser.parse_args()

	if (options.input == None) or (options.d2v_embed == None) or \
					   (options.u2v_path == None) or (options.ws_path == None):
		return

	torch_input_path = options.input
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = options.u2v_path
	ws_path = options.ws_path

	os.system('rm -rf {}'.format(ws_path))
	os.system('mkdir -p {}'.format(ws_path))

	print('Loading url2vec : start')
	dict_url2vec = load_json(url2vec_path)
	print('Loading url2vec : end')

	predictor = AdressaRec(MultiLSTMModel, ws_path, torch_input_path, dict_url2vec)

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

