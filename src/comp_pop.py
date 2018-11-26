
import os

from optparse import OptionParser

import torch
import torch.nn as nn

from adressa_dataset import AdressaRec

from ad_util import load_json

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)

class SingleLSTMModel(nn.Module):
	def __init__(self, embed_size):
		super(SingleLSTMModel, self).__init__()

		hidden_size = 768
		num_layers = 1

		self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_size, embed_size)
		self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

	def forward(self, x, _, seq_lens):
		batch_size = x.size(0)
		embed_size = x.size(2)

		x = pack(x, seq_lens, batch_first=True)
		outputs, _ = self.rnn(x)
		outputs, _ = unpack(outputs, batch_first=True)
		outputs = self.linear(outputs)

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

	predictor = AdressaRec(SingleLSTMModel, ws_path, torch_input_path, dict_url2vec)

	total_mrr = 0.0
	for i in range(10):
		mrr = predictor.pop_20()
		print('{}th : mrr({:.4f})'.format(i, mrr))
		total_mrr += mrr

	print('Final mrr : {:.4f}'.format(total_mrr/10))


if __name__ == '__main__':
	main()