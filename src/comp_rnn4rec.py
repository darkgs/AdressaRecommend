
import os

from optparse import OptionParser

import torch

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


class Rnn4Rec(object):
	def __init__(self, ws_path, torch_input_path, dict_url2vec):
		self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self._ws_path = ws_path

		self.get_dataloader(torch_input_path, dict_url2vec)

	def get_dataloader(self, torch_input_path, dict_url2vec):
		dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)

		rnn_input = AdressaRNNInput(dict_rnn_input_path, dict_url_vec)

		trainloader = torch.utils.data.DataLoader(rnn_input.get_dataset(data_type='train'),
				batch_size=128, shuffle=True, num_workers=16,
				collate_fn=adressa_collate)


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

	model = Rnn4Rec(ws_path, torch_input_path)


if __name__ == '__main__':
	main()
