
import json
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-o', '--seq_graph_path', dest='seq_graph_path', type='string', default=None)

class SequenceGraph(object):

	def __init__(self, rnn_input_json_path, url2vec_path,
			embedding_dimension):

		self._embedding_dimension = embedding_dimension

		with open(rnn_input_json_path, 'r') as f_rnn_input:
			self._dict_rnn_input = json.load(f_rnn_input)

		with open(url2vec_path, 'r') as f_u2v:
			self._dict_url2vec = json.load(f_u2v)

		self._graph = {}

		train_sequences = self._dict_rnn_input['dataset']['train']
		# Initilization
		for _, _, sequence in train_sequences:
			for idx in sequence:
				if self._graph.get(str(idx), None) != None:
					continue
				self._graph[str(idx)] = {
					'prev': {},
					'next': {},
				}

		# count the adjust nodes
		for _, _, sequence in train_sequences:
			key_prev_idx = None
			for idx in sequence:
				key_idx = str(idx)
				if key_prev_idx != None:
					self._graph[key_idx]['prev'][key_prev_idx] = self._graph[key_idx]['prev'].get(key_prev_idx, 0) + 1
					self._graph[key_prev_idx]['next'][key_idx] = self._graph[key_prev_idx]['next'].get(key_idx, 0) + 1

				key_prev_idx = key_idx

		def extract_top_k(dict_data, k):
			sorted_list = sorted([(n_id, n_count) for n_id, n_count in dict_data.items()], key=lambda x: int(x[1]))
			count_sum = 0
			for n_id, n_count in sorted_list:
				count_sum += n_count
			if len(sorted_list) > k:
				sorted_list = sorted_list[-k:]

			sorted_list = [(n_id, float(n_count)/float(count_sum)) for n_id, n_count, in sorted_list]
			
			return sorted_list

		topk = 3
		self._graph_topk = {}
		# Top n of adjacent nodes and normalization
		for node_id, dict_adjacent in self._graph.items():
			self._graph_topk[node_id] = {}
			self._graph_topk[node_id]['prev'] = extract_top_k(self._graph[node_id]['prev'], topk)
			self._graph_topk[node_id]['next'] = extract_top_k(self._graph[node_id]['next'], topk)

	def idx2vec(self, idx):
		key_idx = idx if type(idx) is str else str(idx)
		return self._dict_url2vec[self._dict_rnn_input['idx2url'][key_idx]]

	def similarity(self, idx_0, idx_1):
		def weighted_sum_of_similarity(indices_0, indices_1):
			ret = 0.0
			for idx_ in indices_0:
				for idx__ in indices_1:
					# TODO - similarity from d2v
					ret += np.dot(self.idx2vec(idx_), self.idx2vec(idx__))
			return ret

		weighted_sum_of_similarity(self._graph_topk[str(idx_0)]['prev'], 
				self._graph_topk[str(idx_1)]['prev'])

		weighted_sum_of_similarity(self._graph_topk[str(idx_0)]['next'], 
				self._graph_topk[str(idx_1)]['next'])
			
		self._graph_topk[str(idx_0)]
		self.idx2vec(idx_0)
		self.idx2vec(idx_1)


def main():
	options, args = parser.parse_args()
	if (options.input == None) or (options.d2v_embed == None) or (options.u2v_path == None):
		return

	torch_input_path = options.input
	rnn_input_json_path = '{}/torch_rnn_input.dict'.format(torch_input_path)

	embedding_dimension = int(options.d2v_embed)
	url2vec_path = options.u2v_path

	sg = SequenceGraph(rnn_input_json_path, url2vec_path, embedding_dimension)


if __name__ == '__main__':
	main()

