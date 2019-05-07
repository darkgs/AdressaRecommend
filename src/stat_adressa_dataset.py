
import os

from optparse import OptionParser

from adressa_dataset import AdressaRNNInput

from ad_util import load_json
from ad_util import option2str

from comp_lstm import SingleLSTMModel

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-w', '--ws_path', dest='ws_path', type='string', default=None)

parser.add_option('-l', '--learning_rate', dest='learning_rate', type='float', default=3e-3)
parser.add_option('-a', '--hidden_size', dest='hidden_size', type='int', default=1280)
parser.add_option('-b', '--num_layers', dest='num_layers', type='int', default=1)

parser.add_option('-t', '--trendy_count', dest='trendy_count', type='int', default=1)
parser.add_option('-r', '--recency_count', dest='recency_count', type='int', default=1)

#python3 src/stat_adressa_dataset.py -e $(D2V_EMBED) -u cache/article_to_vec.json -a $(BASE_PATH)/article_info.json 

def pop_skewness(dict_rnn_input):
	pop_datas = [0] * 101

	for data_type in ['train', 'valid', 'test']:
		for timestamp_start, timestamp_end, sequence, time_sequence in \
				dict_rnn_input['dataset'][data_type]:
			for timestamp, article_idx in zip(time_sequence, sequence):
				populars = [a for a, c in dict_rnn_input['trendy_idx'].get(str(timestamp), None)]
				pop_rank = 100
				if article_idx in populars:
					pop_rank = populars.index(article_idx)
				pop_rank = min(pop_rank, 100)
				pop_datas[pop_rank] += 1

	print(pop_datas)

def selected_times(dict_rnn_input):
	dict_datas = {}

	for data_type in ['train', 'valid', 'test']:
		for timestamp_start, timestamp_end, sequence, time_sequence in \
				dict_rnn_input['dataset'][data_type]:
			for timestamp, article_idx in zip(time_sequence, sequence):
				dict_datas[str(article_idx)] = dict_datas.get(str(article_idx), []) + [timestamp]

	size_data = 6 * 24
	ret_data = [0] * size_data
	overs = 0

	for _, times in dict_datas.items():
		first_shot = min(times)

		for t in times:
			diff = int((t - first_shot)/60/10)
			if diff < size_data:
				ret_data[diff%size_data] += 1
			else:
				overs += 1

	print(ret_data, overs/(sum(ret_data)+overs))

def main():
	options, args = parser.parse_args()

	if (options.input == None) or (options.d2v_embed == None) or \
					   (options.u2v_path == None) or (options.ws_path == None):
		return

	torch_input_path = options.input
	embedding_dimension = int(options.d2v_embed)
	url2vec_path = '{}_{}'.format(options.u2v_path, embedding_dimension)
	ws_path = options.ws_path

	if not os.path.exists(ws_path):
		os.system('mkdir -p {}'.format(ws_path))

	dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)
	dict_rnn_input = load_json(dict_rnn_input_path)

	selected_times(dict_rnn_input)

if __name__ == '__main__':
	main()

