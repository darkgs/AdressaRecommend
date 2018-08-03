
import os
import json
import itertools

import torch

from optparse import OptionParser
from multiprocessing import Pool

from ad_util import get_files_under_path
from ad_util import write_log

parser = OptionParser()
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-o', '--output_dir_path', dest='output_dir_path', type='string', default=None)

dict_per_user = {}
list_per_time = []

dict_url_idx = {}
dict_url_vec = {}

output_dir_path = None
separated_output_dir_path = None
merged_sequences = []


def load_per_datas(per_user_path=None, per_time_path=None):
	global dict_per_user, list_per_time

	dict_per_user = {}
	list_per_time = []

	if (per_user_path == None) or (per_time_path == None):
		return

	with open(per_user_path, 'r') as f_user:
		dict_per_user = json.load(f_user)

	with open(per_time_path, 'r') as f_time:
		list_per_time = json.load(f_time)


def generate_unique_url_idxs():
	global dict_per_user, dict_url_idx

	dict_url_idx = {}

	# "cx:i68bn3gbf0ql786n:1hyr7mridb1el": [[1483570820, "http://adressa.no/100sport/ballsport/byasen-fiasko-mot-tabelljumboen-228288b.html"]]
	for user_id, sequence in dict_per_user.items():
		for timestamp, url in sequence:
			dict_url_idx[url] = 0

	cur_idx = 0
	for url, _ in dict_url_idx.items():
		cur_idx += 1
		dict_url_idx[url] = cur_idx
	dict_url_idx['url_pad'] = 0


def separated_process(args=(-1, [])):
	global dict_per_user, dict_url_idx, separated_output_dir_path

	worker_id, user_ids = args

	dict_data = {}
	for user_id in user_ids:
		# remove duplication
		sequence = []

		# "cx:i68bn3gbf0ql786n:1hyr7mridb1el": [[1483570820, "http://adressa.no/100sport/ballsport/byasen-fiasko-mot-tabelljumboen-228288b.html"]]
		prev_url = None
		for seq_entry in dict_per_user[user_id]:
			timestamp, url = seq_entry
			if (prev_url == None) or (url != prev_url):
				prev_url = url
				sequence.append(seq_entry)

			seq_len = len(sequence)

			# Minimum valid sequence length
			if seq_len < 2:
				continue

			# Maximum valid sequence length
#			if seq_len > 20:
#				sequence = sequence[-20:]

			start_time = sequence[0][0]
			end_time = sequence[-1][0]

			idx_sequence = [dict_url_idx[url] for timestamp, url in sequence]

			dict_data[user_id] = {
				'start_time': start_time,
				'end_time': end_time,
				'sequence': idx_sequence,
			}

	with open('{}/{}_data.json'.format(separated_output_dir_path, worker_id), 'w') as f_out:
		json.dump(dict_data, f_out)


def generate_merged_sequences():
	global separated_output_dir_path, merged_sequences

	merged_sequences = []
	separated_files = get_files_under_path(separated_output_dir_path)

	for separated_file in separated_files:
		with open(separated_file, 'r') as f_dict:
			separated_dict = json.load(f_dict)

#		separated_dict[user_id] = {
#			'start_time': start_time,
#			'end_time': end_time,
#			'sequence': idx_sequence,
#		}

		for user_id, dict_data in separated_dict.items():
			sequence_entry = (dict_data['start_time'], dict_data['end_time'],
					dict_data['sequence'])
			merged_sequences.append(sequence_entry)

	merged_sequences.sort(key=lambda x:x[0])


def load_url2vec(url2vec_path=None):
	global dict_url_vec

	dict_url_vec = {}
	if url2vec_path == None:
		return

	with open(url2vec_path, 'r') as f_u2v:
		dict_url_vec = json.load(f_u2v)


def generate_torch_rnn_input():
	global merged_sequences, dict_url_idx, dict_url_vec, list_per_time, output_dir_path

	# idx2vec
	embeding_dimension = len(next(iter(dict_url_vec.items()))[1])
	dict_url_vec['url_pad'] = [float(0)] * embeding_dimension

	dict_idx_vec = {idx:dict_url_vec[url] for url, idx in dict_url_idx.items()}
	
	# sequence_datas
	total_seq_count = len(merged_sequences)

	division_infos = [
		('train', 0, int(total_seq_count * 8 / 10)),
		('valid', int(total_seq_count * 8 / 10), int(total_seq_count * 9 / 10)),
#		('test', int(total_seq_count * 9 / 10), total_seq_count),
		('test', total_seq_count - 1000, total_seq_count),
	]

	dict_seq_datas = {}
	for dataset_name, idx_st, idx_ed in division_infos:
		dict_seq_datas[dataset_name] = merged_sequences[idx_st:idx_ed]
#		dict_seq_datas[dataset_name]['vec'] = [
#			[dict_idx_vec[idx] for idx in sequence] \
#				for (timestamp_start, timestamp_end, sequence) \
#				in itertools.islice(merged_sequences, idx_st, idx_ed)
#		]

	# candidates
	dict_time_idx = {}

	prev_timestamp = None
	for (timestamp, user_id, url) in list_per_time:
		if prev_timestamp != timestamp:
			if prev_timestamp != None:
				dict_time_idx[prev_timestamp]['next_time'] = timestamp
			dict_time_idx[timestamp] = {
				'prev_time': prev_timestamp,
				'next_time': None,
				'indices': {},
			}

		idx_of_url = dict_url_idx[url]
		dict_time_idx[timestamp]['indices'][idx_of_url] = dict_time_idx[timestamp]['indices'].get(idx_of_url, 0) + 1

		prev_timestamp = timestamp

	# save
	dict_torch_rnn_input = {
		'dataset': dict_seq_datas,
		'idx2vec': dict_idx_vec,
		'time_idx': dict_time_idx,
		'pad_idx': dict_url_idx['url_pad'],
		'embedding_dimension': embeding_dimension,
	}

	with open('{}/torch_rnn_input.dict'.format(output_dir_path), 'w') as f_extra:
		json.dump(dict_torch_rnn_input, f_extra)


def main():
	global dict_per_user, separated_output_dir_path, output_dir_path

	options, args = parser.parse_args()

	if (options.data_path == None) or (options.u2v_path == None) or (options.output_dir_path == None):
		return

	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'

	url2vec_path = options.u2v_path
	output_dir_path = options.output_dir_path

	if not os.path.exists(output_dir_path):
		os.system('mkdir -p {}'.format(output_dir_path))

	print('Loading Sequence datas : start')
	load_per_datas(per_user_path=per_user_path, per_time_path=per_time_path)
	print('Loading Sequence datas : end')

	print('Generate unique url indices : start')
	generate_unique_url_idxs()
	print('Generate unique url indices : end')

	print('Seperated by user process : start')
	separated_output_dir_path = '{}/separated'.format(output_dir_path)
	if not os.path.exists(separated_output_dir_path):
		os.system('mkdir -p {}'.format(separated_output_dir_path))

	n_div = 100		# degree of separate
	n_multi = 5		# degree of multiprocess
	user_ids = [user_id for user_id, _ in dict_per_user.items()]
	works = [(i, user_ids[i::n_div]) for i in range(n_div)]

	pool = Pool(n_multi)
	pool.map(separated_process, works)
	pool = None
	print('Seperated by user process : end')

	print('Merging separated infos...')
	generate_merged_sequences()
	print('Merging separated infos... Done')

	print('Loading url2vec : start')
	load_url2vec(url2vec_path=url2vec_path)
	print('Loading url2vec : end')

	print('Generate torch_rnn_input : start')
	generate_torch_rnn_input()
	print('Generate torch_rnn_input : end')


if __name__ == '__main__':
	main()
