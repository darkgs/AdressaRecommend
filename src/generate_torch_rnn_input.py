
import os
import json
import itertools

import torch
import numpy as np

from optparse import OptionParser
from multiprocessing import Pool

from ad_util import get_files_under_path
from ad_util import write_log

parser = OptionParser()
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)
parser.add_option('-o', '--output_dir_path', dest='output_dir_path', type='string', default=None)

dict_per_user = {}
list_per_time = []

dict_url_idx = {}
#dict_url_vec = {}

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

	dict_url_idx['url_pad'] = 0
	cur_idx = 1
	for url, _ in dict_url_idx.items():
		if url == 'url_pad':
			continue
		dict_url_idx[url] = cur_idx
		cur_idx += 1


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
			time_sequence = [timestamp for timestamp, url in sequence]

			dict_data[user_id] = {
				'start_time': start_time,
				'end_time': end_time,
				'sequence': idx_sequence,
				'time_sequence': time_sequence,
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
					dict_data['sequence'], dict_data['time_sequence'])
			merged_sequences.append(sequence_entry)

	merged_sequences.sort(key=lambda x:x[0])


#def load_url2vec(url2vec_path=None):
#	global dict_url_vec
#
#	dict_url_vec = {}
#	if url2vec_path == None:
#		return
#
#	with open(url2vec_path, 'r') as f_u2v:
#		dict_url_vec = json.load(f_u2v)
#

def extract_current_popular_indices(dict_time_idx, item_count=50, window_size=60*60*3):
	dict_trendy_idx = {}
	def generate_trendy_items(dict_target, padding):
		ret = sorted(dict_target.items(), key=lambda x: x[1], reverse=True)
#assert(len(ret) > 10)
		assert(len(ret) > 0)
		if len(ret) < item_count:
			ret += [(padding,0)] * (item_count - len(ret))

		return ret[:item_count]

	prev_timestamp = list_per_time[0][0]
	cur_timestamp = prev_timestamp
	dict_cur_trendy = {}

	# Initialize setting
	while cur_timestamp != None and (int(cur_timestamp) - int(prev_timestamp)) < window_size:
		for idx, count in dict_time_idx[cur_timestamp]['indices'].items():
			dict_cur_trendy[idx] = dict_cur_trendy.get(idx, 0) + count

		cur_timestamp = dict_time_idx[cur_timestamp]['next_time']

	copy_timestamp = prev_timestamp
	while(copy_timestamp is not cur_timestamp):
		dict_trendy_idx[copy_timestamp] = generate_trendy_items(dict_cur_trendy, dict_url_idx['url_pad'])

		copy_timestamp = dict_time_idx[copy_timestamp]['next_time']
	dict_trendy_idx[cur_timestamp] = generate_trendy_items(dict_cur_trendy, dict_url_idx['url_pad'])

	# main step
	while(True):
		# move cur
		cur_timestamp = dict_time_idx[cur_timestamp]['next_time']

		if cur_timestamp == None:
			break

		for idx, count in dict_time_idx[cur_timestamp]['indices'].items():
			dict_cur_trendy[idx] = dict_cur_trendy.get(idx, 0) + count

		# move prev
		to_be_removed = []
		while prev_timestamp != None and int(cur_timestamp) > int(prev_timestamp) and \
								(int(cur_timestamp) - int(prev_timestamp)) > window_size:

			for idx, count in dict_time_idx[prev_timestamp]['indices'].items():
				dict_cur_trendy[idx] = dict_cur_trendy.get(idx, 0) - count
				if dict_cur_trendy[idx] <= 0:
					to_be_removed.append(idx)

			prev_timestamp = dict_time_idx[prev_timestamp]['next_time']

		for idx in to_be_removed:
			dict_cur_trendy.pop(idx, None)

		# Update trendy data
		dict_trendy_idx[cur_timestamp] = generate_trendy_items(dict_cur_trendy, dict_url_idx['url_pad'])

	return dict_trendy_idx

def extract_recency_indices(dict_time_idx, item_count=50, window_size=60*60):
	dict_indices = {}
	def generate_recency_items(dict_target, padding):
		ret = sorted(dict_target.items(), key=lambda x: x[1], reverse=True)
#assert(len(ret) > 10)
		assert(len(ret) > 0)
		if len(ret) < item_count:
			ret += [(padding,0)] * (item_count - len(ret))

		return ret[:item_count]

	prev_timestamp = list_per_time[0][0]
	cur_timestamp = prev_timestamp
	dict_cur_trendy = {}

	# Initialize setting
	while cur_timestamp != None and (int(cur_timestamp) - int(prev_timestamp)) < window_size:
		for idx, count in dict_time_idx[cur_timestamp]['indices'].items():
			dict_cur_trendy[idx] = int(cur_timestamp)

		cur_timestamp = dict_time_idx[cur_timestamp]['next_time']

	copy_timestamp = prev_timestamp
	while(copy_timestamp is not cur_timestamp):
		dict_indices[copy_timestamp] = generate_recency_items(dict_cur_trendy, dict_url_idx['url_pad'])

		copy_timestamp = dict_time_idx[copy_timestamp]['next_time']
	dict_indices[cur_timestamp] = generate_recency_items(dict_cur_trendy, dict_url_idx['url_pad'])

	# main step
	while(True):
		# move cur
		cur_timestamp = dict_time_idx[cur_timestamp]['next_time']

		if cur_timestamp == None:
			break

		for idx, count in dict_time_idx[cur_timestamp]['indices'].items():
			dict_cur_trendy[idx] = cur_timestamp

		# move prev
		to_be_removed = []
		while prev_timestamp != None and int(cur_timestamp) > int(prev_timestamp) and (int(cur_timestamp) - int(prev_timestamp)) > window_size:

			for idx, timestamp in dict_time_idx[prev_timestamp]['indices'].items():
				if dict_cur_trendy.get(idx, None) != None and dict_cur_trendy[idx] <= int(prev_timestamp):
					to_be_removed.append(idx)

			prev_timestamp = dict_time_idx[prev_timestamp]['next_time']

		for idx in to_be_removed:
			dict_cur_trendy.pop(idx, None)

		# Update trendy data
		dict_indices[cur_timestamp] = generate_recency_items(dict_cur_trendy, dict_url_idx['url_pad'])

	return dict_indices

def generate_torch_rnn_input():
	global merged_sequences, dict_url_idx, list_per_time, output_dir_path

	# idx2url
	dict_idx2url = {idx:url for url, idx in dict_url_idx.items()}

	# sequence_datas
	total_seq_count = len(merged_sequences)

	division_infos = [
		('train', 0, int(total_seq_count*0.8)),
		('valid', int(total_seq_count*0.8), int(total_seq_count*0.9)),
		('test', int(total_seq_count*0.9), int(total_seq_count)),
	]

	dict_seq_datas = {}
	for dataset_name, idx_st, idx_ed in division_infos:
		dict_seq_datas[dataset_name] = merged_sequences[idx_st:idx_ed]

	# shuffle and clip test set
	dict_seq_datas['test'] = [ dict_seq_datas['test'][idx] for idx in np.random.permutation(len(dict_seq_datas['test'])).tolist()[:5000]]

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
		dict_time_idx[timestamp]['indices'][idx_of_url] = \
			dict_time_idx[timestamp]['indices'].get(idx_of_url, 0) + 1

		prev_timestamp = timestamp

	# trendy
	dict_trendy_idx = extract_current_popular_indices(dict_time_idx, item_count=100, window_size=60*60*3)

	# recency
	dict_recency_idx = extract_recency_indices(dict_time_idx, item_count=15, window_size=60*60)

	# save
	dict_torch_rnn_input = {
		'dataset': dict_seq_datas,
#		'idx2vec': dict_idx_vec,
		'idx2url': dict_idx2url,
		'time_idx': dict_time_idx,
		'pad_idx': dict_url_idx['url_pad'],
#		'embedding_dimension': embeding_dimension,
		'trendy_idx': dict_trendy_idx,
		'recency_idx': dict_recency_idx,
	}

	with open('{}/torch_rnn_input.dict'.format(output_dir_path), 'w') as f_extra:
		json.dump(dict_torch_rnn_input, f_extra)


def main():
	global dict_per_user, separated_output_dir_path, output_dir_path

	options, args = parser.parse_args()

	if (options.data_path == None) or (options.output_dir_path == None):
		return

	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'

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

#	print('Loading url2vec : start')
#	load_url2vec(url2vec_path=url2vec_path)
#	print('Loading url2vec : end')

	print('Generate torch_rnn_input : start')
	generate_torch_rnn_input()
	print('Generate torch_rnn_input : end')


if __name__ == '__main__':
	main()
