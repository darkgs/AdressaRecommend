
import os
import json

from optparse import OptionParser

from multi_worker import MultiWorker

from ad_util import get_files_under_path
from ad_util import write_log

parser = OptionParser()
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)
parser.add_option('-w', '--w2v_path', dest='w2v_path', type='string', default=None)
parser.add_option('-o', '--output_file_path', dest='output_file_path', type='string', default=None)

dict_w2v = None
dict_per_user = None
dict_url_idx = None
seperated_output_path = None

def generate_unique_url_idxs():
	global dict_per_user

	dict_ret = {}

	# "cx:i68bn3gbf0ql786n:1hyr7mridb1el": [[1483570820, "http://adressa.no/100sport/ballsport/byasen-fiasko-mot-tabelljumboen-228288b.html"]]
	for user_id, sequence in dict_per_user.items():
		for timestamp, url in sequence:
			dict_ret[url] = 0

	cur_idx = 0
	for url in dict_ret.keys():
		cur_idx += 1
		dict_ret[url] = cur_idx
	dict_ret['url_pad'] = 0

	return dict_ret

def preprocess_rnn_input(args=(-1, [])):
	global dict_per_user, dict_url_idx, seperated_output_path

	max_seq_len = 20

	worker_id, user_ids = args

	write_log('worker({}) : start'.format(worker_id))
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

		if seq_len < 2:
			continue

		if seq_len > max_seq_len:
			sequence = sequence[-max_seq_len:]

		start_time = sequence[0][0]
		end_time = sequence[-1][0]
		idx_sequence = list(map(lambda x:dict_url_idx[x[1]], sequence))

		dict_data[user_id] = {
			'start_time': start_time,
			'end_time': end_time,
			'sequence': idx_sequence,
		}

	with open(seperated_output_path + '/' + str(worker_id) + '_data.json', 'w') as f_out:
		json.dump(dict_data, f_out)
	write_log('worker({}) : end'.format(worker_id))

def generate_rnn_input(seperated_input_path=None, output_path=None):
	global dict_url_idx

	if (seperated_input_path == None) or (output_path == None):
		return

	merged_sequences = []

	write_log('Merging seperated infos ...')
	for seperated_path in get_files_under_path(seperated_input_path):
		with open(seperated_path, 'r') as f_dict:
			seperated_dict = json.load(f_dict)

#		seperated_dict[user_id] = {
#			'start_time': start_time,
#			'end_time': end_time,
#			'sequence': idx_sequence,
#		}

		# dict_url_idx
		for user_id, dict_data in seperated_dict.items():
			sequence_entry = (dict_data['start_time'], dict_data['end_time'],
					dict_data['sequence'])
			merged_sequences.append(sequence_entry)

	write_log('Merging seperated infos ...  Done !')
	write_log('Sort by time : start')
	merged_sequences.sort(key=lambda x:x[0])
	write_log('Sort by time : end')

	timestamp = list(map(lambda x:tuple((x[0], x[1])), merged_sequences))
	seq_len = list(map(lambda x:len(x[2]), merged_sequences))
	sequence = list(map(lambda x:x[2], merged_sequences))

	merged_sequences = None
	dict_idx2url = {idx:word for word, idx in dict_url_idx.items()}

	write_log('Save rnn_inputs : start')
	dict_rnn_input = {
		'timestamp': timestamp,
		'seq_len': seq_len,
		'sequence': sequence,
		'idx2url': dict_idx2url,
	}

	with open(output_path, 'w') as f_input:
		json.dump(dict_rnn_input, f_input)
	write_log('Save rnn_inputs : end')

def main():
	global dict_w2v, dict_per_user, dict_url_idx, seperated_output_path

	options, args = parser.parse_args()
	if (options.w2v_path == None) or (options.data_path == None) or (options.output_file_path == None):
		return

	w2v_path = options.w2v_path
	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'

	output_path = options.output_file_path
	seperated_output_path = output_path + '/seperated'

	if not os.path.exists(output_path):
		os.system('mkdir -p ' + output_path)

	if not os.path.exists(seperated_output_path):
		os.system('mkdir -p ' + seperated_output_path)

	write_log('Preprocessing ...')
	with open(w2v_path, 'r') as f_w2v:
		dict_w2v = json.load(f_w2v)

	with open(per_user_path, 'r') as f_user:
		dict_per_user = json.load(f_user)

	user_ids = list(dict_per_user.keys())
	dict_url_idx = generate_unique_url_idxs()

	write_log('Preprocessing End : total {} user_ids'.format(len(user_ids)))

	n_div = 100
	multi_worker = MultiWorker(worker_count=10)
	works = list(map(lambda x: (x[0], x[1]), [(i, user_ids[i::n_div]) for i in range(n_div)]))

	multi_worker.work(works=works, work_function=preprocess_rnn_input)
	multi_worker = None

	# genrate_rnn_input
	generate_rnn_input(seperated_input_path=seperated_output_path,
			output_path=output_path + '/rnn_input.json')

if __name__ == '__main__':
	main()
