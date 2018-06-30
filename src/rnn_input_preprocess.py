
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
seperated_output_path = None

def preprocess_rnn_input(args=(-1, [])):
	global dict_w2v, dict_per_user, seperated_output_path

	worker_id, user_ids = args

	write_log('worker({}) : start'.format(worker_id))
	dict_data = {}
	for user_id in user_ids:
		seq_len = len(dict_per_user[user_id])

		if seq_len < 2:
			continue

		vec_sequence = list(map(lambda x:dict_w2v[x[1]], dict_per_user[user_id]))
		dict_data[user_id] = vec_sequence

	with open(seperated_output_path + '/' + str(worker_id) + '_data.json', 'w') as f_out:
		json.dump(dict_data, f_out)
	write_log('worker({}) : end'.format(worker_id))

def generate_rnn_input(seperated_input_path=None, output_path=None, max_seq_len=10):

	if (seperated_input_path == None) or (output_path == None):
		return

	rnn_input = []
	seq_lengths = []

	write_log('Merging seperated infos ...')
	for seperated_path in get_files_under_path(seperated_input_path):
		with open(seperated_path, 'r') as f_dict:
			seperated_dict = json.load(f_dict)

		for user_id, sequence in seperated_dict.items():
			len_seq = len(sequence)
			if len_seq > max_seq_len:
				sequence = sequence[len_seq-max_seq_len:]
			elif len_seq < max_seq_len:
				sequence += [[0.0]*100] * (max_seq_len - len_seq)

			rnn_input.append(sequence)
			seq_lengths.append(len_seq)
	write_log('Merging seperated infos ...  Done !')

	write_log('Save rnn_inputs : start')
	dict_rnn_input = {
		'seq_lengths': seq_lengths,
		'rnn_input': rnn_input,
	}

	with open(output_path, 'w') as f_input:
		json.dump(dict_rnn_input, f_input)
	write_log('Save rnn_inputs : end')

def main():
	global dict_w2v, dict_per_user, seperated_output_path

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

	write_log('Preprocessing End : total {} user_ids'.format(len(user_ids)))

	n_div = 100
	multi_worker = MultiWorker(worker_count=10)
	works = list(map(lambda x: (x[0], x[1]), [(i, user_ids[i::n_div]) for i in range(n_div)]))

	multi_worker.work(works=works, work_function=preprocess_rnn_input)
	multi_worker = None

	# genrate_rnn_input
	generate_rnn_input(seperated_input_path=seperated_output_path,
			output_path=output_path + '/rnn_input.json', max_seq_len=20)

if __name__ == '__main__':
	main()
