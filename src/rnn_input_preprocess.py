
import os
import json

from optparse import OptionParser

from multi_worker import MultiWorker
from ad_util import write_log

parser = OptionParser()
parser.add_option('-d', '--data_path', dest='data_path', type='string', default=None)
parser.add_option('-w', '--w2v_path', dest='w2v_path', type='string', default=None)
parser.add_option('-o', '--output_path', dest='output_path', type='string', default=None)

dict_w2v = None
dict_per_user = None
output_path = None

def preprocess_rnn_input(args=(-1, [])):
	global dict_w2v, dict_per_user, output_path

	worker_id, user_ids = args

	write_log('worker({}) : start'.format(worker_id))
	dict_data = {}
	for user_id in user_ids:
		seq_len = len(dict_per_user[user_id])

		if seq_len < 2:
			continue

		vec_sequence = list(map(lambda x:dict_w2v[x[1]], dict_per_user[user_id]))
		dict_data[user_id] = vec_sequence

	with open(output_path + '/' + str(worker_id) + '_data.json', 'w') as f_out:
		json.dump(dict_data, f_out)
	write_log('worker({}) : end'.format(worker_id))

def main():
	global dict_w2v, dict_per_user, output_path

	options, args = parser.parse_args()
	if (options.w2v_path == None) or (options.data_path == None) or (options.output_path == None):
		return

	w2v_path = options.w2v_path
	per_time_path = options.data_path + '/per_time.json'
	per_user_path = options.data_path + '/per_user.json'
	output_path = options.output_path

	if not os.path.exists(output_path):
		os.system('mkdir -p ' + output_path)

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

if __name__ == '__main__':
	main()
