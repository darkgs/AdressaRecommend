
import os
import time
import subprocess

from multiprocessing.pool import ThreadPool
from threading import Semaphore

visible_gpus = [0, 1, 2, 3]
visible_gpus_sema = Semaphore(1)

total_works = 0
worker_counter = 0
worker_counter_sema = Semaphore(1)

def worker_function(args):
	global visible_gpus, visible_gpus_sema
	global total_works, worker_counter, worker_counter_sema

	# Get GPU resource
	my_gpu = None
	while(True):
		visible_gpus_sema.acquire()
		if len(visible_gpus) > 0:
			my_gpu = visible_gpus[0]
			visible_gpus = visible_gpus[1:]
			visible_gpus_sema.release()
			break
		visible_gpus_sema.release()
		time.sleep(10)

	assert my_gpu != None

	worker_counter_sema.acquire()
	worker_counter += 1
	my_count = worker_counter
	worker_counter_sema.release()

	model_file = args[0]
	params = args[1]

	print('Processing {}/{} on gpu {} - {}'.format(my_count, total_works, my_gpu, params))

	command = 'bash -c \"'
	command += 'source activate news;'
	command += 'export CUDA_VISIBLE_DEVICES={};'.format(my_gpu)
	command += 'python3 src/{} {};'.format(model_file, params)
	command += 'source deactivate'
	command += '\"'

	subprocess.check_output(command, shell=True)

#os.system(command)

	# Release GPU resource
	visible_gpus_sema.acquire()
	visible_gpus.append(my_gpu)
	visible_gpus_sema.release()

def parameter_search(target_name):
	global total_works

	dict_param_db = {
		'multicell': [
			'comp_multicell.py',
			'-i cache/one_week/torch_input -u cache/article_to_vec.json -w cache/one_week/multicell -z',
			{
				'd2v_embed': [1000],
				'learning_rate': [3e-3],
				'trendy_count': [5, 10],
				'recency_count': [3, 5],
				'hidden_size': [1024, 1208],
				'x2_dropout_rate': [0.3, 0.5, 0.7],
			},
		],
		'lstm': [
			'comp_lstm.py',
			'-i cache/one_week/torch_input -u cache/article_to_vec.json -w cache/one_week/lstm -z',
			{
				'd2v_embed': [1000],
				'learning_rate': [3e-3],
				'hidden_size': [1024, 1280, 1408],
				'num_layers': [1, 2],
			},
		],
		'gru4rec': [
			'comp_gru4rec.py',
			'-i cache/one_week/torch_input -u cache/article_to_vec.json -w cache/one_week/gru4rec -z',
			{
				'd2v_embed': [1000],
				'learning_rate': [3e-3],
				'hidden_size': [424, 512, 786],
				'num_layers': [1, 2, 3],
			},
		],
		'lstm_2input': [
			'comp_lstm.py',
			'-i cache/one_week/torch_input -u cache/article_to_vec.json -w cache/one_week/lstm_2input -z',
			{
				'd2v_embed': [1000],
				'learning_rate': [3e-3],
				'hidden_size': [786, 1024, 1280],
				'num_layers': [1, 2],
			},
		],
	}

	def generate_hyper_params(dict_params):
		if len(dict_params.keys()) <= 0:
			return []

		hyper_params = []

		key = next(iter(dict_params.keys()))
		params = dict_params.pop(key, [])
		child_options = generate_hyper_params(dict_params)

		for param in params:
			option = '--{} {}'.format(key, param)

			if len(child_options) <= 0:
				hyper_params.append(option)
			else:
				for child_option in child_options:
					hyper_params.append(child_option + ' ' + option)

		return hyper_params

	python_file, default_param, dict_params = dict_param_db[target_name]

	params = generate_hyper_params(dict_params)
	params = [default_param + ' ' + param for param in params]
	works = [(python_file, param) for param in params]
	total_works = len(works)

	thread_pool = ThreadPool(4)
	thread_pool.map(worker_function, works)


def show_result(target_name):
	result_dir_path = 'cache/one_week/{}/param_search'.format(target_name)

	results = []

	for (dir_path, dir_names, file_names) in os.walk(result_dir_path):
		for file_name in file_names:
			result_file_path = os.path.join(dir_path, file_name)

			with open(result_file_path, 'r') as f_ret:
				lines = f_ret.readlines()
				results.append((float(lines[0].strip()), file_name))

	results.sort(key=lambda x:x[0], reverse=True)

	for mrr, file_name in results:
		print(mrr, file_name)

def main():
	target_name = 'lstm'
	target_name = 'multicell'
	target_name = 'gru4rec'
	target_name = 'lstm_2input'

	parameter_search(target_name)
	show_result(target_name)

if __name__ == '__main__':
	main()
