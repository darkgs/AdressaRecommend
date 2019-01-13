
import os
import time
import subprocess

from multiprocessing.pool import ThreadPool
from threading import Semaphore

visible_gpus = [0]
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
#	command += 'source activate news;'
	command += 'export CUDA_VISIBLE_DEVICES={};'.format(my_gpu)
	command += 'python3 src/{} {};'.format(model_file, params)
#	command += 'source deactivate'
	command += '\"'

#subprocess.check_output(command, shell=True)

	os.system(command)

	# Release GPU resource
	visible_gpus_sema.acquire()
	visible_gpus.append(my_gpu)
	visible_gpus_sema.release()

def parameter_search(dataset, target_name):
	global total_works

	if dataset == 'adressa':
		dict_param_db = {
			'lstm': [
				'comp_lstm.py',
				'-i cache/adressa/one_week/torch_input -u cache/adressa/article_to_vec.json -w cache/adressa/one_week/lstm -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'hidden_size': [896, 1024, 1280],
					'num_layers': [1, 2],
				},
			],
			'gru4rec': [
				'comp_gru4rec.py',
				'-i cache/adressa/one_week/torch_input -u cache/adressa/article_to_vec.json -w cache/adressa/one_week/gru4rec -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'dropout_rate': [0.3, 0.5],
					'hidden_size': [512, 786],
					'num_layers': [2, 3],
				},
			],
			'lstm_2input': [
				'comp_lstm_2input.py',
				'-i cache/adressa/one_week/torch_input -u cache/adressa/article_to_vec.json -w cache/adressa/one_week/lstm_2input -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'hidden_size': [682, 786, 896],
				},
			],
			'multicell': [
				'comp_multicell.py',
				'-i cache/adressa/one_week/torch_input -u cache/adressa/article_to_vec.json -w cache/adressa/one_week/multicell -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'trendy_count': [5],
					'recency_count': [3],
					'hidden_size': [1024, 1208, 1440],
					'x2_dropout_rate': [0.3, 0.5],
				},
			],
			'yahoo': [
				'comp_yahoo.py',
				'-i cache/adressa/one_week/torch_input -u cache/adressa/article_to_vec.json -w cache/adressa/one_week/yahoo -y cache/adressa/one_week/yahoo_article2vec.json -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'dropout_rate': [0.3, 0.5],
					'hidden_size': [786, 1024, 1208],
					'num_layers': [2, 3],
				},
			],
			'naver': [
				'comp_naver.py',
				'-i cache/adressa/one_week/torch_input -u cache/adressa/article_to_vec.json -w cache/adressa/one_week/naver -c cache/adressa/one_week/article_info.json -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'hidden_size': [896, 1024, 1280],
					'decay_rate': [0.1, 0.2, 0.5],
				},
			],
		}
	elif dataset == 'glob':
		dict_param_db = {
			'lstm': [
				'comp_lstm.py',
				'-i cache/glob/one_week/torch_input -u cache/glob/article_to_vec.json -w cache/glob/one_week/lstm -z',
				{
					'd2v_embed': [250],
					'learning_rate': [3e-3],
					'hidden_size': [512, 896, 1024],
					'num_layers': [1, 2],
				},
			],
			'gru4rec': [
				'comp_gru4rec.py',
				'-i cache/glob/one_week/torch_input -u cache/glob/article_to_vec.json -w cache/glob/one_week/gru4rec -z',
				{
					'd2v_embed': [250],
					'learning_rate': [3e-3],
					'dropout_rate': [0.3, 0.5],
					'hidden_size': [386, 512, 786],
					'num_layers': [2, 3],
				},
			],
			'lstm_2input': [
				'comp_lstm_2input.py',
				'-i cache/glob/one_week/torch_input -u cache/glob/article_to_vec.json -w cache/glob/one_week/lstm_2input -z',
				{
					'd2v_embed': [250],
					'learning_rate': [3e-3],
					'hidden_size': [486, 682, 786],
				},
			],
			'multicell': [
				'comp_multicell.py',
				'-i cache/glob/one_week/torch_input -u cache/glob/article_to_vec.json -w cache/glob/one_week/multicell -z',
				{
					'd2v_embed': [250],
					'learning_rate': [3e-3],
					'trendy_count': [5, 7],
					'recency_count': [3, 5],
					'hidden_size': [1280, 1440],
					'x2_dropout_rate': [0.3, 0.5],
				},
			],
			'yahoo': [
				'comp_yahoo.py',
				'-i cache/glob/one_week/torch_input -u cache/glob/article_to_vec.json -w cache/glob/one_week/yahoo -y cache/glob/one_week/yahoo_article2vec.json -z',
				{
					'd2v_embed': [250],
					'learning_rate': [3e-3],
					'dropout_rate': [0.3, 0.5],
					'hidden_size': [386, 462, 512, 786, 1024],
					'num_layers': [2, 3],
				},
			],
			'naver': [
				'comp_naver.py',
				'-i cache/glob/one_week/torch_input -u cache/glob/article_to_vec.json -w cache/glob/one_week/naver -c cache/glob/one_week/article_info.json -z',
				{
					'd2v_embed': [1000],
					'learning_rate': [3e-3],
					'hidden_size': [896, 1024, 1280],
					'decay_rate': [0.1, 0.2, 0.5],
				},
			],
		}
	else:
		print('Wrong dataset name : {}'.foramt(dataset))
		return

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


def show_result(dataset, target_name):
	result_dir_path = 'cache/{}/one_week/{}/param_search'.format(dataset, target_name)

	results = []

	for (dir_path, dir_names, file_names) in os.walk(result_dir_path):
		for file_name in file_names:
			result_file_path = os.path.join(dir_path, file_name)

			with open(result_file_path, 'r') as f_ret:
				lines = f_ret.readlines()
				results.append(list(map(lambda x: float(x.strip()), lines)) + [file_name])

	results.sort(key=lambda x:x[4], reverse=True)

	for hit_5, auc_10, auc_20, mrr_5, mrr_20, file_name in results:
		print('hit_5({:.4f}) auc({:.4f}) mrr_20({:.4f}) : {}'.format(hit_5, auc_20, mrr_20, file_name))

def main():
	dataset = 'adressa'
	dataset = 'glob'

	target_name = 'lstm'
	target_name = 'gru4rec'
	target_name = 'lstm_2input'
	target_name = 'multicell'
	target_name = 'yahoo'
	target_name = 'naver'

	dataset = 'adressa'
	target_name = 'naver'

	parameter_search(dataset, target_name)
	show_result(dataset, target_name)

if __name__ == '__main__':
	main()
