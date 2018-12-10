
import os
import time

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

	print('Processing {}/{} on gpu {}'.format(my_count, total_works, my_gpu))

	model_file = args[0]
	params = args[1]

	command = ''
#	command += 'source activate news;'
	command += 'export CUDA_VISIBLE_DEVICES={};'.format(my_gpu)
	command += 'python3 src/{} {};'.format(model_file, params)
#	command += 'source deactivate'

	os.system(command)

	# Release GPU resource
	visible_gpus_sema.acquire()
	visible_gpus.append(my_gpu)
	visible_gpus_sema.release()

def main():
	global total_works

	dict_params = {
		'd2v_embed': [200, 500, 1000],
		'learning_rate': [1e-3, 3e-3, 5e-3],
		'trendy_count': [5, 7, 10],
		'recency_count': [3, 5, 7],
		'hidden_size': [721, 1024, 1208],
		'x2_dropout_rate': [0.3, 0.5, 0.7],
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

	params = generate_hyper_params(dict_params)
	default_param = '-i cache/one_week/torch_input -u cache/article_to_vec.json -w cache/one_week/multicell -z'
	params = [default_param + ' ' + param for param in params]
	works = [('comp_multicell.py', param) for param in params]
	total_works = len(works)

	thread_pool = ThreadPool(4)
	thread_pool.map(worker_function, works)


if __name__ == '__main__':
	main()
