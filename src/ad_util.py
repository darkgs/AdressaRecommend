
import os
import datetime

def write_log(log):
	with open('log.txt', 'a') as log_f:
		time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		log_f.write(time_stamp + ' ' + log + '\n')


def get_files_under_path(p_path=None):
	ret = []

	if p_path == None:
		return ret

	for r, d, files in os.walk(p_path):
		for f in files:
			file_path = os.path.join(r,f)
			if not os.path.isfile(file_path):
				continue

			ret.append(file_path)

	return ret


class RNN_Input:
	def __init__(self, rnn_input_path):
		write_log('Initializing RNN_Input instance : start')
		with open(rnn_input_path, 'r') as f_input:
			self._dict_rnn_input = json.load(f_input)

		# Padding, Is there better way?
		self.padding()
		self._url_count = len(self._dict_rnn_input['idx2url'])

		write_log('Initializing RNN_Input instance : end')


	def padding(self):
		max_seq_len = max(self._dict_rnn_input['seq_len'])
		for seq_entry in self._dict_rnn_input['sequence']:
			pad_count = max_seq_len - len(seq_entry)
			if pad_count > 0:
				seq_entry += [0] * pad_count


	def idx2url(self, idx):
		return self._dict_rnn_input['idx2url'][str(idx)]


	def url_count(self):
		return self._url_count

	
	def get_candidates(self, start_time=-1, end_time=-1, idx_count=0):

		if (start_time < 0) or (end_time < 0) or (idx_count <= 0):
			return []

		#	entry of : dict_rnn_input['time_idx']
		#	(timestamp) :
		#	{
		#		prev_time: (timestamp)
		#		next_time: (timestamp)
		#		'indices': { idx:count, ... }
		#	}

		# swap if needed
		if start_time > end_time:
			tmp_time = start_time
			start_time = end_time
			end_time = tmp_time

		cur_time = start_time

		dict_merged = {}
		while(cur_time < end_time):
			cur_time = self._dict_rnn_input['time_idx'][str(cur_time)]['next_time']
			for idx, count in self._dict_rnn_input['time_idx'][str(cur_time)]['indices'].items():
				dict_merged[idx] = dict_merged.get(idx, 0) + count

		steps = 0
		time_from_start = start_time
		time_from_end = end_time
		while(len(dict_merged.keys()) < idx_count):
			if time_from_start == None and time_from_end == None:
				break

			if steps % 3 == 0:
				if time_from_end == None:
					steps += 1
					continue
				cur_time = self._dict_rnn_input['time_idx'][str(time_from_end)]['next_time']
				time_from_end = cur_time
			else:
				if time_from_start == None:
					steps += 1
					continue
				cur_time = self._dict_rnn_input['time_idx'][str(time_from_start)]['prev_time']
				time_from_start = cur_time

			if cur_time == None:
				continue

			for idx, count in self._dict_rnn_input['time_idx'][str(cur_time)]['indices'].items():
				dict_merged[idx] = dict_merged.get(idx, 0) + count

		ret_sorted = sorted(dict_merged.items(), key=lambda x:x[1], reverse=True)
		return list(map(lambda x: int(x[0]), ret_sorted))


	def generate_batchs(self, input_type='train', batch_size=10):

		total_len = len(self._dict_rnn_input['sequence'])
		if batch_size < 0:
			batch_size = total_len

		if input_type == 'train':
			idx_from = 0
			idx_to = int(total_len * 8 / 10)
		elif input_type == 'valid':
			idx_from = int(total_len * 8 / 10)
			idx_to = int(total_len * 9 / 10)
		else:
			idx_from = int(total_len * 9 / 10)
			idx_to = total_len

		batch_size = min(batch_size, idx_to - idx_from)

		data_idxs = list(range(idx_from, idx_to))
		np.random.shuffle(data_idxs)

#	dict_rnn_input['timestamp']
#	dict_rnn_input['seq_len']
#	dict_rnn_input['idx2url']
#	dict_rnn_input['sequence']
#	dict_rnn_input['time_idx']

		sequence = np.matrix(np.array(self._dict_rnn_input['sequence'])[data_idxs][:batch_size].tolist())
		seq_len = np.array(self._dict_rnn_input['seq_len'])[data_idxs][:batch_size]

		timestamps = np.array(self._dict_rnn_input['timestamp'])[data_idxs][:batch_size]

		input_x = sequence[:,:-1]
		input_y = sequence[:,1:]

		return input_x, input_y, seq_len-1, timestamps


	def __del__(self):
		self._dict_rnn_input = None
		write_log('Terminate RNN_Input instance : end')


