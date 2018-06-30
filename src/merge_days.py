
import pathlib
import json

from optparse import OptionParser

from ad_util import write_log
from ad_util import get_files_under_path

parser = OptionParser()
parser.add_option('-m', '--mode', dest='mode', type='string', default=None)
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-w', '--w2v_path', dest='w2v_path', type='string', default=None)

data_mode = None
out_dir = None
per_day_path = None
w2v_path = None
valid_urls = []

def merge_per_user():
	global data_mode, out_dir, per_day_path, valid_urls

	write_log('Merging per_user Start')
	user_files = get_files_under_path(per_day_path + '/per_user')

	dict_merged = {}

	total_count = len(user_files)
	count = 0
	for user_path in user_files:
		write_log('Merging per_user : {}/{}'.format(count, total_count))
		count += 1
		with open(user_path, 'r') as f_data:
			dict_per_user = json.load(f_data)
		write_log('Merging per_user Loaded: {}/{}'.format(count, total_count))
		
		for key in dict_per_user.keys():
			dict_merged[key] = dict_merged.get(key, []) + dict_per_user[key]

		write_log('Merging per_user Merged: {}/{}'.format(count, total_count))
		dict_per_user = None

	write_log('Merging per_user : sorting start')
	for user_id in dict_merged:
		# (timestamp, url)
		dict_merged[user_id] = list(filter(lambda x: x[1] in valid_urls, dict_merged[user_id]))
		dict_merged[user_id].sort(key=lambda x:x[0])
	write_log('Merging per_user : sorting end')

	write_log('Merging per_user start to writing')
	with open(out_dir + '/per_user.json', 'w') as f_user:
		json.dump(dict_merged, f_user)
	write_log('Merging per_user End')

	dict_merged = None

def merge_per_time():
	global data_mode, out_dir, per_day_path, valid_urls
	write_log('Merging per_time Start')

	time_files = get_files_under_path(per_day_path + '/per_time')

	list_merged = []

	write_log('Merging per_time : Load Start')
	for time_path in time_files:
		with open(time_path, 'r') as f_data:
			list_per_time = json.load(f_data)

		list_merged += list_per_time
		list_per_time = None
	write_log('Merging per_time : Load End')

	write_log('Merging per_time : Sort Start')
	# (timestamp, user_id, url)
	list_merged = list(filter(lambda x:x[2] in valid_urls, list_merged))
	list_merged.sort(key=lambda x:x[0])
	write_log('Merging per_time : Sort End')

	with open(out_dir + '/per_time.json', 'w') as f_time:
		json.dump(list_merged, f_time)

	list_merged = None

	write_log('Merging per_time End')

def main():
	global data_mode, out_dir, per_day_path, w2v_path, valid_urls

	options, args = parser.parse_args()

	if (options.mode == None) or (options.output == None) or (options.input == None) or (options.w2v_path == None):
		return

	data_mode = options.mode
	per_day_path = options.input
	out_dir = options.output
	w2v_path = options.w2v_path

	pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

	write_log('w2v Load : start')
	with open(w2v_path, 'r') as f_w2v:
		dict_w2v = json.load(f_w2v)
	write_log('w2v Load : end')

	valid_urls = dict_w2v.keys()
	dict_w2v = None

	merge_per_user()
	merge_per_time()


if __name__ == '__main__':
	main()
