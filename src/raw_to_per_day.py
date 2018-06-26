
import os
import pathlib
import json

from optparse import OptionParser

from multi_worker import MultiWorker
from ad_util import write_log
from ad_util import get_files_under_path

parser = OptionParser()
parser.add_option('-m', '--mode', dest='mode', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)

data_mode = None
out_dir = None
data_path = None

def find_best_url(event_dict=None):
	if event_dict == None:
		return None

	url_keys = ['url', 'cannonicalUrl', 'referrerUrl']
	black_list = ['http://google.no', 'http://facebook.com', 'http://adressa.no/search']

	best_url = None
	for key in url_keys:
		url = event_dict.get(key, None)
		if url == None:
			continue

		if url.count('/') < 3:
			continue

		black_url = False
		for black in black_list:
			if url.startswith(black):
				black_url = True
				break
		if black_url:
			continue

		if (best_url == None) or (len(best_url) < len(url)):
			best_url = url

	return best_url

def raw_to_per_day(raw_path):
	global out_dir

	write_log('Processing : {}'.format(raw_path))

	with open(raw_path, 'r') as f_raw:
		lines = f_raw.readlines()

	dict_per_user = {}
	list_per_time = []

	total_count = len(lines)
	count = 0

	for line in lines:
		if count % 10000 == 0:
			write_log('Processing({}) : {}/{}'.format(raw_path, count, total_count))
		count += 1

		line = line.strip()
		line_json = json.loads(line)
	
		user_id = line_json.get('userId', None)
		url = find_best_url(event_dict=line_json)
		time = line_json.get('time', -1)

		if (user_id == None) or (url == None) or (time < 0):
			continue

		if dict_per_user.get(user_id, None) == None:
			dict_per_user[user_id] = []

		dict_per_user[user_id].append(tuple((time, url)))
		list_per_time.append(tuple((time, user_id, url)))

	lines = None

	per_user_path = out_dir + '/per_user/' + os.path.basename(raw_path)
	per_time_path = out_dir + '/per_time/' + os.path.basename(raw_path)

	with open(per_user_path, 'w') as f_user:
		json.dump(dict_per_user, f_user)

	with open(per_time_path, 'w') as f_time:
		json.dump(list_per_time, f_time)

	dict_per_user = None
	list_per_time = None

	write_log('Done : {}'.format(raw_path))

def main():
	global data_mode, out_dir, data_path
	options, args = parser.parse_args()

	if (options.mode == None) or (options.output == None):
		return

	data_mode = options.mode
	out_dir = options.output

	pathlib.Path(out_dir + '/per_user').mkdir(parents=True, exist_ok=True)
	pathlib.Path(out_dir + '/per_time').mkdir(parents=True, exist_ok=True)

	data_path = 'data/' + data_mode

	multi_worker = MultiWorker(worker_count=5)
	works = list(map(lambda x:tuple([x]), get_files_under_path(data_path)))
	multi_worker.work(works=works, work_function=raw_to_per_day)

	multi_worker = None

if __name__ == '__main__':
	main()
