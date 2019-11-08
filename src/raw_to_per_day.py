
import os
import pathlib
import json

from optparse import OptionParser

from multiprocessing.pool import ThreadPool

from ad_util import write_log
from ad_util import get_files_under_path
from ad_util import find_best_url

parser = OptionParser()
parser.add_option('-m', '--mode', dest='mode', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)
parser.add_option('-d', '--dataset', dest='dataset', type='string', default=None)

data_mode = None
out_dir = None
data_path = None

dict_url2id = {}

def raw_to_per_day(raw_path):
    global out_dir, dict_url2id

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
        article_id = line_json.get('id', None)

        if (user_id == None) or (url == None) or (time < 0) or (article_id == None):
            continue

        if dict_per_user.get(user_id, None) == None:
            dict_per_user[user_id] = []

        dict_per_user[user_id].append(tuple((time, url)))
        list_per_time.append(tuple((time, user_id, url)))

        dict_url2id[url] = article_id

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

def raw_to_per_day_glob(raw_path):
    global out_dir, dict_url2id

    write_log('Processing : {}'.format(raw_path))

    with open(raw_path, 'r') as f_raw:
        lines = f_raw.readlines()

    dict_per_user = {}
    list_per_time = []

    total_count = len(lines)
    count = 0

    dict_header_idx = None
    for line in lines:
        if count % 10000 == 0:
            write_log('Processing({}) : {}/{}'.format(raw_path, count, total_count))
        count += 1

        line = line.strip()
        if dict_header_idx == None:
            dict_header_idx = {}
            for i, k in enumerate(line.split(',')):
                dict_header_idx[k] = i
            continue
             
        line_split = line.split(',')
        
        user_id = 'uid_{}'.format(line_split[dict_header_idx['user_id']])
        time = int(line_split[dict_header_idx['click_timestamp']]) // 1000
        url = 'url_{}'.format(line_split[dict_header_idx['click_article_id']])
        article_id = 'id_{}'.format(line_split[dict_header_idx['click_article_id']])

        if (user_id == None) or (url == None) or (time < 0) or (article_id == None):
            continue

        if dict_per_user.get(user_id, None) == None:
            dict_per_user[user_id] = []

        dict_per_user[user_id].append(tuple((time, url)))
        list_per_time.append(tuple((time, user_id, url)))

        dict_url2id[url] = article_id

    lines = None

    per_user_path = out_dir + '/per_user/' + os.path.splitext(os.path.basename(raw_path))[0]
    per_time_path = out_dir + '/per_time/' + os.path.splitext(os.path.basename(raw_path))[0]

    with open(per_user_path, 'w') as f_user:
        json.dump(dict_per_user, f_user)

    with open(per_time_path, 'w') as f_time:
        json.dump(list_per_time, f_time)

    dict_per_user = None
    list_per_time = None

    write_log('Done : {}'.format(raw_path))


def main():
    global data_mode, out_dir, data_path, dict_url2id
    options, args = parser.parse_args()

    if (options.mode == None) or (options.output == None) or (options.dataset == None):
        return

    data_mode = options.mode
    out_dir = options.output
    dataset = options.dataset

    if dataset not in ['adressa', 'glob']:
        print('Wrong dataset name : {}'.format(dataset))
        return

    if dataset == 'adressa':
        data_path = 'data/' + data_mode
        worker_fn = raw_to_per_day
    elif dataset == 'glob':
        data_path = 'data/glob'
        if data_mode == 'simple':
            data_path += '/simple'
        else:
            data_path += '/clicks'
        worker_fn = raw_to_per_day_glob

    os.system('mkdir -p {}'.format(out_dir + '/per_user'))
    os.system('mkdir -p {}'.format(out_dir + '/per_time'))

    works = get_files_under_path(data_path)

    dict_url2id = {}
    with ThreadPool(8) as pool:
        pool.map(worker_fn, works)

    with open(out_dir + '/url2id.json', 'w') as f_dict:
        json.dump(dict_url2id, f_dict)


if __name__ == '__main__':
    main()

