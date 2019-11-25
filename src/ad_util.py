
import os
import json
import datetime

import numpy as np

import torch
import torch.nn as nn

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

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)


class RNN_Input:
    def __init__(self, rnn_input_path):
        write_log('Initializing RNN_Input instance : start')
        with open(rnn_input_path, 'r') as f_input:
            self._dict_rnn_input = json.load(f_input)

        self._url_count = len(self._dict_rnn_input['idx2url'])
        self._max_seq_len = max(self._dict_rnn_input['seq_len'])

        # Padding, Any better ?
        self.padding()

        write_log('Initializing RNN_Input instance : end')


    def __del__(self):
        self._dict_rnn_input = None
        write_log('Terminate RNN_Input instance : end')


    def padding(self):
        for seq_entry in self._dict_rnn_input['sequence']:
            pad_count = self._max_seq_len - len(seq_entry)
            if pad_count > 0:
                seq_entry += [0] * pad_count


    def idx2url(self, idx):
        return self._dict_rnn_input['idx2url'][str(idx)]


    def max_seq_len(self):
        return self._max_seq_len


    def url_count(self):
        return self._url_count

    
    def get_candidates(self, start_time=-1, end_time=-1, idx_count=0):

        if (start_time < 0) or (end_time < 0) or (idx_count <= 0):
            return []

        #    entry of : dict_rnn_input['time_idx']
        #    (timestamp) :
        #    {
        #        prev_time: (timestamp)
        #        next_time: (timestamp)
        #        'indices': { idx:count, ... }
        #    }

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

#    dict_rnn_input['timestamp']
#    dict_rnn_input['seq_len']
#    dict_rnn_input['idx2url']
#    dict_rnn_input['sequence']
#    dict_rnn_input['time_idx']

        sequence = np.matrix(np.array(self._dict_rnn_input['sequence'])[data_idxs][:batch_size].tolist())
        seq_len = np.array(self._dict_rnn_input['seq_len'])[data_idxs][:batch_size]

        timestamps = np.array(self._dict_rnn_input['timestamp'])[data_idxs][:batch_size]

        input_x = sequence[:,:-1]
        input_y = sequence[:,1:]

        return input_x, input_y, seq_len-1, timestamps


def load_json(dict_path=None):
    dict_ret = {}

    if dict_path == None:
        return

    with open(dict_path, 'r') as f_dict:
        dict_ret = json.load(f_dict)

    return dict_ret

def option2str(options):
    items = [(key, option) for key, option in options.__dict__.items() if '/' not in str(option) ]
    items.sort(key=lambda x: x[0])
    items = [key + '-' + str(option) for key, option in items]
    return '__'.join(items)


