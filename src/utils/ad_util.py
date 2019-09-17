
import os
import json
import datetime

import numpy as np

import torch
import torch.nn as nn

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


def load_json(dict_path=None):
    print('Load_Json Start : {}'.format(dict_path))
    dict_ret = {}

    if dict_path == None:
        return

    with open(dict_path, 'r') as f_dict:
        dict_ret = json.load(f_dict)

    print('Load_Json End : {}'.format(dict_path))
    return dict_ret

def option2str(options):
    items = [(key, option) for key, option in options.__dict__.items() if '/' not in str(option) ]
    items.sort(key=lambda x: x[0])
    items = [key + '-' + str(option) for key, option in items]
    return '__'.join(items)


