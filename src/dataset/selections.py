
import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))

import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset  # For custom datasets

from dataset.mixins import RecInputMixin

from utils import *

class SelectDataset(Dataset):
    def __init__(self, dict_dataset):
        self._dict_dataset = dict_dataset
        self._data_len = len(self._dict_dataset)

    def __getitem__(self, index):
        return self._dict_dataset[index]

    def __len__(self):
        return self._data_len


class SelectRecInput(RecInputMixin):
    def __init__(self, path_rec_input, path_url2vec, options):
        #
        self._options = options

        # load rec_input
        dict_rec_input = ad_util.load_json(path_rec_input)

        # load url2vec
        dict_url2vec = ad_util.load_json(path_url2vec)

        # padding
        self._pad_idx = dict_rec_input['pad_idx']

        # rec_input mixin
        self.load_rec_input(dict_url2vec=dict_url2vec,
                dict_rec_input=dict_rec_input, options=options)

        # generate datasets for pytorch
        self._dataset = {
            'train': self.generate_dataset(dict_rec_input['dataset'], 'train'),
            'valid': self.generate_dataset(dict_rec_input['dataset'], 'valid'),
            'test': self.generate_dataset(dict_rec_input['dataset'], 'test'),
        }

        # save memory
        dict_rec_input['dataset'] = None

    def get_pad_idx(self):
        return self._pad_idx

    def generate_dataset(self, dict_raw_dataset, data_type):
        assert(data_type in ['train', 'valid', 'test'])

        num_prev_watch = self._options.num_prev_watch

        datas = []

        for timestamp_start, timestamp_end, sequence, time_sequence \
                in dict_raw_dataset[data_type]:

            if len(sequence) < (num_prev_watch + 1):
                continue

            if len(sequence) > (num_prev_watch + 10):
                sequence = sequence[-(num_prev_watch+10):]

            for i in range(len(sequence) - num_prev_watch):
                input_indices = sequence[i:i+num_prev_watch]
                target_idx = sequence[i+num_prev_watch]

                input_vector = [self.idx2vec(idx) for idx in input_indices]
                target_vector = self.idx2vec(target_idx)

                candidates = self.get_candidates(
                        time_sequence[i+num_prev_watch], self.get_pad_idx())

                candidate_indices = [idx for idx, count in candidates]
                candidate_vector = [self.idx2vec(idx) for idx, count in candidates]

                datas.append(
                    (input_indices, input_vector,
                    target_idx, target_vector,
                    candidate_indices, candidate_vector)
                )

        return SelectDataset(datas)

    def get_dataset(self, data_type):
        return self._dataset[data_type]


def selection_collate(batch):
    # [padded_indices, seq_len,
    #   padded_seq_vector,
    #   padded_trend_indices, padded_trend_vector,
    #   padded_candidate_indices, padded_candidate_vector]

    input_indices, input_vector, \
        target_idx, target_vector, \
        candidate_indices, candidate_vector = zip(*batch)

    return torch.FloatTensor(input_vector), \
            torch.FloatTensor(target_vector), \
            torch.FloatTensor(candidate_vector), \
            input_indices, target_idx, candidate_indices


class SelectRec(object):
    def __init__(self, path_rec_input, path_url2vec, cls_model, options):
        print('SelectRec Generating...')
        
        #
        self._options = options

        #
        self._rec_input = SelectRecInput(path_rec_input, path_url2vec, options)

        # 
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # pytorch dataloader
        self._dataloader = {
            'train': self.generate_dataloader('train'),
            'valid': self.generate_dataloader('valid'),
            'test': self.generate_dataloader('test'),
        }

        # mkdir workspaces if not exist
        ws_path = options.ws_path
        if not os.path.exists(ws_path):
            os.system('mkdir -p {}'.format(ws_path))

        model_ws_path = '{}/model/{}'.format(ws_path, ad_util.option2str(options))
        if not os.path.exists(model_ws_path):
            os.system('mkdir -p {}'.format(model_ws_path))

        self._ws_path = ws_path

        # Recommendation Model
        self._model = cls_model(options).to(self._device)
        self._model.apply(ad_util.weights_init)

        self._criterion = nn.BCELoss()
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=options.learning_rate)

    def generate_dataloader(self, data_type):
        assert(data_type in ['train', 'valid', 'test'])

        if data_type == 'train':
            batch_size, num_workers = 512, 16
        elif data_type == 'valid':
            batch_size, num_workers = 512, 16
        elif data_type == 'test':
            batch_size, num_workers = 64, 16
        else:
            assert('Should Never Reach Here' and False)

        return torch.utils.data.DataLoader(
            self._rec_input.get_dataset(data_type=data_type),
            batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=selection_collate)

    def to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self._device)
        else:
            return data

    def do_train(self, total_epoch=20, early_stop=10):
        print('[Train] start!!')

        total_epoch = 20

        # Early stop
        endure = 0
        best_valid_loss = sys.float_info.max

        for epoch in range(total_epoch):
            start_time = time.time()
            if endure > early_stop:
                print('Early stop!')
                break

            train_loss, _, _ = self.forward('train')
            valid_loss, _, _ = self.forward('valid')
            _, metric_hit, metric_mrr = self.forward('test')

            print('epoch {} - train loss({:.8f}) valid loss ({:.8f})'
                    .format(epoch, train_loss, valid_loss))
            print('hit_5({:.4f}) mrr_20({:.4f}) tooks {:.2f}'
                    .format(metric_hit, metric_mrr, time.time() - start_time))

            if best_valid_loss > valid_loss:
                best_valid_loss = valid_loss
                endure = 0
            else:
                endure += 1

    def forward(self, data_type):
        assert(data_type in ['train', 'valid', 'test'])

        if data_type == 'train':
            self._model.train()
        elif (data_type == 'valid') or (data_type == 'test'):
            self._model.eval()
        else:
            assert('Never Reach Here' and False)

        # loss
        total_loss = 0.0
        data_count = len(self._dataloader[data_type].dataset)

        # metric
        total_hit = 0.0
        total_mrr = 0.0

        # batch evaluation
        for batch_idx, batch_input in enumerate(self._dataloader[data_type]):
            input_vector, \
            target_vector, \
            candidate_vector, \
            input_indices, target_idx, candidate_indices = \
                [self.to_device(input_item) for input_item in batch_input]

            self._model.zero_grad()
            self._optimizer.zero_grad()

            input_vector = self.to_device(input_vector)

            outputs = self._model(input_vector)
            loss = self._criterion(F.softmax(outputs, dim=1), F.softmax(target_vector, dim=1))

            if data_type == 'train':
                loss.backward()
                self._optimizer.step()

            if data_type == 'test':
                hit, mrr = self.evaluation(5, 20, outputs, target_vector, candidate_vector)

                total_hit += hit
                total_mrr += mrr

            total_loss += loss.item()

        return total_loss / data_count, total_hit / data_count, total_mrr / data_count

    def evaluation(self, hit_count, mrr_count, predict, target_vector, candidates, candidate_count=20):
        # candidates
        candidates = candidates[:,:candidate_count,:]

        target_vector = torch.unsqueeze(target_vector, dim=1)
        candidates = torch.cat([candidates, target_vector], dim=1)

        # score
        scores = torch.squeeze(
                torch.bmm(candidates, torch.unsqueeze(predict, dim=2)),
                dim=2)
        _, indices = torch.sort(scores, dim=1)

        ranks = indices[:,candidate_count].to(dtype=torch.float32)

        # hit
        hit = torch.where(
                    ranks < hit_count,
                    torch.ones(*ranks.shape, dtype=torch.float32).to(self._device),
                    torch.zeros(*ranks.shape, dtype=torch.float32).to(self._device))

        # mrr
        mrr = torch.where(
                    ranks < mrr_count,
                    1.0 / (ranks + 1.0),
                    torch.zeros(*ranks.shape, dtype=torch.float32).to(self._device))

        return torch.sum(hit).item(), torch.sum(mrr).item()


