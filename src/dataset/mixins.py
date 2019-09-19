
import numpy as np

class RecInputMixin(object):
    def load_rec_input(self, dict_url2vec={}, dict_rec_input={}, options={}):
        assert(dict_rec_input and dict_url2vec)

        self._dict_url2vec = dict_url2vec
        self._dict_rec_input = dict_rec_input

        self._trendy_count = options.trendy_count
        self._recency_count = options.recency_count

    def idx2url(self, idx):
        return self._dict_rec_input['idx2url'][str(idx)]

    def idx2vec(self, idx):
        return self._dict_url2vec[self.idx2url(idx)]
#        if self._dict_yahoo_url2vec == None:
#            return self._dict_url2vec[self._dict_rnn_input['idx2url'][str(idx)]]
#        else:
#            return self._dict_yahoo_url2vec[self._dict_rnn_input['idx2url'][str(idx)]]

    def get_trendy(self, cur_time=-1, padding=0):
        trendy_list = self._dict_rec_input['trendy_idx'].get(str(cur_time), None)
        recency_list = self._dict_rec_input['recency_idx'].get(str(cur_time), None)

        x2_list = []

        if trendy_list == None:
            trendy_list = [[padding, 0]] * self._trendy_count

        if recency_list == None:
            recency_list = [[padding, 0]] * self._recency_count

        assert(len(trendy_list) >= self._trendy_count)
        assert(len(recency_list) >= self._recency_count)

        return trendy_list[:self._trendy_count] + recency_list[:self._recency_count]

    def get_candidates(self, cur_time=-1, padding=0):
        candidates_max = 100

        trendy_list = self._dict_rec_input['trendy_idx'].get(str(cur_time), None)

        if trendy_list == None:
            trendy_list = [[padding, 0]] * candidates_max

        if len(trendy_list) < candidates_max:
            trendy_list += [[padding, 0]] * (candidates_max - len(trendy_list))
        elif len(trendy_list) > candidates_max:
            trendy_list = trendy_list[:candidates_max]

        assert(len(trendy_list) == candidates_max)

        return trendy_list

    def get_candidates_(self, start_time=-1, end_time=-1, idx_count=0):
        if (start_time < 0) or (end_time < 0) or (idx_count <= 0):
            return []

        #    entry of : dict_rec_input['time_idx']
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
            cur_time = self._dict_rec_input['time_idx'][str(cur_time)]['next_time']
            for idx, count in self._dict_rec_input['time_idx'][str(cur_time)]['indices'].items():
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
                cur_time = self._dict_rec_input['time_idx'][str(time_from_end)]['next_time']
                time_from_end = cur_time
            else:
                if time_from_start == None:
                    steps += 1
                    continue
                cur_time = self._dict_rec_input['time_idx'][str(time_from_start)]['prev_time']
                time_from_start = cur_time

            if cur_time == None:
                continue

            for idx, count in self._dict_rec_input['time_idx'][str(cur_time)]['indices'].items():
                dict_merged[idx] = dict_merged.get(idx, 0) + count

        ret_sorted = sorted(dict_merged.items(), key=lambda x:x[1], reverse=True)
        if len(ret_sorted) > idx_count:
            ret_sorted = ret_sorted[:idx_count]
        return list(map(lambda x: int(x[0]), ret_sorted))

    def get_mrr_recency_candidates(self, cur_time=-1, padding=0):
        candidates_max = 100

        recency_candidates = []

        trendy_list = self._dict_rec_input['trendy_idx'].get(str(cur_time), None)
        recency_list = self._dict_rec_input['recency_idx'].get(str(cur_time), None)

        if trendy_list == None:
            trendy_list = []

        if recency_list == None:
            recency_list = []

        recency_articles = [r for r, r_c in recency_list]
        remains = []
        for t, t_c in trendy_list:
            if t in recency_articles:
                recency_candidates.append([t, t_c])
            else:
                remains.append(t)

        for r in remains:
            recency_candidates.append([r,0])

        if len(recency_candidates) < candidates_max:
            recency_candidates += [[padding, 0]] * (candidates_max - len(recency_candidates))
        elif len(recency_candidates) > candidates_max:
            recency_candidates = recency_candidates[:candidates_max]

        return recency_candidates


class WordEmbedMixin(object):
    def load_word_embed_input(self, dict_url2wi={}, dict_wi2vec={}, options={}):
        assert(dict_url2wi and dict_wi2vec)

        self._dict_url2wi = dict_url2wi
        self._dict_wi2vec = dict_wi2vec

        word_embed_size = len(next(iter(self._dict_wi2vec.values())))
        self._word_padding = [0.] * word_embed_size

    def url2wi_vecs(self, url):
        return np.stack([self._dict_wi2vec[wi] for wi in self._dict_url2wi[url]], axis=0)

    def url2wi_vecs_with_padding(self, url, fixed_size):
        def pad_words(words, count, padding):
            diff = count - len(words)

            if diff > 0:
                words += [padding] * diff

            return words[:count]

        if (url == 'url_pad'):
            return [self.get_word_vec_padding()] * fixed_size

        return pad_words(
                [self._dict_wi2vec[wi] for wi in self._dict_url2wi[url]],
                fixed_size,
                self.get_word_vec_padding())

    def get_word_vec_padding(self):
        return self._word_padding


