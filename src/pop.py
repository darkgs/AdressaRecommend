
from optparse import OptionParser

from ad_util import RNN_Input 
from ad_util import write_log

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)

def main():

	options, args = parser.parse_args()
	if options.input == None:
		return

	rnn_input_path = options.input + '/rnn_input.json'

	rnn_input = RNN_Input(rnn_input_path)

	max_seq_len = rnn_input.max_seq_len()
	for epoch in range(10):
		test_x, test_y, test_seq_len, test_timestamps = rnn_input.generate_batchs(input_type='test', batch_size=100)

		predict_total = 0
		predict_mrr = 0.0
		for batch_idx in range(len(test_seq_len)):
			cand_start_time, cand_end_time = test_timestamps[batch_idx]

			cand_indices = rnn_input.get_candidates(start_time=cand_start_time,
					end_time=cand_end_time, idx_count=100)

			# POP@20
			cand_indices = cand_indices[:20]

			for i in range(test_seq_len[batch_idx]):
				predict_total += 1
			
				pred_idx = test_y[batch_idx].tolist()[0][i]

				hit_rank = 0
				if pred_idx in cand_indices:
					hit_rank = cand_indices.index(pred_idx)

				if hit_rank > 0:
					predict_mrr += 1.0/float(hit_rank)

		mrr_metric = predict_mrr / float(predict_total)
		print('epoch : {} - mrr:{}'.format(epoch, mrr_metric))


if __name__ == '__main__':
	main()
