
from optparse import OptionParser

from ad_util import load_json

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)

def main():
	options, args = parser.parse_args()

	if options.input == None:
		return

	torch_input_path = options.input
	dict_rnn_input_path = '{}/torch_rnn_input.dict'.format(torch_input_path)

	print('Loading torch input : start')
	dict_rnn_input = load_json(dict_rnn_input_path)
	print('Loading torch input : end')

	sequence_count = 0
	event_count = 0
	for dataset_name in ['train', 'valid', 'test']:
		sequence_count += len(dict_rnn_input['dataset'][dataset_name]) 
		for sequence in dict_rnn_input['dataset'][dataset_name]:
			event_count += len(sequence[2])

	article_count = len(dict_rnn_input['idx2url'])

	print('number of session : {}'.format(sequence_count))
	print('number of event : {}'.format(event_count))
	print('number of article : {}'.format(article_count))

if __name__ == '__main__':
	main()
