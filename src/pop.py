
from optparse import OptionParser

from ad_util import RNN_Input 

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)

rnn_input = None

def main():
	global rnn_input

	options, args = parser.parse_args()
	if options.input == None:
		return

	rnn_input_path = options.input + '/rnn_input.json'

	rnn_input = RNN_Input(rnn_input_path)

if __name__ == '__main__':
	main()
