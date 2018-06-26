
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)

def main():
	options, args = parser.parse_args()
	print(options.input)
	print(options.output)
	print('hello')

if __name__ == '__main__':
	main()
