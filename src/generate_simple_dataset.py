
import os
import pathlib

from optparse import OptionParser

from ad_util import get_files_under_path

parser = OptionParser()
parser.add_option('-i', '--input', dest='input', type='string', default=None)
parser.add_option('-o', '--output', dest='output', type='string', default=None)

one_week_path = None
simple_path = None

def generate_simple_dataset():
	global one_week_path, simple_path

	for data_path in get_files_under_path(one_week_path):
		simple_data = ''
		with open(data_path, 'r') as f_data:
			for i in range(1000):
				simple_data += f_data.readline().strip() + '\n'

		target_path = os.path.join(simple_path, os.path.basename(data_path))
		with open(target_path, 'w') as f_simple:
			f_simple.write(simple_data)


def main():
	global one_week_path, simple_path

	# Parse arguments
	options, args = parser.parse_args()
	if (options.input == None) or (options.output == None):
		return

	one_week_path = options.input
	simple_path = options.output

	pathlib.Path(simple_path).mkdir(parents=True, exist_ok=True)

	generate_simple_dataset()


if __name__ == '__main__':
	main()
