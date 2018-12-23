
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-e', '--d2v_embed', dest='d2v_embed', type='string', default='1000')
parser.add_option('-u', '--u2v_path', dest='u2v_path', type='string', default=None)
parser.add_option('-a', '--u2i_path', dest='u2i_path', type='string', default=None)

#python3 src/stat_adressa_dataset.py -e $(D2V_EMBED) -u cache/article_to_vec.json -a $(BASE_PATH)/article_info.json 

def main():
	print('hello')

if __name__ == '__main__':
	main()

