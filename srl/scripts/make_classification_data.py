import sys

def main():
	data_file = sys.argv[1]
	indexing_file = sys.argv[2]
	with open(indexing_file, 'r') as f:
		lines = [l.rstrip() for l in f]
		pairs = [line.split(' ') for line in lines]
		label_str_to_int = {l[0]: l[1] for l in pairs}

	with open(data_file, 'r') as f:
		for l in f.readlines():
			fields   = l.rstrip().split(' ')
			print(label_str_to_int[fields[6]])


if __name__ == '__main__':
	main()