import sys

def make_dictionary(data,key):
	unique = set([d[key] for d in data])
	return {name : index for index, name  in enumerate(unique)}

def process_data_for_sentence(data):
	sids = set([d['sid'] for d in data])
	assert(len(sids) == 1)

	pred_to_index = make_dictionary(data,'pred')
	arg_to_index  = make_dictionary(data,'arg')

	def process_line(data):
		data['pred'] = pred_to_index[data['pred']]
		data['arg']  = arg_to_index[data['arg']]

		return data

	return [process_line(d) for d in data]

def main():
	data_file = sys.argv[1]
	indexing_file = sys.argv[2]
	with open(indexing_file, 'r') as f:
		lines = [l.rstrip() for l in f]
		pairs = [line.split(' ') for line in lines]
		label_str_to_int = {l[0]: l[1] for l in pairs}


	with open(data_file, 'r') as f:
		cur_data = []
		prev_sent_id = ""
		for l in f.readlines():
			fields   = l.rstrip().split(' ')
			data = {}
			sent_id  = fields[0]
			data['sid'] = sent_id
			data['pred'] = fields[1] + "-" + fields[2]
			data['arg']  = fields[3] + "-" + fields[4]
			data['role'] = label_str_to_int[fields[6]]


			if(prev_sent_id == sent_id):
				cur_data.append(data)
			else:
				if(len(cur_data) > 0):
					for d in process_data_for_sentence(cur_data):
						print("{} {} {} {}".format(d['sid'],d['pred'],d['arg'],d['role']))
				cur_data = [data]
				prev_sent_id = sent_id
        #this is for the final sentence
        for d in process_data_for_sentence(cur_data):
            print("{} {} {} {}".format(d['sid'],d['pred'],d['arg'],d['role']))

if __name__ == '__main__':
	main()
