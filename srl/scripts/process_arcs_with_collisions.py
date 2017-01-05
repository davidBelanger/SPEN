import sys

def make_dictionary(data,key):
	unique = set([d[key] for d in data])
	return {name : index for index, name  in enumerate(unique)}, unique

def process_data_for_sentence(data):
	sids = set([d['sid'] for d in data])
	assert(len(sids) == 1)

	pred_to_index, pred_ranges = make_dictionary(data,'pred')
	arg_to_index, arc_ranges  = make_dictionary(data,'arg')

	def process_line(data):
		data['pred'] = pred_to_index[data['pred']]
		data['arg']  = arg_to_index[data['arg']]

		return data

	arcs = [process_line(d) for d in data]

	def overlapping(range1,range2):
		start1, end1 = range1
		start2, end2 = range2
		return (start1 >= start2 and start1 <= end2) or (start2 >= start1 and start2 <= end1) or (end1 <= end2 and end1 >= start2) or (end2 <= end1 and end2 >= start1)

	def get_collisions(dict1,dict2):
		items1 = dict1.keys()
		items2 = dict2.keys()
		collisions = []
		for i in range(len(items1)):
			for j in range(len(items2)):
				if(i != j):
					k1 = items1[i]
					k2 = items2[j]
					if(overlapping(k1,k2)):
						collisions.append((dict1[k1], dict2[k2]))
		return collisions

	pred_to_arg_collisions  = get_collisions(pred_to_index, arg_to_index)
	pred_to_pred_collisions = get_collisions(pred_to_index, pred_to_index)
	arg_to_arg_collisions   = get_collisions(arg_to_index, arg_to_index)

	return arcs, pred_to_arg_collisions, pred_to_pred_collisions, arg_to_arg_collisions

def write_data_for_sentence(cur_data, arc_writer, collision_writers):
	arcs, pred_to_arg_collisions, pred_to_pred_collisions, arg_to_arg_collisions = process_data_for_sentence(cur_data)
	
	for d in arcs:
		arc_writer.write("{} {} {} {}\n".format(d['sid'],d['pred'],d['arg'],d['role']))

	def write_collisions(writer, collisions):
		for c in collisions:
			writer.write("{},{} ".format(c[0],c[1]))
		writer.write('\n')

	write_collisions(collision_writers['p2a'], pred_to_arg_collisions)
	write_collisions(collision_writers['p2p'], pred_to_pred_collisions)
	write_collisions(collision_writers['a2a'], arg_to_arg_collisions)


def main():
	data_file = sys.argv[1]
	indexing_file = sys.argv[2]
	out_file = sys.argv[3]

	with open(indexing_file, 'r') as f:
		lines = [l.rstrip() for l in f]
		pairs = [line.split(' ') for line in lines]
		label_str_to_int = {l[0]: l[1] for l in pairs}


	with open(out_file + ".arcs",'w') as arcs_out:
		collision_writers = { ext: open(out_file+"."+ext, 'w') for ext in ['p2a','p2p','a2a'] }

		with open(data_file, 'r') as f:
			cur_data = []
			prev_sent_id = ""
			for l in f.readlines():
				fields   = l.rstrip().split(' ')
				data = {}
				sent_id  = fields[0]
				data['sid'] = sent_id
				data['pred'] = (int(fields[1]),int(fields[2]))
				data['arg']  = (int(fields[3]),int(fields[4]))
				data['role'] = label_str_to_int[fields[6]]


				if(prev_sent_id == sent_id):
					cur_data.append(data)
				else:
					if(len(cur_data) > 0):
						write_data_for_sentence(cur_data, arcs_out, collision_writers)
					cur_data = [data]
					prev_sent_id = sent_id
	        #this is for the final sentence
	        write_data_for_sentence(cur_data, arcs_out, collision_writers)
	        

if __name__ == '__main__':
	main()
