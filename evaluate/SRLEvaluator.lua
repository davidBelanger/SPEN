local SRLEvaluator, parent = torch.class('SRLEvaluator')

--todo: this should take a factory for an iterator

function SRLEvaluator:__init(batcher, predictor, null_arc_index, prediction_writing_info)
	self.batcher = batcher
	self.predictor = predictor
	self.null_arc_index = null_arc_index
	self.prediction_writing_info = prediction_writing_info
	if(prediction_writing_info) then
		self.labels_int2string = {}
		print('reading from labels map from '..prediction_writing_info.label_map_file)
		for s in io.lines(prediction_writing_info.label_map_file) do
			local s1, s2 = unpack(Util:splitByDelim(s," "))
			local int_key = tonumber(s2) + 1 --add one because the preprocessing was 1-indexed but the label_map_file wasn't changed
			self.labels_int2string[int_key] = s1
		end
	end
end

function SRLEvaluator:evaluate(name)
	print('Beginning Evaluation')
	self.predictor:evaluate()

	local pits = {}
	local gits = {}
	
	local example_count = 0
	local batch_iterator = self.batcher:get_iterator()
	local batch_count = 0
	local report_about_rejected_examples
	local write_predictions = self.prediction_writing_info

	local prediction_writer, results_file
	if(write_predictions) then
		results_file = self.prediction_writing_info.out_file_base.."."..name
		print('writing evaluation results to: '..results_file)
		prediction_writer = io.open(results_file, "w")
	end

	local num_non_null_predictions = 0
	local num_correct_non_null_predictions = 0
	while(true) do
		local batch_data = batch_iterator()

		local batch_labels, batch_inputs, num_actual_data = unpack(batch_data)
		if(batch_inputs == nil) then 
			report_about_rejected_examples = batch_labels
			break 
		end


		batch_count = batch_count + 1

		local preds = self.predictor:predict(batch_inputs)

		--now explicitly mask the preds so that we don't predict anything non-null in places that were filtered
		local pit = preds:narrow(1,1,num_actual_data)
		local mask = batch_inputs[2]:float():narrow(1,1,num_actual_data)--:clone():float():mul(-1):add(1)
		local neg_mask = mask:clone():mul(-1):add(1)
		pit = pit:float():cmul(mask) + neg_mask

		local git = batch_labels:narrow(1,1,num_actual_data)
		table.insert(pits,pit:clone())
		table.insert(gits,git:clone())	
		if(write_predictions) then
			for t = 1,num_actual_data do
				local example_index = example_count + t
				local labs = self.prediction_writing_info.labels[example_index]
				local row_indices = labs:select(2,2)
				local col_indices = labs:select(2,3)
				local labels = labs:select(2,4)
				for tt = 1,labs:size(1) do
					local int_label = preds[t][row_indices[tt]][col_indices[tt]]
					if(int_label ~= self.null_arc_index) then num_non_null_predictions = num_non_null_predictions + 1 end
					local gt_label = labels[tt]
					if(gt_label == int_label and (gt_label ~= self.null_arc_index)) then num_correct_non_null_predictions = num_correct_non_null_predictions + 1 end
					local str_label = self.labels_int2string[int_label]
					prediction_writer:write(str_label.."\n")
				end
			end
		end

		example_count = example_count + num_actual_data
	end
	if(prediction_writer) then 
		assert(example_count == #self.prediction_writing_info.labels)
		prediction_writer:close() 
		cmd="sh "..self.prediction_writing_info.evaluation_script.." "..results_file.." > "..results_file..".results &"
		print('writing official scoring script results to:\n'..results_file..".results")
		os.execute(cmd)
	end

	print('processed '..example_count.."  evaluation examples in "..batch_count.." batches")
	local pit_all = nn.JoinTable(1):forward(pits)
	local git_all = nn.JoinTable(1):forward(gits)

	local gt_non_null = git_all:ne(self.null_arc_index)

	assert(report_about_rejected_examples)

	print(report_about_rejected_examples)
	local ground_truth_count = gt_non_null:sum()*1.0 + report_about_rejected_examples.total_rejected_positive_arcs
	local correct_detections = pit_all:eq(git_all):cmul(gt_non_null):sum()*1.0
	local detections = pit_all:ne(self.null_arc_index):float():sum()*1.0

	if(prediction_writer) then
		print('num detections = '..detections)
		print('num correct = '..num_correct_non_null_predictions)
		assert(num_correct_non_null_predictions == correct_detections)
		assert(detections == num_non_null_predictions, detections.." "..num_non_null_predictions)
	end

	--this computes micro-averaged precision and recall
	local precision = correct_detections/detections
	local recall = correct_detections/ground_truth_count

	print(string.format('precision_numerator precision_denominator recall_numerator recall_denominator'))
	print(correct_detections,detections,correct_detections,ground_truth_count)


	local f1 = 2*precision*recall/(precision + recall)
	print('\nEvaluation')
	print(string.format(name..': F1: %f, Prec: %f, Rec: %f',f1,precision,recall))

	self.predictor:training()
end
