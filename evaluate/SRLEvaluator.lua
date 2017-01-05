local SRLEvaluator, parent = torch.class('SRLEvaluator')

--todo: this should take a factory for an iterator

function SRLEvaluator:__init(batcher,predictor,null_arc_index)
	self.batcher = batcher
	self.predictor = predictor
	self.null_arc_index = null_arc_index
end

--todo: implement narrowing on first dim to eliminate padding
function SRLEvaluator:evaluate(name)
	print('Beginning Evaluation')
	self.predictor:evaluate()

	local pits = {}
	local gits = {}
	
	local exampleCount = 0
	local batch_iterator = self.batcher:get_iterator()
	local batch_count = 0
	local report_about_rejected_examples
	while(true) do
		local batch_data = batch_iterator()

		local batch_labels, batch_inputs, num_actual_data = unpack(batch_data)
		if(batch_inputs == nil) then 
			report_about_rejected_examples = batch_labels
			break 
		end
		batch_count = batch_count + 1

		local preds = self.predictor:predict(batch_inputs)
		local pit = preds:narrow(1,1,num_actual_data)

		local git = batch_labels:narrow(1,1,num_actual_data)
		table.insert(pits,pit:clone())
		table.insert(gits,git:clone())	
	
		exampleCount = exampleCount + num_actual_data
	end

	print('processed '..exampleCount.."  evaluation examples in "..batch_count.." batches")
	local pit_all = nn.JoinTable(1):forward(pits)
	local git_all = nn.JoinTable(1):forward(gits)

	local gt_non_null = git_all:ne(self.null_arc_index)

	assert(report_about_rejected_examples)

	print(report_about_rejected_examples)
	local ground_truth_count = gt_non_null:sum()*1.0 + report_about_rejected_examples.total_rejected_positive_arcs
	local correct_detections = pit_all:eq(git_all):cmul(gt_non_null):sum()*1.0
	local detections = pit_all:ne(self.null_arc_index):sum()*1.0

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
