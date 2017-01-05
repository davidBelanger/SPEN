local HammingEvaluator, parent = torch.class('HammingEvaluator','Evaluator')

--todo: this should take a factory for an iterator

function HammingEvaluator:__init(test_batcher,predict_func)
	parent.__init(self,test_batcher, predict_func)
end

--todo: implement narrowing on first dim to eliminate padding
function HammingEvaluator:score_prediction(y_pred,y)
	return y_pred:eq(y:typeAs(y_pred)):sum()
end
