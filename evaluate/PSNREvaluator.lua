local PSNREvaluator, parent = torch.class('PSNREvaluator','Evaluator')

function PSNREvaluator:__init(test_batcher,predict_func)
	parent.__init(self,test_batcher, predict_func)
end

function PSNREvaluator:score_prediction(y_pred,y)
	y_pred:clamp(0,1)
	local rmse = math.sqrt((y_pred - y):pow(2):mean())
	local psnr = 20*math.log10(y:max()/rmse)
	return psnr
end
