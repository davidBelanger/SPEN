local MLE = torch.class('MLE')

--MLE training of Gibbs dist. with P(y) propto exp(-energy_net(y))
function MLE:__init(preprocess_net, energy_net,logZ_net,preprocess_labels_net)
	self.loss_net = self:construct_loss_net(preprocess_net, energy_net, logZ_net, preprocess_labels_net)
end

function MLE:construct_loss_net(preprocess_net, energy_net, logZ_net, preprocess_labels_net)
	local x = nn.Identity()()
	local y = nn.Identity()()

	local conditioning_values = preprocess_net(x)
	print(preprocess_labels_net)
	local y0 = preprocess_labels_net and preprocess_labels_net(y) or y

	--score net is assumed to be the energy function, 
	local score_on_ground_truth = nn.MulConstant(-1)(energy_net({y0,conditioning_values})) --note that this is y-x, rather than x-y, to reflect the convention used elsewhere in the code
	local logZ = logZ_net(conditioning_values)
	local log_likelihood = nn.CSubTable()({score_on_ground_truth,logZ})
	
	local log_loss = nn.MulConstant(-1)(log_likelihood)
	return nn.gModule({x,y},{log_loss})
end


function MLE:accumulate_gradient(x,y)
	local loss_value = self.loss_net:forward({x,y})
	self.bg = self.bg or loss_value:clone():fill(1.0/y:size(1))
	self.loss_net:backward({x,y},self.bg)
	return loss_value:mean()
end