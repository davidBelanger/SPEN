local ReshapeAs, parent = torch.class('nn.ReshapeAs', 'nn.Module')

function ReshapeAs:__init()
	   parent.__init(self)
	   self.gradInput = {}
	   self.gradInput[1] = torch.Tensor()
	   self.gradInput[2] = torch.Tensor()
end

function ReshapeAs:updateOutput(input)
   assert(input[1]:isContiguous())
   assert(input[2]:isContiguous())
   self.output:view(input[1], input[2]:size())
   return self.output
end

function ReshapeAs:updateGradInput(input, gradOutput)
   assert(gradOutput:isContiguous())
   self.gradInput[1]:viewAs(gradOutput, input[1]) 
	self.gradInput[2]:resizeAs(input[2])
	self.gradInput[2]:zero()

   return self.gradInput
end


function ReshapeAs:__tostring__()
  return torch.type(self)
end
