local ExpMul, parent = torch.class('nn.ExpMul', 'nn.Module')

function ExpMul:__init(initValue)
   parent.__init(self)

   self.weight = torch.Tensor(1):fill(initValue)
   self.gradWeight = torch.Tensor(1)
end



function ExpMul:updateOutput(input)
   self.output:resizeAs(input):copy(input);
   local w = math.exp(self.weight[1])
   self.output:mul(w);
   return self.output
end

function ExpMul:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local w = math.exp(self.weight[1])
   self.gradInput:add(w, gradOutput)
   return self.gradInput
end

function ExpMul:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   local w = math.exp(self.weight[1])
   local gExpWeight = scale*input:dot(gradOutput)
   self.gradWeight[1] = self.gradWeight[1] + w*gExpWeight
end

