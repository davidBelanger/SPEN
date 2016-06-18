local InterpCriterion, parent = torch.class('nn.InterpCriterion', 'nn.Criterion')

function InterpCriterion:__init(criterion,weight)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
   self.weights = weights
   self.sizeAverage = true
end

function InterpCriterion:add(criterion, weight)
   error("should not use this")
end

function InterpCriterion:updateOutput(input, target)
   self.output = 0
   self.target_i = self.target_i or target:clone()
   local target_i = self.target_i
   target_i:resizeAs(target)
   for i = 1,#input do
      local weight = self.weights and self.weights[i] or 1.0
      local gamma = i/#input
      target_i:copy(input[1]):mul(1-gamma):add(gamma,target)
      self.output = self.output + weight*self.criterion:updateOutput(input[i],target_i)
   end

   if(self.sizeAverage) then self.output = self.output/#input end
   return self.output
end

function InterpCriterion:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   local target_i = self.target_i 
   assert(target_i)
   
   local scale = self.sizeAverage and 1/#input or 1.0
   for i = 1,#input do
      local weight = self.weights and scale*self.weights[i] or scale
      local gamma = i/#input
      target_i:copy(input[1]):mul(1-gamma):add(gamma,target)
      nn.utils.recursiveAdd(self.gradInput[i], weight, self.criterion:updateGradInput(input[i], target))
   end

   return self.gradInput
end

function InterpCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end

