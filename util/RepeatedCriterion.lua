local RepeatedCriterion, parent = torch.class('nn.RepeatedCriterion', 'nn.Criterion')

function RepeatedCriterion:__init(criterion,weights)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
   self.weights = weights
   self.sizeAverage = true
   self.called_once = false
end

function RepeatedCriterion:add(criterion, weight)
   error("should not use this")
end

function RepeatedCriterion:updateOutput(input, target)
   self.output = 0
   local scale = self.sizeAverage and 1/#input or 1.0

   for i = 1,#input do
      local weight = self.weights and scale*self.weights[i] or scale
      self.output = self.output + weight*self.criterion:updateOutput(input[i],target)
   end
   return self.output
end

function RepeatedCriterion:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   local scale = self.sizeAverage and 1/#input or 1.0

   for i = 1,#input do
      local weight = self.weights and scale*self.weights[i] or scale

      if(not self.called_once) then
         local bg = self.criterion:updateGradInput(input[i], target):clone()
         self.criterion:updateOutput(input[i], target)
         local bg2 = self.criterion:updateGradInput(input[i], target)
         assert(bg:eq(bg2):all(),'RepeatedCriterion assumes that computation in the inner criterion is stateless')
         self.called_once = true
      end
      nn.utils.recursiveAdd(self.gradInput[i], weight, self.criterion:updateGradInput(input[i], target))
   end
   return self.gradInput
end

function RepeatedCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end

