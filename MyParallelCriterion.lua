local RepeatedCriterion, parent = torch.class('nn.RepeatedCriterion', 'nn.Criterion')

function RepeatedCriterion:__init(criterion)
   parent.__init(self)
   self.criterion = criterion
   self.gradInput = {}
end

function RepeatedCriterion:add(criterion, weight)
   error("should not use this")
end

function RepeatedCriterion:updateOutput(input, target)
   self.output = 0
   for i = 1,#input do
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i],target)
   end
   return self.output
end

function RepeatedCriterion:updateGradInput(input, target)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   for i = 1,#input do
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target))
   end
   return self.gradInput
end

function RepeatedCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end

