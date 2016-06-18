local SelectivelyWeightedMSECriterion, parent = torch.class('nn.SelectivelyWeightedMSECriterion', 'nn.Criterion')

local eps = 1e-12

function SelectivelyWeightedMSECriterion:__init(positiveWeight,condition_function)
   parent.__init(self)
   self.sizeAverage = true
   self.positiveWeight = positiveWeight
   self.condition_function = condition_function
end

function SelectivelyWeightedMSECriterion:updateOutput(input, target)
   self.term1 = self.term1 or input.new()
   self.term1:resizeAs(input)
   self.term1:copy(input):add(-1,target)
   self.term1:pow(2)

   self.term2 = self.term2 or input.new()
   self.term2:resizeAs(target):copy(target)
   self.term2:fill(1.0)
   self.term2[self.condition_function(input,target)] = self.positiveWeight

   self.term1:cmul(self.term2)
   self.output = self.term1:sum()
   if self.sizeAverage then
      self.output = self.output/target:nElement()
   end

   return self.output
end

function SelectivelyWeightedMSECriterion:updateGradInput(input, target)
   self.gradInput = self.gradInput or input.new()
   self.gradInput:resizeAs(input)
   self.gradInput:copy(input):add(-1,target)

   --self.term2 is set in the forward pass
   
   self.gradInput:cmul(self.term2)
   self.gradInput:mul(2)
   if self.sizeAverage then
      self.gradInput:div(target:nElement())
   end

   return self.gradInput
end

