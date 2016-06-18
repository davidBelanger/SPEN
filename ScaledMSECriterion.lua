local ScaledMSECriterion, parent = torch.class('nn.ScaledMSECriterion', 'nn.Criterion')

local eps = 1e-12
--this weights instances where the target is > 0 by positiveWeight

function ScaledMSECriterion:__init(positiveWeight,value)
   parent.__init(self)
   self.sizeAverage = true
   self.positiveWeight = positiveWeight
   self.sliceByValue = value ~= nil
   self.value = value
end

function ScaledMSECriterion:updateOutput(input, target)
   self.term1 = self.term1 or input.new()
   self.term1:resizeAs(input)
   self.term1:copy(input):add(-1,target)
   self.term1:pow(2)

   self.term2 = self.term2 or input.new()
   self.term2:resizeAs(target):copy(target)
   self.term2:fill(1.0)

   local posClass =  self.sliceByValue and self.term2:select(self.term2:dim(),self.value) or self.term2[self.term2:gt(0)]
   posClass:fill(self.positiveWeight)
 
   self.term1:cmul(self.term2)
   self.output = self.term1:sum()
   if self.sizeAverage then
      self.output = self.output/target:nElement()
   end

   return self.output
end

function ScaledMSECriterion:updateGradInput(input, target)
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

