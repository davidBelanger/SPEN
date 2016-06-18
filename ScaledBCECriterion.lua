local ScaledBCECriterion, parent = torch.class('nn.ScaledBCECriterion', 'nn.Criterion')

local eps = 1e-12

function ScaledBCECriterion:__init(positiveWeight,value)
   parent.__init(self)
   self.sizeAverage = true
   self.positiveWeight = positiveWeight
   self.sliceByValue = value ~= nil
   self.value = value
end

function ScaledBCECriterion:updateOutput(input, target)
   -- log(input) * target + log(1 - input) * (1 - target)
   self.term1 = self.term1 or input.new()
   self.term2 = self.term2 or input.new()
   self.term3 = self.term3 or input.new()

   self.term1:resizeAs(input)
   self.term2:resizeAs(input)
   self.term3:resizeAs(input)

   self.term1:fill(1):add(-1,target)
   self.term2:fill(1):add(-1,input):add(eps):log():cmul(self.term1)

   self.term3:copy(input):add(eps):log():cmul(target)
   self.term3:add(self.term2)

   if self.sizeAverage then
      self.term3:div(target:nElement())
   end

   self.term4 = self.term4 or input.new()
   self.term4:resizeAs(target)
   self.term4:copy(target)
   local dim = self.term4:dim()
   local posClass =  nil
   if(self.sliceByValue) then
      posClass = self.term4:select(dim,self.value)
   else
      posClass = self.term4
   end
   posClass:mul(self.positiveWeight)
   self.term4:add(1.0)
   self.term3:cmul(self.term4)
   self.output = - self.term3:sum()

   return self.output
end

function ScaledBCECriterion:updateGradInput(input, target)
   -- target / input - (1 - target) / (1 - input)
   self.term1 = self.term1 or input.new()
   self.term2 = self.term2 or input.new()
   self.term3 = self.term3 or input.new()

   self.term1:resizeAs(input)
   self.term2:resizeAs(input)
   self.term3:resizeAs(input)

   self.term1:fill(1):add(-1,target)
   self.term2:fill(1):add(-1,input)

   self.term2:add(eps)
   self.term1:cdiv(self.term2)

   self.term3:copy(input):add(eps)


   self.term4 =	self.term4 or input.new()
   self.term4:copy(target)
   local dim = self.term4:dim()
   local posClass =  nil
   if(self.sliceByValue) then
      posClass = self.term4:select(dim,self.value)
   else
      posClass = self.term4
   end
   posClass:mul(self.positiveWeight)
   self.term4:add(1.0)

   self.gradInput:resizeAs(input)
   self.gradInput:copy(target):cdiv(self.term3)

   self.gradInput:add(-1,self.term1)

   self.gradInput:cmul(self.term4)
   if self.sizeAverage then
      self.gradInput:div(target:nElement())
   end

   self.gradInput:mul(-1)

   return self.gradInput
end

