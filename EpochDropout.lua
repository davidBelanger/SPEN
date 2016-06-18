local EpochDropout, Parent = torch.class('nn.EpochDropout', 'nn.Module')

function EpochDropout:__init(p,v1,sizeTable,inplace)
   Parent.__init(self)
   self.p = p or 0.5
   self.train = true
   self.inplace = inplace
   -- version 2 scales output during training instead of evaluation
   self.v2 = not v1
   if self.p >= 1 or self.p < 0 then
      error('<EpochDropout> illegal percentage, must be 0 <= p < 1')
   end
   self.sizeTable = sizeTable


   self.weight = torch.Tensor(1,table.unpack(self.sizeTable)) --noise
   self.bias = torch.Tensor(1):fill(1) --needFreshNoise
   self.gradWeight = self.weight:clone()
   self.gradBias = self.bias:clone()

end

function EpochDropout:clearState()
end
function EpochDropout:reset()
   self.bias[1] = 1
end

function EpochDropout:makeNoise(input,p)
   -- if(self.bias:dim() == 0) then self.bias = torch.Tensor(1):fill(1):typeAs(input) end
   if(self.bias[1] == 1) then
      -- if(self.weight:dim() == 0) then
      --    self.weight = torch.Tensor(1,table.unpack(self.sizeTable)):typeAs(input) --this shouldn't be necessary, but clearstate is messing things up somehow
      -- end
      self.weight:bernoulli(1-self.p)
      if self.v2 then
         self.weight:div(1-self.p)
      end
      self.bias[1] = 0
   end
end

function EpochDropout:updateOutput(input)
   if self.inplace then
      self.output:set(input)
   else
      self.output:resizeAs(input):copy(input)
   end
   if self.p > 0 then
      if self.train then
         self:makeNoise(input,self.p)
         self.expandedWeight = self.weight:expand(input:size(1),table.unpack(self.sizeTable))
         self.output:cmul(self.expandedWeight)
      elseif not self.v2 then
         self.output:mul(1-self.p)
      end
   end
   return self.output
end

function EpochDropout:updateGradInput(input, gradOutput)
   if self.inplace then
      self.gradInput:set(gradOutput)
   else
      self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   end
   if self.train then
      if self.p > 0 then
         self.gradInput:cmul(self.expandedWeight) -- simply mask the gradients with the noise vector
      end
   else
      if not self.v2 and self.p > 0 then
         self.gradInput:mul(1-self.p)
      end
   end
   return self.gradInput
end

function EpochDropout:setp(p)
   self.p = p
end

function EpochDropout:__tostring__()
   return string.format('%s(%f)', torch.type(self), self.p)
end


function EpochDropout:clearState()

end

