local Constant, parent = torch.class("nn.Constant", "nn.Module")

function Constant:__init(value)
   self.value = value
   if torch.type(self.value) == 'number' then
      self.value = torch.Tensor(self.value)
   end
   parent.__init(self)
   self.gradInput_initialized = false
   self.output = nil
end

function Constant:updateOutput(input)
   self.output = self.output or torch.Tensor(self.value:size()):typeAs(self.value)
   self.output:resize(self.value:size()):copy(self.value)
   return self.output
end

function Constant:zeros(input)
   if(torch.isTensor(input)) then
     return input:clone():zero()
   else
     return Util:deep_apply(input,function(t) return t:clone():zero() end)
   end
end

function Constant:updateGradInput(input, gradOutput)
   assert(input)
   if(not self.gradInput_initialized) then
      self.gradInput = self:zeros(input)
      self.gradInput_initialized = true
   end

   return self.gradInput
end


function Constant:training()
end

function Constant:evaluate()
end

