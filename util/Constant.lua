--this is copied from the DPNN project

------------------------------------------------------------------------
--[[ Constant ]]--
-- Outputs a constant value given an input.
-- If nInputDim is specified, uses the input to determine the size of 
-- the batch. The value is then replicated over the batch.
-- You can use this with nn.ConcatTable() to append constant inputs to
-- an input : nn.ConcatTable():add(nn.Constant(v)):add(nn.Identity()) .
------------------------------------------------------------------------
local Constant, parent = torch.class("nn.Constant", "nn.Module")

function Constant:__init(value)
   self.value = value
   if torch.type(self.value) == 'number' then
      self.value = torch.Tensor{self.value}
   end
   assert(torch.isTensor(self.value), "Expecting number or tensor at arg 1")
   parent.__init(self)
   self.output = nil
end

function Constant:updateOutput(input)
   -- if self.nInputDim and input:dim() > self.nInputDim then
   --    local vsize = self.value:size():totable()
   --    self.output = self.output or torch.Tensor(table.unpack(vsize)):typeAs(self.value)

   --    self.output:resize(input:size(1), table.unpack(vsize))
   --    local value = self.value:view(1, table.unpack(vsize))
   --    self.output:copy(value:expand(self.output:size())) 
   -- else

   self.output = self.output or torch.Tensor(self.value:size()):typeAs(self.value)
   self.output:resize(self.value:size()):copy(self.value)
   --end
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
   if((not torch.isTensor(input)) or self.gradInput:nDimension() == 0) then
      self.gradInput = self:zeros(input)
   end

   return self.gradInput
end


function Constant:training()
end

function Constant:evaluate()
end

