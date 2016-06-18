local Diag, parent = torch.class('nn.Diag', 'nn.Module')

function Diag:__init()
   parent.__init(self)
end

function Diag:diagByReference(x)
	assert(x:isContiguous())
	if(x:dim() == 2) then
		return self:diagByReference2(x)
	elseif(x:dim() == 3) then
		return self:diagByReference3(x)
	else
		assert(false,'only supported for 2d or 3d tensors')
	end
end

function Diag:diagByReference2(x)
	assert(x:size(1) == x:size(2),'must be a batch of square tensors')
	local n = x:size(1)
	return self.output.new(x:storage(),1,n,n+1) --this is like calling torch.Tensor(...), but it will give you whatever type the module is set for (by looking at the type of self.output)
end

function Diag:diagByReference3(x)
	local b = x:size(1)
	local n = x:size(2)
	assert(x:size(2) == x:size(3),'must be a batch of square tensors')

	local sizes = torch.LongStorage({b,n})
	local strides = torch.LongStorage({n*n,n+1})
	return self.output.new(x:storage(),1,sizes,strides)
end

function Diag:updateOutput(input)
   self.output = self:diagByReference(input)
   return self.output
end

function Diag:updateGradInput(input, gradOutput)
    local d = input:size(2)
	local b = input:size(1)

   if(input:dim() == 3) then
	   self.gradInput = self.gradInput:resize(b,d,d)
   	else
	   self.gradInput = self.gradInput:resize(b,d)
   	end

   self.gradInput:zero() 
   self.diagGradInput = self:diagByReference(self.gradInput)

   self.diagGradInput:copy(gradOutput)
   return self.gradInput
end




