local PrintNoNewline, parent = torch.class('nn.PrintNoNewline', 'nn.Module')

function PrintNoNewline:__init(printInput,printGradOutput,msg,printValues)
   self.msg = msg or ''
	self.printInput = printInput
	self.printGradOutput = printGradOutput
   self.printValues = printValues
end

function PrintNoNewline:updateOutput(input)
   self.output = input
   if(self.printInput) then
   		self:prettyPrintNoNewline(input)
   end
   return self.output
end

function PrintNoNewline:updateGradInput(input, gradOutput)
   self.gradInput = gradOutput
   if(self.printGradOutput) then
   		self:prettyPrintNoNewline(gradOutput)
   end
   return self.gradInput
end

function PrintNoNewline:prettyPrintNoNewline(data)
   if(self.printValues) then
      self.printValues(data)
   else
      self:prettyPrintNoNewlineRecurse(data)
   end
end


function PrintNoNewline:prettyPrintNoNewlineRecurse(data)
   if(torch.isTensor(data) or torch.isStorage(data)) then
         print(data:size())
   else
      print('{')
      for k,v in ipairs(data) do
         self:prettyPrintNoNewlineRecurse(v)
      end
      print('}')

   end

end


