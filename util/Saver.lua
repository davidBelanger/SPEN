local Saver = torch.class('Saver')


function Saver:__init(out_base_name, things_to_save)
	self.out_base_name = out_base_name
	self.things_to_save = things_to_save
end

function Saver:save(group_name)
	--TODO: add a call to clearState()
	
	for thing_name, thing in pairs(self.things_to_save) do
		local outname = self.out_base_name.."."..group_name.."."..thing_name
		print('writing: '..outname)
		torch.save(outname, thing)
	end
end
