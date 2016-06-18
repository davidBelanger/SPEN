local SPENOptimizer, Parent = torch.class('SPENOptimizer','Optimizer')


function SPENOptimizer:__init(model,modules_to_update,criterion,options,problem)
    Parent.__init(self,model,modules_to_update,criterion,options)
    self.problem = problem

end
function SPENOptimizer:preBatch()
    self.problem:reset()
end


