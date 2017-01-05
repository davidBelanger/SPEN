local a = torch.Tensor(5,4):zero()
--semantics: a[inds[i]][i] = v[i]
local inds = torch.LongTensor({2,2,3,4})
inds = inds:view(1,4)
local v = torch.rand(1,4)
a:scatter(1,inds,v)

print(a)


local a = torch.Tensor(5,4):zero()
--semantics: a[i][inds[i]] = v[i]
local inds = torch.LongTensor({2,2,3,4,1})
inds = inds:view(5,1)
local v = torch.rand(5,1)
a:scatter(2,inds,v)

print(a)
print(v)

local a = torch.Tensor(5,2,3,4):zero()
local labs = torch.LongTensor(5,2,3):fill(2)
labs = labs:view(5,2,3,1)
local weights = torch.randn(labs:size())
a:scatter(4,labs,weights)
print(a)
-- local inds = torch.LongTensor({1,2,3,4})
-- inds = inds:view(1,4)
-- local v = torch.rand(1,4)

-- a:scatter(1,inds,v)

-- print(a)



local b = 3
local n = 4
local m = 5
local d = 6

local soft_pred = torch.rand(b*n*m,d):log()
local yp = torch.rand(b*n*m):mul(d):floor():add(1):long()
yp = yp:view(yp:size(1),1)
assert(soft_pred:dim() == 2)

print(soft_pred:size())
print(yp:size())

local instance_weights = torch.rand(b,n,m):view(yp:size())
local log_of_ground_truth = soft_pred:gather(2,yp)
local norm = log_of_ground_truth:norm()
local loss = log_of_ground_truth:cmul(instance_weights):sum()*-1
assert(norm == soft_pred:gather(2,yp):norm(),"the gather did things in place")

local self = {}
self.d_pred = self.d_pred or soft_pred:clone()
self.d_pred:zero()
instance_weights = instance_weights:view(yp:size())
print('--------')
print(self.d_pred:size(),yp:size(),instance_weights:size())
self.d_pred:scatter(2,yp,instance_weights)
self.d_pred:mul(-1)
print(self.d_pred:size())

	-- self.soft_predictor:backward(x,d_loss)

