

local a = torch.LongTensor({1,2,3,4})

local target = torch.Tensor(5,6):zero()

local source = torch.randn(4,6)
target:scatter(1,a,source)

