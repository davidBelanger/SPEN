require 'nn'
require 'GradientDirectionEfficient'

local h = 10
local h2 = 15
local left = nn.Sequential():add(nn.Linear(h,h)):add(nn.Square()):add(nn.Linear(h,5))
local right = nn.Sequential():add(nn.Linear(h2,h2)):add(nn.Square()):add(nn.Linear(h2,5))

local energy = nn.Sequential():add(nn.ParallelTable():add(left):add(right)):add(nn.CMulTable()):add(nn.Square()):add(nn.Linear(5,1)):add(nn.MulConstant(100))


local dat = {torch.randn(32,h),torch.randn(32,h2)}

local gd = nn.Sequential():add(nn.GradientDirectionEfficient(energy,1,0.0001,false)):add(nn.MulConstant(-1))

local e2 = energy:clone():add(nn.MulConstant(-1))
local gd2 = nn.GradientDirectionEfficient(e2,1,0.0001,false)
local o = gd:forward(dat)
local bg = torch.randn(o:size())

-- print('ooo')
-- print(gd:updateGradInput(dat,bg)[1])
-- print('---')
-- print(gd2:updateGradInput(dat,bg)[1])

local p,g = gd:getParameters()
local p2,g2 = gd2:getParameters()
gd:zeroGradParameters()
gd2:zeroGradParameters()
print('ooo')
gd:forward(dat)
gd:backward(dat,bg)
print('xxx')
gd2:forward(dat)
gd2:backward(dat,bg)

local diff0 = gd.gradInput[1] - gd2.gradInput[1]
print(diff0:norm())

local diff1 = gd.gradInput[2] - gd2.gradInput[2]
print(diff1:norm())
local diff = g - g2
print(diff:norm())

