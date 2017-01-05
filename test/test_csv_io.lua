package.path = package.path .. ';util/?.lua'
require 'IO'

local labels = torch.randn(5,3)
local data = torch.randn(5,3,4)


local fn = '/usr/tmp/tst' --you will need to change this if /usr/tmp doesn't exist.

IO:save_csv(fn..".labels", labels)
local labels2 = IO:load_csv(fn..".labels")

local diff = labels - labels2
assert(diff:abs():max() < 1e-10)


