function SquaredLossPerBatchItem()
	local y1 = nn.Identity()()
	local y2 = nn.Identity()()
	local diff_squared = nn.Square()(nn.CSubTable()({y1,y2}))

	local loss = nn.Sum(2)(nn.Reshape(-1)(diff_squared))

	return nn.gModule({y1,y2},{loss})
end

