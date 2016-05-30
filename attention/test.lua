require 'nn'
require 'data'
require 'torch'

cmd = torch.CmdLine()
cmd:option('-data','','data file name')
opts = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
local data = parsedata(opts.data)
print(data[{{1,10},{}}])

