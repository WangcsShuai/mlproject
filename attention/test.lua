require 'nn'
require 'dataset'
require 'torch'

cmd = torch.CmdLine()
cmd:option('-data','','data file name')
opts = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
local data = parsedatafordnn(opts.data)
print(data[{{1,10},{}}])

-- shuffle the tensor row wise
function getshuffleddata(datatensor)
	local shuffle = torch.randperm(datatensor:size())