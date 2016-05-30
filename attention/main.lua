require 'nn'
require 'data'
require 'attention'
require 'xlua'
require 'optim'
--require 'cunn'
--require 'cutorch'
local cmd = torch.CmdLine()
cmd:option('-data','','Datafile')
cmd:option('-epochs',1000,'Number of epochs')
cmd:option('-mode','CPU','Whether use cuda')

local opts = cmd:parse(arg)

local train,target = parsedata(opts.data)

criterion = nn.ClassNLLCriterion()

model = nn.Sequential()

model:add(nn.Linear(2,32))
model:add(nn.LeakyReLU())

-- model:add(nn.Linear(32,32))
model:add(nn.Attention(32))
model:add(nn.Linear(32,4))
model:add(nn.LogSoftMax())

local trainutts = #train

if opts.mode == 'GPU' then
	model:cuda()
	criterion:cuda()
end

local parameters,gradparams = model:getParameters()

local confusionmat = optim.ConfusionMatrix(4)

optimstate = {
	learningRate = 1.5,
	learningRateDecay = 0.1,
	momentum = 0.9,
	nesterov=true,
	dampening=0
}

for i=1,trainutts do
	local mean = train[i]:mean(1)
	local std = train[i]:std(1)
	for j=1,train[i]:size(1)do
		train[i][j]:csub(mean)
		train[i][j]:cdiv(std)
	end
end

local accumulateERR =0
for epoch=1,opts.epochs do
	local randp = torch.randperm(trainutts)
	confusionmat:zero()

	accumulateERR =0

	for i=1,trainutts do
		xlua.progress(i,trainutts)
		
		local idx = randp[i]

		if opts.mode == 'GPU' then
			train[idx] = train[idx]:cuda()
		end

		local output,eer
		local function feval()
			gradparams:zero()
			output = model:forward(train[idx])
			eer = criterion:forward(output,target[idx])
			local gradoutputs = criterion:backward(output,target[idx])
			model:backward(train[idx],gradoutputs)
			return eer,gradparams
		end
		optim.sgd(feval,parameters,optimstate)
		accumulateERR = accumulateERR + eer
		confusionmat:add(output,target[idx])	
		confusionmat:updateValids()
	end

	if epoch % 1 == 0 then
		print(confusionmat)	
		print(accumulateERR)
	end

end
