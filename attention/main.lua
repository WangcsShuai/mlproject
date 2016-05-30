require 'nn'
require 'data'
require 'attention'
require 'xlua'
require 'optim'
require 'cunn'
require 'cutorch'

local cmd = torch.CmdLine()
cmd:option('-traindata','train1.scp','train_data Datafile')
cmd:option('-testdata','test1.scp','test_data Datafile')
cmd:option('-epochs',10000,'Number of epochs')
cmd:option('-mode','CPU','Whether use cuda')

local opts = cmd:parse(arg)

local train_data, traintarget = parsedata(opts.traindata)
local test_data, testtarget = parsedata(opts.testdata)

criterion = nn.ClassNLLCriterion()

model = nn.Sequential()

model:add(nn.Linear(2,32))
model:add(nn.LeakyReLU())

-- model:add(nn.Linear(32,32))
model:add(nn.Attention(32))
model:add(nn.Linear(32,4))
model:add(nn.LogSoftMax())

local trainutts = #train_data
local testutts = #test_data

if opts.mode == 'GPU' then
	model:cuda()
	criterion:cuda()
end

local parameters,gradparams = model:getParameters()
local trainconfusion = optim.ConfusionMatrix(4)
local testconfusion = optim.ConfusionMatrix(4)

optimstate = {
	learningRate = 1.5,
	learningRateDecay = 0.1,
	momentum = 0.9,
	nesterov=true,
	dampening=0
}

-- Normalize the data
for i=1,trainutts do
	local mean = train_data[i]:mean(1)
	local std = train_data[i]:std(1)
	for j=1,train_data[i]:size(1)do
		train_data[i][j]:csub(mean)
		train_data[i][j]:cdiv(std)
	end
end

for i=1,testutts do
	local mean = test_data[i]:mean(1)
	local std = test_data[i]:std(1)
	for j=1,test_data[i]:size(1)do
		test_data[i][j]:csub(mean)
		test_data[i][j]:cdiv(std)
	end
end
---------------------------------------------------------
local accumulateERR =0

function train()
	-- randomize the inputs at utterance level
	local randp = torch.randperm(trainutts)
	trainconfusion:zero()
	accumulateERR =0

	for i=1,trainutts do
		xlua.progress(i,trainutts)
		
		local idx = randp[i]

		if opts.mode == 'GPU' then
			train_data[idx] = train_data[idx]:cuda()
		end

		local output,err
		local function feval()
			gradparams:zero()
			output = model:forward(train_data[idx])
			err = criterion:forward(output,traintarget[idx])
			local gradoutputs = criterion:backward(output,traintarget[idx])
			model:backward(train_data[idx],gradoutputs)
			return err,gradparams
		end
		optim.sgd(feval,parameters,optimstate)
		accumulateERR = accumulateERR + err
		trainconfusion:add(output,traintarget[idx])	
		trainconfusion:updateValids()
	end
end

function test()
	testconfusion:zero()
	for i=1, testutts do
		xlua.progress(i,testutts)
		if opts.mode == 'GPU' then
			test_data[i] = test_data[i]:cuda()
		end
		local output = model:forward(test_data[i])
		testconfusion:add(output,testtarget[i])
		testconfusion:updateValids()
	end

end

for epoch=1,opts.epochs do
	train()
	if epoch % 1 == 0 then
		print(trainconfusion)	
		print(accumulateERR)
	end
	test()
	if epoch % 1 == 0 then
		print(testconfusion)	
		print(accumulateERR)
	end
end
