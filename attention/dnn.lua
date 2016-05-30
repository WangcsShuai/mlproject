require 'nn'
require 'dataset'
require 'xlua'
require 'optim'
-- require 'cunn'
-- require 'cutorch'
require 'torch'

local cmd = torch.CmdLine()
cmd:option('-traindata','train1.scp','training data scp file')
cmd:option('-testdata','test.scp','test data scp file')
cmd:option('-mode','CPU', 'CPU|GPU')
cmd:option('-epochs',1000,'Max epochs')
cmd:option('-optimization','SGD','Optimization method')
cmd:option('-batchSize',1,'batchSize')


local opts = cmd:parse(arg or {})
torch.setdefaulttensortype('torch.FloatTensor')

-- Define the model

model = nn.Sequential()
model:add(nn.Linear(2,32))
model:add(nn.LeakyReLU())
model:add(nn.Linear(32,32))
model:add(nn.LeakyReLU())

model:add(nn.Linear(32,4))
model:add(nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()

if opts.mode == 'GPU' then
    model:cuda()
    criterion:cuda()
end

-- This line must be writtent after the final model is settled.
local parameters, gradparams = model:getParameters()
----------------------------------------------------------------------
-- load the data

function normalizedata(data)
    local mean = {}
    local stdv = {}
    for i=1, data:size(2)-1 do
        mean[i]=data[{{},i}]:mean()
        stdv[i]=data[{{},i}]:std()
        data[{{},i}]:add(-mean[i])
        data[{{},i}]:div(stdv[i])
    end
    return data
end
local trainingdata = normalizedata(parsedatafordnn(opts.traindata))
--local testdata = normalizedata(parsedatafordnn(opts.testdata))
local confusion = optim.ConfusionMatrix(4)


optimstate = {
    learningRate = 1.5,
    learningRateDecay = 0.1,
    momentum = 0.9,
}

function train(dataset)
    epoch = epoch or 1
    model:training()
    confusion:zero()
    local nsamples = dataset:size(1)
    local nfeatdim = dataset:size(2) - 1

    local shuffle = torch.randperm(nsamples)
    local batchSize = opts.batchSize
    
    for t = 1, nsamples, batchSize do
        xlua.progress(t,nsamples)
        inputs = {}
        targets = {}
        
        local k = 1
        for i = t, math.min(t+batchSize-1, nsamples) do
            local sample = dataset[shuffle[i]]
            local input = sample[{{1,nfeatdim}}]:totable()
            local target = sample[{{nfeatdim+1}}][1]
            if opts.mode == 'GPU' then
                input = input:cuda()
                target = target:cuda()
            end
            inputs[k] = input
            targets[k] = target
            k = k + 1
        end
        local feval = function()
            gradparams:zero()
            inputs = torch.Tensor(inputs)
            targets = torch.Tensor(targets)
            local outputs = model:forward(inputs)
            local error = criterion:forward(outputs,targets)
            local gradoutputs = criterion:backward(outputs,targets)
            model:backward(inputs,gradoutputs)
            confusion:batchAdd(outputs, targets)
            return error,gradparams
        end
        optim.sgd(feval,parameters,optimstate)
    end
    print(confusion)
    epoch = epoch + 1
end

while true do
    train(trainingdata)
end

     
