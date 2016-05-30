require 'optim'

local cmd = torch.CmdLine()
cmd:option('-traindata','train1.scp','train_data Datafile')
cmd:option('-testdata','test1.scp','test_data Datafile')
cmd:option('-epochs',10000,'Number of epochs')
cmd:option('-mode','CPU','Whether use cuda')

local opts = cmd:parse(arg)

confusion = optim.ConfusionMatrix(4)
function parsedata(filename)
	local linetotarget = {}
	local overalldata = {}
	local predicttarget = {}

	for line in io.lines(filename) do
		local splits = line:split(' ')
		local fname = splits[1]
		local target=tonumber(splits[2])
        print("---------------------------")
		local predict, data = getdatafromfile(fname)
        print("predict " .. predict)
        print("true " .. target)
        print("---------------------------")
		local nsamples = data:size(1)
		overalldata[#overalldata+1] = data
		linetotarget[#linetotarget+1] = tonumber(target)
		predicttarget[#predicttarget+1] = tonumber(predict)
	end

	return linetotarget, predicttarget
end

function getdatafromfile(fname)
	local tabletensor ={}
	local linecount = 1
	for line in io.lines(fname) do
		linecount= linecount + 1
	end
	local outputtensor = torch.Tensor(linecount)
	local c = 1
	for line in io.lines(fname) do
		local splits = line:split('\t')
		local val1 = splits[1]
		local val2 = splits[2]
		if tonumber(val2) ~= 0 then
			outputtensor[c] = tonumber(val2)
			c = c + 1
        end
	end
    d = math.floor(c / 3)
    startvalue = outputtensor[{{1,d}}]:mean()
    middlevalue = outputtensor[{{d+1,2*d}}]:mean()
    endvalue = outputtensor[{{2*d+1,c-1}}]:mean()
    print('startvalue',startvalue)
    print('midlevalue',middlevalue)
    print('endvalue',endvalue)
    
    tmp = torch.Tensor({startvalue,middlevalue,endvalue})
    tmp12 = torch.Tensor({startvalue,middlevalue})
    tmp23 = torch.Tensor({middlevalue,endvalue})
    stdv = torch.std(tmp)+torch.std(tmp12)+torch.std(tmp23)

    local predict

    if startvalue > middlevalue and middlevalue > endvalue then 
        predict =  4
    elseif startvalue < middlevalue and middlevalue < endvalue then 
        predict =  2
    elseif startvalue > middlevalue and middlevalue <  endvalue then 
        predict =  3
    else 
        predict = 1
    end

    if (stdv < 50) then
        predict = 1
    end
    --print(outputtensor)
	return predict,outputtensor
end


local truelabels, predictlabels = parsedata(opts.traindata)
--print(truelables)
--print(predictlabels)
confusion:zero()
--print(predictlabels)
--print(torch.Tensor(predictlabels))
confusion:batchAdd(torch.Tensor(predictlabels),torch.Tensor(truelabels))
print(confusion)
