-- one utterance one tensor,return a table of tensor
function parsedataforattention(filename)
	local table_alldata = {}
	for line in io.lines(filename) do
		local splits = line:split(' ')
		local fname = splits[1]
		local label = splits[2]
		local tensor_data = readDataFromFile(fname)
		local nsamples = tensor_data:size(1)
		local nfeaturedim = tensor_data:size(2)
		local tensor_all = torch.Tensor(nsamples, nfeaturedim+1)
		tensor_all[{{1,nsamples},{1,nfeaturedim}}]:copy(tensor_data)
		tensor_all[{{1,nsamples},nfeaturedim+1}]:fill(label)
		table.insert(table_alldata, tensor_all)
	end
	return table_alldata
end

-- all data one tensor,return a tensor.
function parsedatafordnn(filename)
	local table_alldata = {}
	for line in io.lines(filename) do
		local splits = line:split(' ')
		local fname = splits[1]
		local label = splits[2]
        for line in io.lines(fname) do
            local splits = line:split('\t')
            local energy = splits[1]
            local f0 = splits[2]
            table.insert(table_alldata,{energy,f0,label})
        end
	end
	return torch.Tensor(table_alldata)
end

function readDataFromFile(filename)
	local table_data = {}
	for line in io.lines(filename) do
		local splits = line:split('\t')
		local energy = splits[1]
		local f0 = splits[2]
		table.insert(table_data,{energy,f0})
	end
	return torch.Tensor(table_data)
end

function getdatafromCSV(filename)
	local linetotarget = {}
	local overalldata = {}
	for line in io.lines(filename) do
		local splits = line:split(',')
		local featdim = #splits - 1
		--print(splits)
		local temp = torch.Tensor(splits)
		local data = temp[{{1,featdim}}]
		local target = tonumber(splits[featdim+1])
		overalldata[#overalldata+1] = data:totable()
		linetotarget[#linetotarget+1] = target + 1
	end

	return torch.Tensor(overalldata),torch.Tensor(linetotarget)
end

		
