function parsedata(filename)
	local linetotarget = {}
	local overalldata = {}
	for line in io.lines(filename) do
		local splits = line:split(' ')
		local fname = splits[1]
		local target=tonumber(splits[2])
		local data = getdatafromfile(fname)
		local nsamples = data:size(1)
		overalldata[#overalldata+1] = data
		linetotarget[#linetotarget+1] = target
	end

	return overalldata,linetotarget
end

function getdatafromfile(fname)
	local tabletensor ={}
	local linecount = 1
	for line in io.lines(fname) do
		linecount= linecount + 1
	end
	local outputtensor = torch.Tensor(linecount,2)	
	local c = 1 
	for line in io.lines(fname) do
		local splits = line:split('\t')
		local val1 = splits[1]
		local val2 = splits[2]
		if val2 ~= 0 then
			outputtensor[{{c},{1}}] = tonumber(val1)
			outputtensor[{{c},{2}}] = tonumber(val2)
			c = c + 1
		end
		
	end
	return outputtensor
end