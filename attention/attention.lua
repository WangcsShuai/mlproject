require 'rnn'
local Attention, parent = torch.class('nn.Attention','nn.Module')

function Attention:__init(inputsize,outputsize)
   assert(inputsize,"First argument is inputsize")
   parent.__init(self)
   self.inputsize = inputsize
   self.outputsize = inputsize
   -- build the model
   self.module = self:buildModel()
end

function Attention:buildModel()

   local model = nn.Sequential()
   local concat = nn.ConcatTable()
   -- A simple pass through, which does nothing just passes arguments on
   local passthrough = nn.Sequential()
   passthrough:add(nn.Identity())

   -- The model for the softmax output for timestepst t=1,N
   local softmaxout = nn.Sequential()
   local attention = nn.Sequential()
   local dimreduce = nn.Sequential()
   dimreduce:add(nn.Linear(self.inputsize,1))
   dimreduce:add(nn.Tanh())
   -- Collapse all dimensions
   softmaxout:add(nn.View(-1))
   softmaxout:add(nn.SoftMax())

   -- Add to the attentnion model
   attention:add(dimreduce)
   attention:add(softmaxout)
   -- Restructure to make it possible for MM:
   -- Batchsize B and Dimension D :
   -- ( D X B ) X ( B X 1)
   attention:add(nn.View(-1,1))
   concat:add(attention)
   concat:add(passthrough)

   -- Add the two segment part to the model
   model:add(concat)
   -- Fuse the outputs to a single vector
   -- Calculate \sum_t^T h_t * alpha_t , whereas h_t are the transposed
   -- columns of the input matrix H, and alpha_t are the values within the
   -- reduced vector by the attention model
   model:add(nn.MM(true,false))
   -- Remove extra dimensions
   model:add(nn.View(-1))
   return model
end

function Attention:updateOutput(inputTensor)
   assert(torch.isTensor(inputTensor), "expecting input tensor")
   assert(inputTensor:dim() == 2,"Currently only batch mode is supported")
   local dim = inputTensor:size(2)
   -- Resize the backprop buffers
   -- self.outputs = self.outputs:resize(timesteps)
   -- we output a single vector after concatenating all the timesteps
   self.output = self.output:resize(dim):zero()
   -- Do the forwarding and store the denominators
   -- self.outputs = self.module:forward(trimmedinput):contiguous()
   -- Do the batchwise normalization
   -- local softmaxout = self.softmaxmodule:forward(self.outputs):contiguous()
   self.output:copy(self.module:forward(inputTensor))
   -- print(self.module.modules[1].modules[2].modules[2].modules[2].output:size())

   return self.output
end

function Attention:updateGradInput(inputTensor, gradOutput)
   assert(torch.isTensor(gradOutput), "expecting gradOutput Tensor")

   self.gradInput:resizeAs(inputTensor):zero()
   -- Back propagation is done the following:
   -- We compute the derivative of \sum a_t h_t = \sum_t \delta a_t * h_t
   -- Thus we compute the derivative of the sigmoid function first, then
   -- pass this derivative to the TanH module which computes our final derivatives
   self.gradInput:copy(self.module:backward(inputTensor,gradOutput))
   return self.gradInput
end

function Attention:__tostring__()
   return string.format('%s(Sequence -> %i)', torch.type(self), self.inputsize)
end
