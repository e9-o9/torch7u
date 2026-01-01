-- nnn: Nested Neural Nets Functional Operator System
-- Transforms nn.* operations to work with nested tensors (nestors)
-- Similar to how nn.* transforms * with tensor embeddings,
-- nnn.* transforms * with nestor (nested tensor) metagraph embeddings

local nnn = {}

-- Import utilities from nn
nnn.NestedTensor = require('nn.NestedTensor')
nnn.PrimeFactorType = require('nn.PrimeFactorType')

-- Module registry for transformed modules
nnn._registry = {}

-- Core transformation function: wraps any nn module to work with nested tensors
function nnn.transform(module, config)
    config = config or {}
    
    local NestedOperator, parent = torch.class('nnn.NestedOperator', 'nn.Module')
    
    function NestedOperator:__init(wrappedModule)
        parent.__init(self)
        self.module = wrappedModule
        self.maxDepth = config.maxDepth or 10
        self.aggregation = config.aggregation or 'preserve'  -- 'preserve', 'flatten', 'mean'
    end
    
    -- Process nested structure recursively
    function NestedOperator:processNested(input, depth)
        depth = depth or 0
        
        if depth > self.maxDepth then
            error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
        end
        
        -- Base case: tensor input
        if torch.isTensor(input) then
            return self.module:forward(input)
        end
        
        -- Recursive case: nested table structure
        if type(input) == 'table' then
            local outputs = {}
            for k, v in pairs(input) do
                outputs[k] = self:processNested(v, depth + 1)
            end
            return outputs
        end
        
        error("input must be a tensor or table")
    end
    
    function NestedOperator:updateOutput(input)
        self.output = self:processNested(input, 0)
        return self.output
    end
    
    -- Backward pass through nested structure
    function NestedOperator:processNestedBackward(input, gradOutput, depth)
        depth = depth or 0
        
        -- Base case: tensor input
        if torch.isTensor(input) then
            return self.module:backward(input, gradOutput)
        end
        
        -- Recursive case: nested table structure
        if type(input) == 'table' then
            local gradInputs = {}
            for k, v in pairs(input) do
                gradInputs[k] = self:processNestedBackward(v, gradOutput[k], depth + 1)
            end
            return gradInputs
        end
        
        error("input must be a tensor or table")
    end
    
    function NestedOperator:updateGradInput(input, gradOutput)
        self.gradInput = self:processNestedBackward(input, gradOutput, 0)
        return self.gradInput
    end
    
    function NestedOperator:accGradParameters(input, gradOutput, scale)
        -- The wrapped module's parameters are updated directly
        -- We just need to ensure backward is called with correct inputs
        local function accumulate(inp, gradOut, depth)
            depth = depth or 0
            
            if torch.isTensor(inp) then
                self.module:accGradParameters(inp, gradOut, scale)
            elseif type(inp) == 'table' then
                for k, v in pairs(inp) do
                    accumulate(v, gradOut[k], depth + 1)
                end
            end
        end
        
        accumulate(input, gradOutput, 0)
    end
    
    function NestedOperator:type(type, tensorCache)
        parent.type(self, type, tensorCache)
        self.module:type(type, tensorCache)
        return self
    end
    
    function NestedOperator:clearState()
        self.module:clearState()
        return parent.clearState(self)
    end
    
    function NestedOperator:reset(stdv)
        if self.module.reset then
            self.module:reset(stdv)
        end
    end
    
    return NestedOperator(module)
end

-- Convenience wrapper for common nn modules
function nnn.wrap(module)
    return nnn.transform(module)
end

-- Create nnn versions of common nn modules
-- These maintain compatibility while adding nested tensor support

-- nnn.Sequential: extends nn.Sequential for nested tensors
function nnn.Sequential()
    local wrapper = nnn.transform(nn.Sequential())
    wrapper.__typename = 'nnn.Sequential'
    return wrapper
end

-- nnn.Linear: extends nn.Linear for nested tensors
function nnn.Linear(inputSize, outputSize, bias)
    local wrapper = nnn.transform(nn.Linear(inputSize, outputSize, bias))
    wrapper.__typename = 'nnn.Linear'
    return wrapper
end

-- nnn.Tanh: extends nn.Tanh for nested tensors
function nnn.Tanh()
    local wrapper = nnn.transform(nn.Tanh())
    wrapper.__typename = 'nnn.Tanh'
    return wrapper
end

-- nnn.ReLU: extends nn.ReLU for nested tensors
function nnn.ReLU(inplace)
    local wrapper = nnn.transform(nn.ReLU(inplace))
    wrapper.__typename = 'nnn.ReLU'
    return wrapper
end

-- nnn.Sigmoid: extends nn.Sigmoid for nested tensors
function nnn.Sigmoid()
    local wrapper = nnn.transform(nn.Sigmoid())
    wrapper.__typename = 'nnn.Sigmoid'
    return wrapper
end

-- nnn.SoftMax: extends nn.SoftMax for nested tensors
function nnn.SoftMax()
    local wrapper = nnn.transform(nn.SoftMax())
    wrapper.__typename = 'nnn.SoftMax'
    return wrapper
end

-- Modal Classifier: extends any criterion to work with nested tensors
local NestedCriterion, criterionParent = torch.class('nnn.NestedCriterion', 'nn.Criterion')

function NestedCriterion:__init(criterion)
    criterionParent.__init(self)
    self.criterion = criterion
    self.maxDepth = 10
end

-- Process nested structure for criterion
function NestedCriterion:processNested(input, target, depth)
    depth = depth or 0
    
    if depth > self.maxDepth then
        error(string.format('Nesting depth %d exceeds maximum depth %d', depth, self.maxDepth))
    end
    
    -- Base case: tensor input
    if torch.isTensor(input) and torch.isTensor(target) then
        return self.criterion:forward(input, target)
    end
    
    -- Recursive case: nested table structure
    if type(input) == 'table' and type(target) == 'table' then
        local totalLoss = 0
        local count = 0
        for k, v in pairs(input) do
            totalLoss = totalLoss + self:processNested(v, target[k], depth + 1)
            count = count + 1
        end
        -- Average loss across branches
        return totalLoss / count
    end
    
    error("input and target must both be tensors or tables")
end

function NestedCriterion:updateOutput(input, target)
    self.output = self:processNested(input, target, 0)
    return self.output
end

-- Backward pass for nested criterion
function NestedCriterion:processNestedGrad(input, target, depth)
    depth = depth or 0
    
    -- Base case: tensor input
    if torch.isTensor(input) and torch.isTensor(target) then
        return self.criterion:backward(input, target)
    end
    
    -- Recursive case: nested table structure
    if type(input) == 'table' and type(target) == 'table' then
        local gradInputs = {}
        local count = 0
        for k in pairs(input) do
            count = count + 1
        end
        
        for k, v in pairs(input) do
            gradInputs[k] = self:processNestedGrad(v, target[k], depth + 1)
            -- Scale gradient by number of branches
            if torch.isTensor(gradInputs[k]) then
                gradInputs[k]:div(count)
            end
        end
        return gradInputs
    end
    
    error("input and target must both be tensors or tables")
end

function NestedCriterion:updateGradInput(input, target)
    self.gradInput = self:processNestedGrad(input, target, 0)
    return self.gradInput
end

-- Factory function for nnn.Criterion
function nnn.Criterion(criterion)
    return nnn.NestedCriterion(criterion)
end

-- Specific criterion wrappers
function nnn.MSECriterion()
    return nnn.NestedCriterion(nn.MSECriterion())
end

function nnn.ClassNLLCriterion(weights)
    return nnn.NestedCriterion(nn.ClassNLLCriterion(weights))
end

function nnn.BCECriterion(weights)
    return nnn.NestedCriterion(nn.BCECriterion(weights))
end

function nnn.CrossEntropyCriterion(weights)
    return nnn.NestedCriterion(nn.CrossEntropyCriterion(weights))
end

-- Embedding modules from nn
nnn.NestedEmbedding = require('nn.NestedEmbedding')
nnn.NestedNeuralNet = require('nn.NestedNeuralNet')

-- Factory function for creating nnn versions of any nn module
function nnn.fromNN(moduleName, ...)
    local nnModule = nn[moduleName]
    if not nnModule then
        error(string.format("nn.%s does not exist", moduleName))
    end
    
    local instance = nnModule(...)
    return nnn.transform(instance)
end

-- Helper to check if input is nested
function nnn.isNested(input)
    return type(input) == 'table' and not torch.isTensor(input)
end

-- Get depth of nested structure
function nnn.depth(input)
    return nnn.NestedTensor.depth(input)
end

-- Flatten nested structure
function nnn.flatten(input)
    return nnn.NestedTensor.flatten(input)
end

-- Clone nested structure
function nnn.clone(input)
    return nnn.NestedTensor.clone(input)
end

-- Apply function to all tensors in nested structure
function nnn.map(input, func)
    return nnn.NestedTensor.map(input, func)
end

-- Module version info
nnn._VERSION = '1.0.0'
nnn._DESCRIPTION = 'Nested Neural Nets (NNN) - Functional operators for nested tensor operations'

-- Future extensions placeholder
nnn.operad = {}
nnn.operad._FUTURE = 'Operad gadgets indexed by prime factorizations as orbifold symmetries'

return nnn
