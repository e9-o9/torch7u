-- ============================================================================
-- GaugeTransformer: Gauge Transformations for Geometric Unity
-- ============================================================================
-- Implements the action of the "tilted gauge group" on GU fields.
-- In GU, the gauge group acts on both the base and fiber components
-- of the Observerse, with the specific structure determined by the
-- geometry of the Chimeric Bundle.
-- ============================================================================

local GaugeTransformer, parent = torch.class('gu.GaugeTransformer', 'nn.Module')

function GaugeTransformer:__init(dim, config)
    parent.__init(self)
    
    config = config or {}
    
    self.dim = dim
    self.gauge_type = config.gauge_type or 'tilted'  -- 'tilted', 'standard', 'inhomogeneous'
    self.learnable = config.learnable ~= false  -- Default true
    
    -- Gauge transformation matrix
    -- For tilted gauge group, this is not a standard Lie group element
    self.gauge_matrix = torch.Tensor(dim, dim)
    self.gradGaugeMatrix = torch.Tensor(dim, dim)
    
    -- For inhomogeneous gauge group, we also have a translation component
    if self.gauge_type == 'inhomogeneous' then
        self.translation = torch.Tensor(dim)
        self.gradTranslation = torch.Tensor(dim)
    end
    
    -- Base space transformation (for full Observerse transformation)
    self.base_dim = config.base_dim or 4
    self.base_matrix = torch.Tensor(self.base_dim, self.base_dim)
    self.gradBaseMatrix = torch.Tensor(self.base_dim, self.base_dim)
    
    self:reset()
end

function GaugeTransformer:reset()
    -- Initialize gauge matrix close to identity
    self.gauge_matrix:eye(self.dim)
    
    -- Add small perturbation for learnable case
    if self.learnable then
        self.gauge_matrix:add(torch.randn(self.dim, self.dim):mul(0.01))
    end
    
    -- Initialize base matrix as identity
    self.base_matrix:eye(self.base_dim)
    
    if self.gauge_type == 'inhomogeneous' then
        self.translation:zero()
    end
end

function GaugeTransformer:updateOutput(input)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        -- Transform both base and fiber components
        local base_tensor = input.base
        local fiber_tensor = input.fiber
        
        -- Ensure 2D
        local base_was_1d = base_tensor:dim() == 1
        local fiber_was_1d = fiber_tensor:dim() == 1
        
        local base_view = base_was_1d and base_tensor:view(1, -1) or base_tensor
        local fiber_view = fiber_was_1d and fiber_tensor:view(1, -1) or fiber_tensor
        
        -- Apply gauge transformation to fiber (main gauge action)
        local transformed_fiber = torch.mm(fiber_view, self.gauge_matrix:t())
        
        -- Apply base transformation (coordinate transformation)
        local transformed_base = torch.mm(base_view, self.base_matrix:t())
        
        -- Add translation for inhomogeneous gauge
        if self.gauge_type == 'inhomogeneous' then
            transformed_fiber:add(self.translation:view(1, -1):expandAs(transformed_fiber))
        end
        
        -- Reshape if needed
        if base_was_1d then
            transformed_base = transformed_base:view(self.base_dim)
        end
        if fiber_was_1d then
            transformed_fiber = transformed_fiber:view(self.dim)
        end
        
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.output = ObserverseTensor.create(transformed_base, transformed_fiber)
    else
        -- Transform single tensor
        local was_1d = input:dim() == 1
        local input_view = was_1d and input:view(1, -1) or input
        
        self.output = torch.mm(input_view, self.gauge_matrix:t())
        
        if self.gauge_type == 'inhomogeneous' then
            self.output:add(self.translation:view(1, -1):expandAs(self.output))
        end
        
        if was_1d then
            self.output = self.output:view(self.dim)
        end
    end
    
    return self.output
end

function GaugeTransformer:updateGradInput(input, gradOutput)
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        local base_tensor = input.base
        local fiber_tensor = input.fiber
        local gradBase = gradOutput.base
        local gradFiber = gradOutput.fiber
        
        -- Ensure 2D
        local base_was_1d = base_tensor:dim() == 1
        local fiber_was_1d = fiber_tensor:dim() == 1
        
        local gradBase_view = base_was_1d and gradBase:view(1, -1) or gradBase
        local gradFiber_view = fiber_was_1d and gradFiber:view(1, -1) or gradFiber
        
        -- Backprop through gauge transformation
        local grad_fiber_input = torch.mm(gradFiber_view, self.gauge_matrix)
        local grad_base_input = torch.mm(gradBase_view, self.base_matrix)
        
        -- Reshape if needed
        if base_was_1d then
            grad_base_input = grad_base_input:view(self.base_dim)
        end
        if fiber_was_1d then
            grad_fiber_input = grad_fiber_input:view(self.dim)
        end
        
        local ObserverseTensor = require 'gu.ObserverseTensor'
        self.gradInput = ObserverseTensor.create(grad_base_input, grad_fiber_input)
    else
        local was_1d = input:dim() == 1
        local gradOutput_view = was_1d and gradOutput:view(1, -1) or gradOutput
        
        self.gradInput = torch.mm(gradOutput_view, self.gauge_matrix)
        
        if was_1d then
            self.gradInput = self.gradInput:view(self.dim)
        end
    end
    
    return self.gradInput
end

function GaugeTransformer:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    
    if not self.learnable then
        return
    end
    
    local is_observerse = type(input) == 'table' and input._type == 'ObserverseTensor'
    
    if is_observerse then
        local fiber_tensor = input.fiber
        local gradFiber = gradOutput.fiber
        local base_tensor = input.base
        local gradBase = gradOutput.base
        
        -- Ensure 2D
        local fiber_was_1d = fiber_tensor:dim() == 1
        local base_was_1d = base_tensor:dim() == 1
        
        local fiber_view = fiber_was_1d and fiber_tensor:view(1, -1) or fiber_tensor
        local gradFiber_view = fiber_was_1d and gradFiber:view(1, -1) or gradFiber
        local base_view = base_was_1d and base_tensor:view(1, -1) or base_tensor
        local gradBase_view = base_was_1d and gradBase:view(1, -1) or gradBase
        
        -- Gradient for gauge_matrix
        self.gradGaugeMatrix:addmm(scale, gradFiber_view:t(), fiber_view)
        
        -- Gradient for base_matrix
        self.gradBaseMatrix:addmm(scale, gradBase_view:t(), base_view)
        
        -- Gradient for translation
        if self.gauge_type == 'inhomogeneous' then
            self.gradTranslation:add(scale, gradFiber_view:sum(1):view(self.dim))
        end
    else
        local was_1d = input:dim() == 1
        local input_view = was_1d and input:view(1, -1) or input
        local gradOutput_view = was_1d and gradOutput:view(1, -1) or gradOutput
        
        -- Gradient for gauge_matrix
        self.gradGaugeMatrix:addmm(scale, gradOutput_view:t(), input_view)
        
        -- Gradient for translation
        if self.gauge_type == 'inhomogeneous' then
            self.gradTranslation:add(scale, gradOutput_view:sum(1):view(self.dim))
        end
    end
end

function GaugeTransformer:parameters()
    local params = {self.gauge_matrix, self.base_matrix}
    local gradParams = {self.gradGaugeMatrix, self.gradBaseMatrix}
    
    if self.gauge_type == 'inhomogeneous' then
        table.insert(params, self.translation)
        table.insert(gradParams, self.gradTranslation)
    end
    
    return params, gradParams
end

function GaugeTransformer:__tostring()
    return string.format('%s(dim=%d, base_dim=%d, type=%s, learnable=%s)',
        torch.type(self), self.dim, self.base_dim, self.gauge_type, tostring(self.learnable))
end

return GaugeTransformer
