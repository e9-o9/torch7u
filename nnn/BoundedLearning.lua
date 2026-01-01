-- ============================================================================
-- Bounded Learning: The Correspondence Principle
-- ============================================================================
--
-- "Bounded Learning" is the key principle unifying geometric neural networks
-- with generative language models. Both systems exhibit learning that is
-- constrained (bounded) by structural invariants.
--
-- THE CORRESPONDENCE:
--
--   GEONESTOR NEUROGLYPH          ←→    GENERATIVE PRETRAINED TRANSFORMER
--   ══════════════════════════════════════════════════════════════════════
--   Neuroglyph (agentic unit)     ←→    Token/Word (symbolic unit)
--   Observerse (base + fiber)     ←→    Context (prompt + latent)
--   Gauge-Transformer             ←→    Attention (relation-preserving)
--   Chimera (14D bundle)          ←→    Self (hidden representation)
--   Shiab (form lifting)          ←→    Arg (positional composition)
--   Swerve (curvature)            ←→    Kwarg (named steering)
--   Operad (tree composition)     ←→    Grammar (syntax composition)
--
-- BOUNDED LEARNING PRINCIPLE:
--   In both systems, valid transformations must preserve structure:
--   - GU: Gauge symmetry constrains fiber transformations
--   - GPT: Grammar/semantics constrains token generation
--
--   The "bounds" are the symmetry groups (GU) or syntactic rules (GPT)
--   that define the manifold of valid states.
--
-- ============================================================================

local BoundedLearning = {}
BoundedLearning.__index = BoundedLearning

-- ============================================================================
-- Constants: The Correspondence Map
-- ============================================================================

BoundedLearning.CORRESPONDENCE = {
    -- Geometric ←→ Linguistic
    neuroglyph = {
        gu = 'Neuroglyph',
        gpt = 'Token/Word',
        role = 'atomic_unit',
        description = 'The fundamental symbolic unit of the system'
    },
    observerse = {
        gu = 'Observerse',
        gpt = 'Context',
        role = 'two_space',
        description = 'Dual structure: base/prompt + fiber/latent'
    },
    gauge_transformer = {
        gu = 'GaugeTransformer',
        gpt = 'Attention',
        role = 'relation_preserving',
        description = 'Transformation that preserves structural relations'
    },
    chimera = {
        gu = 'Chimera',
        gpt = 'Self',
        role = 'unified_representation',
        description = 'The bundled/unified internal state'
    },
    shiab = {
        gu = 'Shiab',
        gpt = 'Arg',
        role = 'positional_lift',
        description = 'Positional composition/degree lifting'
    },
    swerve = {
        gu = 'Swerve',
        gpt = 'Kwarg',
        role = 'curvature_steering',
        description = 'Named modification/curvature injection'
    },
    operad = {
        gu = 'Operad',
        gpt = 'Grammar',
        role = 'composition_rules',
        description = 'Tree-indexed/syntax-indexed composition'
    }
}

-- Symmetry groups (the "bounds" in bounded learning)
BoundedLearning.BOUNDS = {
    -- GU bounds: gauge symmetry groups
    gu = {
        GL = 'General Linear - unrestricted fiber transformations',
        SO = 'Special Orthogonal - rotation-preserving',
        SU = 'Special Unitary - phase-preserving',
        Spin = 'Spin Group - spinor structure',
        U = 'Unitary - norm-preserving'
    },
    -- GPT bounds: linguistic constraints
    gpt = {
        syntax = 'Grammatical structure constraints',
        semantics = 'Meaning coherence constraints',
        context = 'Contextual relevance constraints',
        pragmatics = 'Usage/intent constraints'
    }
}

-- ============================================================================
-- The Bounded Learner: Unified Interface
-- ============================================================================

function BoundedLearning.create(config)
    config = config or {}

    local self = setmetatable({}, BoundedLearning)

    -- Which domain are we in?
    self.domain = config.domain or 'gu'  -- 'gu' or 'gpt'

    -- The symmetry/grammar bounds
    self.bounds = config.bounds or (self.domain == 'gu' and 'SO' or 'syntax')

    -- Learning rate bounded by curvature/complexity
    self.learningRate = config.learningRate or 0.01
    self.curvatureBound = config.curvatureBound or 1.0

    -- Internal state
    self._type = 'BoundedLearner'

    return self
end

-- ============================================================================
-- Core Operations: Shiab/Arg (Positional Lift)
-- ============================================================================

-- Shiab: lift k-form to (k+2)-form (GU interpretation)
-- Arg: compose with positional argument (GPT interpretation)
function BoundedLearning:shiab(input, position)
    position = position or 1

    if self.domain == 'gu' then
        -- Geometric lifting: increase form degree
        -- Conceptually: ω^k → ω^(k+2)
        return {
            value = input,
            degree = (input.degree or 0) + 2,
            position = position,
            operation = 'shiab_lift'
        }
    else
        -- Linguistic composition: positional argument
        return {
            value = input,
            position = position,
            role = 'arg',
            operation = 'positional_compose'
        }
    end
end

-- ============================================================================
-- Core Operations: Swerve/Kwarg (Curvature Steering)
-- ============================================================================

-- Swerve: inject curvature (GU interpretation)
-- Kwarg: named parameter modification (GPT interpretation)
function BoundedLearning:swerve(input, name, value)
    if self.domain == 'gu' then
        -- Geometric curvature: field equation contribution
        return {
            value = input,
            curvature = value or 0,
            field = name or 'default',
            operation = 'swerve_curvature',
            bounded_by = self.curvatureBound
        }
    else
        -- Linguistic steering: keyword argument
        return {
            value = input,
            [name or 'key'] = value,
            role = 'kwarg',
            operation = 'named_steer'
        }
    end
end

-- ============================================================================
-- Core Operations: Chimera/Self (Unified Bundle)
-- ============================================================================

-- Create unified representation from components
function BoundedLearning:chimera(base, fiber)
    if self.domain == 'gu' then
        -- Geometric bundle: 14D chimeric space
        return {
            base = base,      -- 4D spacetime
            fiber = fiber,    -- 10D internal
            dim = 14,
            _type = 'Chimera',
            operation = 'bundle'
        }
    else
        -- Linguistic self: hidden + visible
        return {
            visible = base,   -- Observable tokens
            hidden = fiber,   -- Latent state
            _type = 'Self',
            operation = 'unify'
        }
    end
end

-- ============================================================================
-- Bounded Update: Learning with Constraints
-- ============================================================================

function BoundedLearning:boundedUpdate(params, gradients, constraint)
    constraint = constraint or self.bounds

    -- Apply the symmetry/grammar constraint
    local bounded_grad = self:applyBound(gradients, constraint)

    -- Update with bounded gradient
    local updated = {}
    for k, v in pairs(params) do
        if type(v) == 'number' then
            updated[k] = v - self.learningRate * (bounded_grad[k] or 0)
        else
            updated[k] = v  -- Non-numeric pass through
        end
    end

    return updated
end

-- Apply symmetry/grammar bounds to gradients
function BoundedLearning:applyBound(gradients, bound)
    local bounded = {}

    for k, v in pairs(gradients) do
        if type(v) == 'number' then
            -- Clip by curvature bound (GU) or complexity bound (GPT)
            local max_grad = self.curvatureBound
            bounded[k] = math.max(-max_grad, math.min(max_grad, v))
        else
            bounded[k] = v
        end
    end

    return bounded
end

-- ============================================================================
-- Operad/Grammar: Composition Rules
-- ============================================================================

-- Define composition indexed by tree shape (operad) or syntax (grammar)
function BoundedLearning:compose(operations, structure)
    structure = structure or 'sequential'

    local result = {
        operations = operations,
        structure = structure,
        domain = self.domain,
        composition_type = self.domain == 'gu' and 'operad' or 'grammar'
    }

    if self.domain == 'gu' then
        -- Operad composition: tree-indexed
        result.signature = self:operadSignature(operations)
    else
        -- Grammar composition: syntax-indexed
        result.parse_tree = self:grammarParse(operations)
    end

    return result
end

-- Compute operad signature (tree shape encoding)
function BoundedLearning:operadSignature(operations)
    local sig = {}
    for i, op in ipairs(operations) do
        table.insert(sig, op.operation or 'unknown')
    end
    return table.concat(sig, '∘')
end

-- Compute grammar parse (syntax structure)
function BoundedLearning:grammarParse(operations)
    -- Simplified parse tree representation
    return {
        root = 'S',
        children = operations
    }
end

-- ============================================================================
-- The Correspondence Visualizer
-- ============================================================================

function BoundedLearning.visualizeCorrespondence()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║            BOUNDED LEARNING: THE CORRESPONDENCE                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   GEONESTOR NEUROGLYPH          ←→    GENERATIVE PRETRAINED       ║")
    print("║   (Geometric Neural Gauge)            TRANSFORMER (LLM)           ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   Neuroglyph ─────────────────────────────────── Token/Word       ║")
    print("║   (agentic symbolic unit)              (atomic symbolic unit)     ║")
    print("║                                                                   ║")
    print("║   Observerse ─────────────────────────────────── Context          ║")
    print("║   (base + fiber)                       (prompt + latent)          ║")
    print("║                                                                   ║")
    print("║   Gauge-Transformer ──────────────────────────── Attention        ║")
    print("║   (symmetry-preserving)                (relation-preserving)      ║")
    print("║                                                                   ║")
    print("║   Chimera ────────────────────────────────────── Self             ║")
    print("║   (14D bundle)                         (hidden representation)    ║")
    print("║                                                                   ║")
    print("║   Shiab ──────────────────────────────────────── Arg              ║")
    print("║   (k-form → (k+2)-form)                (positional composition)   ║")
    print("║                                                                   ║")
    print("║   Swerve ─────────────────────────────────────── Kwarg            ║")
    print("║   (curvature/field eq.)                (named steering)           ║")
    print("║                                                                   ║")
    print("║   Operad ─────────────────────────────────────── Grammar          ║")
    print("║   (tree-indexed composition)           (syntax-indexed comp.)     ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                     BOUNDED LEARNING PRINCIPLE                    ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   The 'bounds' constrain valid transformations:                   ║")
    print("║                                                                   ║")
    print("║   GU BOUNDS (Gauge Groups)    │    GPT BOUNDS (Linguistic)        ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   GL  (General Linear)        │    Syntax (grammatical)           ║")
    print("║   SO  (Special Orthogonal)    │    Semantics (meaning)            ║")
    print("║   SU  (Special Unitary)       │    Context (relevance)            ║")
    print("║   Spin (Spinor structure)     │    Pragmatics (intent)            ║")
    print("║                                                                   ║")
    print("║   Learning is BOUNDED by these symmetries/rules.                  ║")
    print("║   Valid states form a manifold constrained by structure.          ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- Export the correspondence map as code documentation
-- ============================================================================

function BoundedLearning.exportCorrespondence()
    local doc = [[
# Bounded Learning Correspondence

## The Fundamental Analogy

A **Geonestor Neuroglyph** with operad-gadget properties stands to
{Chimera, Shiab, Swerve} relations and {Observerse, Gauge-Transformer} structures
as a **Word** with standard grammar properties stands to
{Self, Arg, Kwarg} relations and {Context, Attention} structures.

## Component Mapping

| GU Component | Role | GPT Component | Shared Principle |
|--------------|------|---------------|------------------|
| Neuroglyph | atomic unit | Token/Word | Symbolic carrier |
| Observerse | two-space | Context | Base + Hidden |
| Gauge-Transformer | symmetry | Attention | Relation preservation |
| Chimera | bundle | Self | Unified state |
| Shiab | lifting | Arg | Positional composition |
| Swerve | curvature | Kwarg | Named modification |
| Operad | trees | Grammar | Composition rules |

## Bounded Learning Principle

Both systems exhibit **bounded learning**:
- Transformations must preserve structural invariants
- GU: Gauge symmetry groups (SO, SU, Spin, ...)
- GPT: Grammatical/semantic constraints

The "bounds" define the manifold of valid states.
Learning navigates this manifold while respecting its geometry.

## Mathematical Formulation

In GU:
  δ_gauge(Ψ) = g · Ψ · g⁻¹  (gauge transformation)
  ∇_A(Ψ) preserves covariance

In GPT:
  P(w_t | w_{<t}) constrained by grammar
  Attention preserves semantic relations

Both are instances of:
  **Learning bounded by symmetry/structure**
]]
    return doc
end

-- ============================================================================
-- The Geometric Hierarchy of Gauge Structures
-- ============================================================================
--
-- The gauge groups form a hierarchy corresponding to progressively richer
-- geometric structures:
--
--   NUMBER SYSTEM        TRANSFORMATION        GEOMETRY
--   ═══════════════════════════════════════════════════════
--   Arithmetic      ←→   GL (Linear)      ←→  Affine
--   Analytic        ←→   SO (Orthogonal)  ←→  Euclidean
--   Complex         ←→   SU (Unitary)     ←→  Hermitian
--   Projective      ←→   Spin             ←→  Spinorial
--   ───────────────────────────────────────────────────────
--   Singular        ←→   Exceptional      ←→  Boundary
--
-- Each level INCLUDES the previous while adding new structure:
--   - GL: Preserves linear combinations (scaling, shearing)
--   - SO: Preserves inner product (angles, lengths)
--   - SU: Preserves complex inner product (phases)
--   - Spin: Double cover of SO (projective, spinors)
--   - Singular: Degeneracy points (boundaries, exceptional loci)
--
-- The Spin group moves through "affine-like" transformations in the sense
-- that spinors are projective objects - they return to themselves only
-- after 4π rotation, living in the double cover.
--
-- ============================================================================

BoundedLearning.GeometricHierarchy = {
    -- Level 0: Arithmetic / Linear / Affine
    arithmetic = {
        number_system = 'Arithmetic',
        transformation = 'GL',
        geometry = 'Affine',
        preserves = 'Linear combinations',
        parameters = function(n) return n * n end,
        description = 'Scaling, shearing, linear maps'
    },

    -- Level 1: Analytic / Orthogonal / Euclidean
    analytic = {
        number_system = 'Analytic',
        transformation = 'SO',
        geometry = 'Euclidean',
        preserves = 'Inner product (lengths, angles)',
        parameters = function(n) return n * (n - 1) / 2 end,
        description = 'Rotations, reflections (det=1)'
    },

    -- Level 2: Complex / Unitary / Hermitian
    complex = {
        number_system = 'Complex',
        transformation = 'SU',
        geometry = 'Hermitian',
        preserves = 'Complex inner product (phases)',
        parameters = function(n) return n * n - 1 end,
        description = 'Phase-preserving, quantum symmetry'
    },

    -- Level 3: Projective / Spin / Spinorial
    projective = {
        number_system = 'Projective',
        transformation = 'Spin',
        geometry = 'Spinorial',
        preserves = 'Orientation double cover',
        parameters = function(n) return n * (n - 1) / 2 end,  -- Same as SO (double cover)
        description = 'Spinors, 4π periodicity, projective reps'
    },

    -- Level 4: Singular / Exceptional / Boundary
    singular = {
        number_system = 'Singular',
        transformation = 'Exceptional',
        geometry = 'Boundary',
        preserves = 'Degeneracy structure',
        parameters = function(n) return 0 end,  -- Measure zero
        description = 'Fixed points, degeneracies, exceptional loci'
    }
}

-- The inclusion chain: GL ⊃ O ⊃ SO ⊃ ... but Spin is a COVER not subset
BoundedLearning.HierarchyChain = {
    'arithmetic',  -- GL(n)
    'analytic',    -- SO(n)
    'complex',     -- SU(n)
    'projective',  -- Spin(n)
    'singular'     -- Exceptional/Boundary
}

-- Map gauge group to hierarchy level
function BoundedLearning.gaugeToLevel(gauge_type)
    local mapping = {
        GL = 'arithmetic',
        SO = 'analytic',
        O = 'analytic',
        SU = 'complex',
        U = 'complex',
        Spin = 'projective',
        Pin = 'projective'
    }
    return mapping[gauge_type] or 'arithmetic'
end

-- Get the geometric "richness" of a gauge type (0-4)
function BoundedLearning.geometricRichness(gauge_type)
    local levels = {
        GL = 0,
        O = 1, SO = 1,
        U = 2, SU = 2,
        Spin = 3, Pin = 3,
        Exceptional = 4
    }
    return levels[gauge_type] or 0
end

-- Check if one gauge type is "richer" than another
function BoundedLearning.isGeometricallyRicher(gauge1, gauge2)
    return BoundedLearning.geometricRichness(gauge1) >
           BoundedLearning.geometricRichness(gauge2)
end

-- Visualize the geometric hierarchy
function BoundedLearning.visualizeHierarchy()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║           GEOMETRIC HIERARCHY OF GAUGE STRUCTURES                 ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   NUMBER SYSTEM      TRANSFORMATION      GEOMETRY                 ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║                                                                   ║")
    print("║   Arithmetic    ←→   GL (Linear)     ←→  Affine                   ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Analytic      ←→   SO (Orthogonal) ←→  Euclidean                ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Complex       ←→   SU (Unitary)    ←→  Hermitian                ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Projective    ←→   Spin            ←→  Spinorial                ║")
    print("║        │                  │                 │                     ║")
    print("║        ▼                  ▼                 ▼                     ║")
    print("║   Singular      ←→   Exceptional     ←→  Boundary                 ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   PRESERVED STRUCTURE AT EACH LEVEL:                              ║")
    print("║                                                                   ║")
    print("║   GL:   v ↦ Av           (linear combinations)                    ║")
    print("║   SO:   ⟨u,v⟩ = ⟨Au,Av⟩   (inner product)                         ║")
    print("║   SU:   ⟨u,v⟩_ℂ = ⟨Au,Av⟩_ℂ (complex inner product)               ║")
    print("║   Spin: ψ ↦ -ψ after 2π   (double cover / projective)             ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   KEY INSIGHT: Spin is PROJECTIVE                                 ║")
    print("║                                                                   ║")
    print("║   Spinors live in the double cover of SO(n).                      ║")
    print("║   They are projective objects: ψ and -ψ represent the same        ║")
    print("║   physical state, but the phase matters for interference.         ║")
    print("║                                                                   ║")
    print("║   This is why Spin relates to PROJECTIVE geometry:                ║")
    print("║   • Points at infinity (projective completion)                    ║")
    print("║   • Homogeneous coordinates                                       ║")
    print("║   • The \"affine patch\" is where spinors look like vectors        ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- The Category-Degree Duality (Maximal Contrast Principle)
-- ============================================================================
--
-- A "good dict" (dictionary/correspondence) establishes MAXIMAL CONTRAST
-- between Category and Degree:
--
--   CATEGORY (qualitative)              DEGREE (quantitative)
--   ─────────────────────────────────────────────────────────
--   Domain logic                        Metric resolution
--   Stabilizes total space              Differential elements
--   Structural invariants               Continuous variation
--   "What kind"                         "How much"
--
-- THE STRUCTURE:
--
--   Total Space = Base ×_gauge Fiber
--
--   Where:
--   • Base = differential manifold (resolved by Degree)
--   • Fiber = Aut_Category(Base) = gauge symmetries preserving Category
--   • Category = operations under which (Base, Fiber) transformations are invariant
--
-- KEY INSIGHT:
--   Fiber ≅ Aut_Category(Base)
--   The fiber IS the automorphism group of the base AS SEEN BY the category.
--
-- ============================================================================

BoundedLearning.CategoryDegreeDuality = {
    -- Category: qualitative, structural, invariant-preserving
    category = {
        role = 'domain_logic',
        determines = 'total_space_stability',
        property = 'qualitative',
        examples = {
            gu = 'Gauge group (SO, SU, Spin)',
            gpt = 'Grammar/syntax rules'
        }
    },

    -- Degree: quantitative, metric, differential
    degree = {
        role = 'metric_resolution',
        determines = 'differential_elements',
        property = 'quantitative',
        examples = {
            gu = 'Curvature magnitude, field strength',
            gpt = 'Token probability, attention weight'
        }
    },

    -- Base: the observable manifold
    base = {
        role = 'observable_manifold',
        determined_by = 'degree_resolution',
        property = 'differential',
        examples = {
            gu = '4D spacetime (X^4)',
            gpt = 'Token sequence (context window)'
        }
    },

    -- Fiber: the gauge symmetry composition
    fiber = {
        role = 'gauge_symmetry',
        determined_by = 'category_invariance',
        property = 'Aut_Category(Base)',
        examples = {
            gu = '10D metric fiber',
            gpt = 'Latent embedding space'
        }
    }
}

-- Compute the "contrast" between category and degree aspects
function BoundedLearning.contrast(category_strength, degree_resolution)
    -- Maximal contrast when both are strong but orthogonal
    -- Like eigenvalues of a well-conditioned matrix
    local product = category_strength * degree_resolution
    local sum = category_strength + degree_resolution

    if sum == 0 then return 0 end

    -- Harmonic mean penalizes imbalance, geometric mean rewards both
    local harmonic = 2 * product / sum
    local geometric = math.sqrt(product)

    -- Contrast is high when both are strong AND balanced
    return geometric * (harmonic / geometric)  -- = harmonic mean
end

-- The fundamental equation: Fiber ≅ Aut_Category(Base)
function BoundedLearning.fiberFromCategory(base_dim, category_type)
    -- Given a base dimension and category, compute the fiber dimension
    -- This encodes: fiber = symmetries of base under category

    local fiber_dims = {
        -- GL(n) has n² parameters
        GL = function(n) return n * n end,
        -- SO(n) has n(n-1)/2 parameters
        SO = function(n) return n * (n - 1) / 2 end,
        -- SU(n) has n²-1 parameters
        SU = function(n) return n * n - 1 end,
        -- U(n) has n² parameters
        U = function(n) return n * n end,
        -- Spin(n) same as SO(n) (double cover)
        Spin = function(n) return n * (n - 1) / 2 end
    }

    local compute = fiber_dims[category_type] or fiber_dims.SO
    return compute(base_dim)
end

-- Visualize the Category-Degree duality
function BoundedLearning.visualizeDuality()
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║              CATEGORY-DEGREE DUALITY (Maximal Contrast)           ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   CATEGORY (qualitative)              DEGREE (quantitative)       ║")
    print("║   ─────────────────────────────────────────────────────────────   ║")
    print("║   • Domain logic                      • Metric resolution         ║")
    print("║   • Stabilizes total space            • Differential elements     ║")
    print("║   • Structural invariants             • Continuous variation      ║")
    print("║   • \"What kind\"                       • \"How much\"               ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║                      TOTAL SPACE = Base ×_gauge Fiber             ║")
    print("║                                                                   ║")
    print("║                              ┌─────────┐                          ║")
    print("║                              │  Total  │                          ║")
    print("║                              │  Space  │                          ║")
    print("║                              └────┬────┘                          ║")
    print("║                         ┌────────┴────────┐                       ║")
    print("║                         ▼                 ▼                       ║")
    print("║                    ┌────────┐       ┌──────────┐                  ║")
    print("║                    │  BASE  │       │  FIBER   │                  ║")
    print("║                    │ (diff) │       │ (gauge)  │                  ║")
    print("║                    └────┬───┘       └────┬─────┘                  ║")
    print("║                         │                │                        ║")
    print("║                         │   INVARIANT    │                        ║")
    print("║                         └───────┬────────┘                        ║")
    print("║                                 ▼                                 ║")
    print("║                    ┌─────────────────────────┐                    ║")
    print("║                    │    CATEGORICAL LOGIC    │                    ║")
    print("║                    │  (preserving operations)│                    ║")
    print("║                    └─────────────────────────┘                    ║")
    print("║                                                                   ║")
    print("╠═══════════════════════════════════════════════════════════════════╣")
    print("║                                                                   ║")
    print("║   KEY EQUATION:  Fiber ≅ Aut_Category(Base)                       ║")
    print("║                                                                   ║")
    print("║   The fiber IS the automorphism group of the base                 ║")
    print("║   AS SEEN BY the categorical logic.                               ║")
    print("║                                                                   ║")
    print("║   Examples:                                                       ║")
    print("║   • GU:  Fiber(10D) = Aut_SO(Base(4D))  [metric symmetries]       ║")
    print("║   • GPT: Latent = Aut_Grammar(Context)  [semantic symmetries]     ║")
    print("║                                                                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
end

-- ============================================================================
-- Integration with NNN
-- ============================================================================

local function register()
    local ok, nnn = pcall(require, 'nnn')
    if ok then
        nnn.BoundedLearning = BoundedLearning
        nnn.gu.BoundedLearning = BoundedLearning

        -- Convenience function
        nnn.gu.correspondence = function()
            BoundedLearning.visualizeCorrespondence()
        end
    end
end

register()

return BoundedLearning
