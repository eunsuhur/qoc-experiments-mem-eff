"""
heap_types.jl

Heap-allocated replacements for StaticArrays-based types in RobotDynamics/TrajectoryOptimization.
Required for large state dimensions (n > ~100) where SVector{n}/MVector{n} blow up the compiler.
"""

# RD and TO are assumed to be defined in the including file.

# ─────────────────────────────────────────────────────────
#  HeapKnotPoint: Vector-backed knot point for large n
# ─────────────────────────────────────────────────────────

mutable struct HeapKnotPoint{T,N,M} <: RD.AbstractKnotPoint{T,N,M}
    z::Vector{T}
    dt::T
    t::T
end

function HeapKnotPoint{T,N,M}(x::AbstractVector, u::AbstractVector, dt, t=0.0) where {T,N,M}
    HeapKnotPoint{T,N,M}(Vector{T}([x; u]), T(dt), T(t))
end

function HeapKnotPoint(x::AbstractVector, u::AbstractVector, dt::T, t=zero(T)) where T
    n, m = length(x), length(u)
    HeapKnotPoint{T,n,m}(Vector{T}([x; u]), dt, t)
end

# Terminal knot point (no control, dt=0)
function HeapKnotPoint(x::AbstractVector, m::Int, t::T=zero(T)) where T
    HeapKnotPoint{T,length(x),m}(Vector{T}([x; zeros(T, m)]), zero(T), t)
end

@inline RD.state(z::HeapKnotPoint{T,N,M}) where {T,N,M} = view(z.z, 1:N)
@inline RD.control(z::HeapKnotPoint{T,N,M}) where {T,N,M} = view(z.z, N+1:N+M)
@inline RD.is_terminal(z::HeapKnotPoint) = z.dt == 0

function RD.set_state!(z::HeapKnotPoint{T,N,M}, x) where {T,N,M}
    z.z[1:N] .= x
end

function RD.set_control!(z::HeapKnotPoint{T,N,M}, u) where {T,N,M}
    z.z[N+1:N+M] .= u
end

function Base.copy(z::HeapKnotPoint{T,N,M}) where {T,N,M}
    HeapKnotPoint{T,N,M}(copy(z.z), z.dt, z.t)
end

# Provide _x and _u for compatibility with code that accesses z._x, z._u
function Base.getproperty(z::HeapKnotPoint{T,N,M}, s::Symbol) where {T,N,M}
    s === :_x && return 1:N
    s === :_u && return N+1:N+M
    return getfield(z, s)
end

# ─────────────────────────────────────────────────────────
#  Trajectory construction with HeapKnotPoint
# ─────────────────────────────────────────────────────────

function heap_traj(X::Vector, U::Vector, dt::Vector, t::Vector)
    n = length(X[1])
    m = length(U[1])
    T = promote_type(eltype(X[1]), eltype(U[1]), eltype(dt))
    Z = HeapKnotPoint{T,n,m}[]
    for k = 1:length(U)
        push!(Z, HeapKnotPoint{T,n,m}(X[k], U[k], dt[k], t[k]))
    end
    # Terminal knot point
    push!(Z, HeapKnotPoint(X[end], m, t[end]))
    return RD.Traj(Z)
end

# Override Base.copy for Traj of HeapKnotPoint to avoid upstream KnotPoint constructor
function Base.copy(Z::RD.Traj{n,m,T,<:HeapKnotPoint}) where {n,m,T}
    RD.Traj([copy(z) for z in Z])
end

# ─────────────────────────────────────────────────────────
#  HeapProblem: Problem without MVector/SVector for x0/xf
# ─────────────────────────────────────────────────────────

struct HeapProblem{Q<:RD.QuadratureRule,T<:AbstractFloat}
    model::RD.AbstractModel
    obj::TO.AbstractObjective
    constraints::TO.ConstraintList
    x0::Vector{T}
    xf::Vector{T}
    Z::RD.Traj
    N::Int
    t0::T
    tf::T
end

Base.size(prob::HeapProblem) = size(prob.model)..., prob.N
TO.integration(prob::HeapProblem{Q}) where Q = Q
TO.controls(prob::HeapProblem) = TO.controls(prob.Z)

function TO.rollout!(prob::HeapProblem{Q}) where Q
    RD.rollout!(Q, prob.model, prob.Z, prob.x0)
end

# ─────────────────────────────────────────────────────────
#  HeapGoalConstraint: GoalConstraint without SVector/MVector
# ─────────────────────────────────────────────────────────

struct HeapGoalConstraint <: TO.StateConstraint
    n::Int
    xf::Vector{Float64}
    inds::Vector{Int}
end

TO.sense(::HeapGoalConstraint) = TO.Equality()
Base.length(con::HeapGoalConstraint) = length(con.inds)
RD.state_dim(con::HeapGoalConstraint) = con.n
RD.control_dim(::HeapGoalConstraint) = 0
Base.copy(con::HeapGoalConstraint) = HeapGoalConstraint(con.n, copy(con.xf), copy(con.inds))
TO.is_bound(::HeapGoalConstraint) = true

function TO.evaluate(con::HeapGoalConstraint, x::AbstractVector)
    return x[con.inds] - con.xf
end

function TO.jacobian!(J, con::HeapGoalConstraint, x::AbstractVector)
    fill!(J, zero(eltype(J)))
    for (i, j) in enumerate(con.inds)
        J[i, j] = one(eltype(J))
    end
    return true
end

# Override gen_jacobian to return plain Matrix instead of SizedMatrix
function TO.gen_jacobian(con::HeapGoalConstraint, i=1)
    p = length(con)
    w = RD.state_dim(con)
    return zeros(p, w)
end

# widths for StateConstraint is inherited: returns (state_dim(con),)
