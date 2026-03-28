"""
shuttling_memeff.jl
"""

WDIR = joinpath(@__DIR__, "../..")
MEM_EFFICIENT_ALTRO_PATH = abspath(joinpath(WDIR, "..", "mem_efficient_altro"))
MEM_EFFICIENT_ALTRO_PATH in LOAD_PATH || pushfirst!(LOAD_PATH, MEM_EFFICIENT_ALTRO_PATH)
include(joinpath(WDIR, "src", "shuttling", "shuttling.jl"))

using Altro
using FFTW
using ForwardDiff
using HDF5
using LinearAlgebra
using Printf

const TO = Altro.TrajectoryOptimization
const RD = Altro.RobotDynamics

const EXPERIMENT_META = "shuttling"
const EXPERIMENT_NAME = "shuttling_memeff"
const SAVE_PATH = abspath(joinpath(WDIR, "out", EXPERIMENT_META, EXPERIMENT_NAME))

struct ShuttlingDiagonalCost{T} <: TO.CostFunction
    Q::Vector{T}
    R::Vector{T}
    xref::Vector{T}
    uref::Vector{T}
    terminal::Bool
end

TO.state_dim(cost::ShuttlingDiagonalCost) = length(cost.Q)
TO.control_dim(cost::ShuttlingDiagonalCost) = length(cost.R)

function TO.stage_cost(cost::ShuttlingDiagonalCost, x::AbstractVector)
    J = zero(promote_type(eltype(x), eltype(cost.Q), eltype(cost.xref)))
    @inbounds @simd for i in eachindex(cost.Q, x, cost.xref)
        dx = x[i] - cost.xref[i]
        J += cost.Q[i] * dx * dx
    end
    return J / 2
end

function TO.stage_cost(cost::ShuttlingDiagonalCost, x::AbstractVector, u::AbstractVector)
    J = TO.stage_cost(cost, x)
    @inbounds @simd for i in eachindex(cost.R, u, cost.uref)
        du = u[i] - cost.uref[i]
        J += cost.R[i] * du * du / 2
    end
    return J
end

function TO.gradient!(E, cost::ShuttlingDiagonalCost, z::RD.AbstractKnotPoint, cache=nothing)
    fill!(E.grad, zero(eltype(E.grad)))
    x = RD.state(z)
    @inbounds @simd for i in eachindex(cost.Q, x, cost.xref)
        E.q[i] = cost.Q[i] * (x[i] - cost.xref[i])
    end
    if !RD.is_terminal(z)
        u = RD.control(z)
        @inbounds @simd for i in eachindex(cost.R, u, cost.uref)
            E.r[i] = cost.R[i] * (u[i] - cost.uref[i])
        end
    end
    return true
end

function TO.hessian!(E, cost::ShuttlingDiagonalCost, z::RD.AbstractKnotPoint, cache=nothing)
    @inbounds for i in eachindex(cost.Q)
        E.Q[i, i] = cost.Q[i]
    end
    if !RD.is_terminal(z)
        @inbounds for i in eachindex(cost.R)
            E.R[i, i] = cost.R[i]
        end
    end
    return true
end

mutable struct PropagationCache{T}
    V::Vector{T}
    psi::Vector{Complex{T}}
    psi_k::Vector{Complex{T}}
    exp_T::Vector{Complex{T}}
    F::Any
    iF::Any
end

function PropagationCache(::Type{Float64}, N_x::Int)
    psi = zeros(ComplexF64, N_x)
    psi_k = zeros(ComplexF64, N_x)
    return PropagationCache(
        zeros(Float64, N_x),
        psi,
        psi_k,
        zeros(ComplexF64, N_x),
        plan_fft(psi),
        plan_ifft(psi_k),
    )
end

mutable struct DualPropagationCache{T,V,N}
    V_x::Vector{T}
    psi::Vector{Complex{T}}
    psi_k::Vector{Complex{T}}
    exp_T::Vector{Complex{T}}
    psi_components::Matrix{Complex{V}}
    psik_components::Matrix{Complex{V}}
    F::Any
    iF::Any
end

function DualPropagationCache(::Type{T}, N_x::Int) where {Tag,V,N,T<:ForwardDiff.Dual{Tag,V,N}}
    psi_components = zeros(Complex{V}, N_x, N + 1)
    psik_components = zeros(Complex{V}, N_x, N + 1)
    return DualPropagationCache{T,V,N}(
        zeros(T, N_x),
        zeros(Complex{T}, N_x),
        zeros(Complex{T}, N_x),
        zeros(Complex{T}, N_x),
        psi_components,
        psik_components,
        plan_fft(psi_components, (1,)),
        plan_ifft(psik_components, (1,)),
    )
end

struct Model{Tis,Tic,Tid} <: RD.AbstractModel
    n::Int
    m::Int
    N_x::Int
    xs::Vector{Float64}
    dx::Float64
    dt::Float64
    kxs_sq::Vector{Float64}
    c2_base::Float64
    inv_w0sq::Float64
    time_optimal::Bool
    cache::Base.RefValue{Any}
    state1_idx::Tis
    state2_idx::Tis
    controls_idx::Tic
    dcontrols_idx::Tic
    d2controls_idx::Tic
    dt_idx::Tid
end

function Model(xs::Vector{Float64}, dt::Float64; T_well::Float64=1e-3, w0::Float64=500.0,
               time_optimal::Bool=false)
    N_x = length(xs)
    state_count = N_x
    control_count = CONTROL_COUNT
    n = 2 * state_count + 2 * control_count
    m = time_optimal ? control_count + 1 : control_count

    state1_idx = collect(1:N_x)
    state2_idx = collect((state1_idx[end] + 1):(state1_idx[end] + N_x))
    controls_idx = collect((state2_idx[end] + 1):(state2_idx[end] + control_count))
    dcontrols_idx = collect((controls_idx[end] + 1):(controls_idx[end] + control_count))
    d2controls_idx = collect(1:control_count)
    dt_idx = collect((d2controls_idx[end] + 1):(d2controls_idx[end] + 1))

    dx = xs[2] - xs[1]
    kxs = 2π .* Vector{Float64}(FFTW.fftfreq(N_x, 1 / dx))
    kxs_sq = kxs .^ 2

    Tis = typeof(state1_idx)
    Tic = typeof(controls_idx)
    Tid = typeof(dt_idx)

    return Model{Tis,Tic,Tid}(
        n, m, N_x, xs, dx, dt, kxs_sq, KB * T_well, 2.0 / w0^2, time_optimal, Ref{Any}(nothing),
        state1_idx, state2_idx, controls_idx, dcontrols_idx, d2controls_idx, dt_idx,
    )
end

@inline Base.size(model::Model) = model.n, model.m
RD.state_dim(model::Model) = model.n
RD.control_dim(model::Model) = model.m
RD.diffmethod(::Model) = RD.ForwardAD()

@inline center_idx(model::Model) = model.controls_idx[1]
@inline amplitude_idx(model::Model) = model.controls_idx[2]
@inline dcenter_idx(model::Model) = model.dcontrols_idx[1]
@inline damplitude_idx(model::Model) = model.dcontrols_idx[2]
@inline d2center_idx(model::Model) = model.d2controls_idx[1]
@inline d2amplitude_idx(model::Model) = model.d2controls_idx[2]
@inline dt_idx(model::Model) = model.dt_idx[1]

@inline function unpack_state!(psi::AbstractVector{<:Complex}, astate::AbstractVector, state1_idx, state2_idx)
    @inbounds for i in eachindex(state1_idx)
        psi[i] = Complex(astate[state1_idx[i]], astate[state2_idx[i]])
    end
    return psi
end

@inline function pack_state!(astate::AbstractVector, psi::AbstractVector{<:Complex}, state1_idx, state2_idx)
    @inbounds for i in eachindex(state1_idx)
        astate[state1_idx[i]] = real(psi[i])
        astate[state2_idx[i]] = imag(psi[i])
    end
    return astate
end

@inline ramp_time_for_peak_velocity(acc::Float64, v_peak::Float64) = v_peak / acc

function get_propagation_cache!(model::Model, ::Type{T}) where T
    cache = model.cache[]
    if cache isa PropagationCache{T} || cache isa DualPropagationCache{T}
        return cache
    end
    if T === Float64
        cache = PropagationCache(Float64, model.N_x)
    elseif T <: ForwardDiff.Dual
        cache = DualPropagationCache(T, model.N_x)
    else
        throw(ArgumentError("split-step propagation only supports Float64 and ForwardDiff.Dual element types; got $(T)"))
    end
    model.cache[] = cache
    return cache
end

@inline function half_potential_step!(psi::AbstractVector{<:Complex}, V::AbstractVector, dt_local)
    vfac = Complex(zero(eltype(V)), -dt_local / (2 * HBAR))
    @inbounds @simd for i in eachindex(psi, V)
        psi[i] *= Base.exp(vfac * V[i])
    end
    return psi
end

@inline function kinetic_phase!(exp_T::AbstractVector{<:Complex}, model::Model, dt_local)
    T = typeof(real(exp_T[1]))
    tfac = Complex(zero(T), -HBAR_SQUARED_BY_2M_RB87 * dt_local / HBAR)
    @inbounds @simd for i in eachindex(exp_T, model.kxs_sq)
        exp_T[i] = Base.exp(tfac * model.kxs_sq[i])
    end
    return exp_T
end

# One split-step step matching the Strang splitting in pejl/src/quantum.jl:split_step_1d,
# which is the evolution system used by shuttling-well-sim.jl.
function split_step_step!(psi::Vector{ComplexF64}, psi_k::Vector{ComplexF64}, exp_T::Vector{ComplexF64},
                          V::Vector{Float64}, cache::PropagationCache{Float64},
                          model::Model, center, amplitude, dt_local)
    potential_profile!(V, model.xs, center, amplitude, model.c2_base, model.inv_w0sq)
    half_potential_step!(psi, V, dt_local)
    kinetic_phase!(exp_T, model, dt_local)
    mul!(psi_k, cache.F, psi)
    @inbounds @simd for i in eachindex(psi_k, exp_T)
        psi_k[i] *= exp_T[i]
    end
    mul!(psi, cache.iF, psi_k)
    half_potential_step!(psi, V, dt_local)
    return psi
end

@inline function unpack_dual_components!(components::AbstractMatrix{Complex{V}},
                                         psi::AbstractVector{Complex{T}}) where {Tag,V,N,T<:ForwardDiff.Dual{Tag,V,N}}
    @inbounds for i in eachindex(psi)
        psi_re = real(psi[i])
        psi_im = imag(psi[i])
        components[i, 1] = Complex(ForwardDiff.value(psi_re), ForwardDiff.value(psi_im))
        for j = 1:N
            components[i, j + 1] = Complex(ForwardDiff.partials(psi_re, j), ForwardDiff.partials(psi_im, j))
        end
    end
    return components
end

@inline function pack_dual_components!(psi::AbstractVector{Complex{T}},
                                       components::AbstractMatrix{Complex{V}}) where {Tag,V,N,T<:ForwardDiff.Dual{Tag,V,N}}
    @inbounds for i in eachindex(psi)
        re_dual = T(real(components[i, 1]), ForwardDiff.Partials(ntuple(j -> real(components[i, j + 1]), Val(N))))
        im_dual = T(imag(components[i, 1]), ForwardDiff.Partials(ntuple(j -> imag(components[i, j + 1]), Val(N))))
        psi[i] = Complex(re_dual, im_dual)
    end
    return psi
end

function fft_dual!(psi_k::AbstractVector{Complex{T}}, psi::AbstractVector{Complex{T}},
                   cache::DualPropagationCache{T,V,N}) where {Tag,V,N,T<:ForwardDiff.Dual{Tag,V,N}}
    unpack_dual_components!(cache.psi_components, psi)
    mul!(cache.psik_components, cache.F, cache.psi_components)
    pack_dual_components!(psi_k, cache.psik_components)
    return psi_k
end

function ifft_dual!(psi::AbstractVector{Complex{T}}, psi_k::AbstractVector{Complex{T}},
                    cache::DualPropagationCache{T,V,N}) where {Tag,V,N,T<:ForwardDiff.Dual{Tag,V,N}}
    unpack_dual_components!(cache.psik_components, psi_k)
    mul!(cache.psi_components, cache.iF, cache.psik_components)
    pack_dual_components!(psi, cache.psi_components)
    return psi
end

function split_step_step!(psi::AbstractVector{Complex{T}}, psi_k::AbstractVector{Complex{T}},
                          exp_T::AbstractVector{Complex{T}}, V::AbstractVector{T},
                          cache::DualPropagationCache{T,VBase,N},
                          model::Model, center, amplitude, dt_local) where {Tag,VBase,N,T<:ForwardDiff.Dual{Tag,VBase,N}}
    potential_profile!(V, model.xs, center, amplitude, model.c2_base, model.inv_w0sq)
    half_potential_step!(psi, V, dt_local)
    kinetic_phase!(exp_T, model, dt_local)
    fft_dual!(psi_k, psi, cache)
    @inbounds @simd for i in eachindex(psi_k, exp_T)
        psi_k[i] *= exp_T[i]
    end
    ifft_dual!(psi, psi_k, cache)
    half_potential_step!(psi, V, dt_local)
    return psi
end

function propagate_dynamics!(astate_next::AbstractVector, model::Model,
                             astate::AbstractVector, acontrol::AbstractVector, dt::Real)
    dt_local = model.time_optimal ? acontrol[dt_idx(model)]^2 : dt
    T = promote_type(eltype(astate_next), eltype(astate), eltype(acontrol), typeof(dt_local))
    center = astate[center_idx(model)]
    dcenter = astate[dcenter_idx(model)]
    amplitude = astate[amplitude_idx(model)]
    damplitude = astate[damplitude_idx(model)]
    d2center = acontrol[d2center_idx(model)]
    d2amplitude = acontrol[d2amplitude_idx(model)]

    if T === Float64
        cache = get_propagation_cache!(model, Float64)
        psi = unpack_state!(cache.psi, astate, model.state1_idx, model.state2_idx)
        split_step_step!(psi, cache.psi_k, cache.exp_T, cache.V, cache, model, center, amplitude, dt_local)
        pack_state!(astate_next, psi, model.state1_idx, model.state2_idx)
    else
        cache = get_propagation_cache!(model, T)
        psi = unpack_state!(cache.psi, astate, model.state1_idx, model.state2_idx)
        split_step_step!(psi, cache.psi_k, cache.exp_T, cache.V_x, cache, model, center, amplitude, dt_local)
        pack_state!(astate_next, psi, model.state1_idx, model.state2_idx)
    end

    center_next = center + dt_local * dcenter + 0.5 * dt_local^2 * d2center
    dcenter_next = dcenter + dt_local * d2center
    amplitude_next = amplitude + dt_local * damplitude + 0.5 * dt_local^2 * d2amplitude
    damplitude_next = damplitude + dt_local * d2amplitude

    astate_next[center_idx(model)] = center_next
    astate_next[dcenter_idx(model)] = dcenter_next
    astate_next[amplitude_idx(model)] = amplitude_next
    astate_next[damplitude_idx(model)] = damplitude_next
    return astate_next
end

abstract type EXP <: RD.Explicit end

function RD.discrete_dynamics(::Type{EXP}, model::Model,
                              astate::AbstractVector,
                              acontrol::AbstractVector, time::Real, dt::Real)
    T = promote_type(eltype(astate), eltype(acontrol), typeof(dt))
    astate_next = zeros(T, model.n)
    return propagate_dynamics!(astate_next, model, astate, acontrol, dt)
end

function RD.discrete_dynamics!(astate_next::AbstractVector, ::Type{EXP}, model::Model,
                               astate::AbstractVector, acontrol::AbstractVector,
                               time::Real, dt::Real)
    propagate_dynamics!(astate_next, model, astate, acontrol, dt)
    return nothing
end

function RD._discrete_jacobian!(::RD.ForwardAD, ::Type{Q}, grad, model::Model,
                                z::RD.AbstractKnotPoint{T,N,M}) where {T,N,M,Q<:RD.Explicit}
    ix, iu, dt = z._x, z._u, z.dt
    t = z.t
    fd_aug(s) = RD.discrete_dynamics(Q, model, s[ix], s[iu], t, dt)
    grad .= ForwardDiff.jacobian(fd_aug, z.z)
    return nothing
end

function RD._discrete_jacobian!(::RD.ForwardAD, ::Type{Q}, grad, model::Model,
                                z::RD.AbstractKnotPoint{T,N,M}, cache) where {T,N,M,Q<:RD.Explicit}
    ix, iu, dt = z._x, z._u, z.dt
    t = z.t
    fd_aug(s) = RD.discrete_dynamics(Q, model, s[ix], s[iu], t, dt)
    grad .= ForwardDiff.jacobian(fd_aug, z.z)
    return nothing
end

struct ShuttlingDynamicsConstraint{Q<:RD.Explicit,L<:RD.AbstractModel} <: TO.AbstractDynamicsConstraint
    model::L
end

TO.integration(::ShuttlingDynamicsConstraint{Q}) where {Q} = Q
TO.widths(con::ShuttlingDynamicsConstraint, n::Int=RD.state_dim(con.model),
          m::Int=RD.control_dim(con.model)) = (n + m, n + m)
TO.widths(con::ShuttlingDynamicsConstraint{<:RD.Explicit}, n::Int=RD.state_dim(con.model),
          m::Int=RD.control_dim(con.model)) = (n + m, n)
TO.get_inds(::ShuttlingDynamicsConstraint{<:RD.Explicit}, n, m) = (1:n + m, (n + m) .+ (1:n))

function TO.evaluate(con::ShuttlingDynamicsConstraint{Q}, z1::RD.AbstractKnotPoint,
                     z2::RD.AbstractKnotPoint) where {Q<:RD.Explicit}
    return RD.discrete_dynamics(Q, con.model, z1) - RD.state(z2)
end

function TO.jacobian!(grad, con::ShuttlingDynamicsConstraint{Q},
                      z1::RD.AbstractKnotPoint, z2::RD.AbstractKnotPoint{<:Any,n},
                      i::Int=1) where {Q<:RD.Explicit,n}
    if i == 1
        RD.discrete_jacobian!(Q, grad, con.model, z1, nothing)
        return false
    elseif i == 2
        fill!(grad, zero(eltype(grad)))
        for j = 1:n
            grad[j, j] = -1
        end
        return true
    end
    return false
end

function TO.add_dynamics_constraints!(prob::TO.Problem{EXP}, integration::Type{EXP}=EXP, idx::Int=-1)
    if !(prob.model isa Model)
        n, m = size(prob)
        con_set = prob.constraints
        dyn_con = TO.DynamicsConstraint{integration}(prob.model, prob.N)
        TO.add_constraint!(con_set, dyn_con, 1:prob.N-1, idx)
        SVector = getfield(TO, :SVector)
        init_con = TO.GoalConstraint(n, prob.x0, SVector{n}(1:n))
        TO.add_constraint!(con_set, init_con, 1, 1)
        return nothing
    end

    n, _ = size(prob)
    con_set = prob.constraints
    dyn_con = ShuttlingDynamicsConstraint{integration,typeof(prob.model)}(prob.model)
    TO.add_constraint!(con_set, dyn_con, 1:prob.N-1, idx)
    SVector = getfield(TO, :SVector)
    init_con = TO.GoalConstraint(n, prob.x0, SVector{n}(1:n))
    TO.add_constraint!(con_set, init_con, 1, 1)
    return nothing
end

function Altro.copy_jacobian!(D,
                              con::TO.AbstractConstraintValues{<:ShuttlingDynamicsConstraint{<:RD.Explicit}},
                              cinds, xinds, uinds)
    for (i, k) in enumerate(con.inds)
        zind1 = [xinds[k]; uinds[k]]
        D[cinds[i], zind1] .= con.jac[i, 1]
        D[cinds[i], xinds[k + 1]] .= con.jac[i, 2][:, xinds[1]]
    end
end

struct AccelerationGradientConstraint <: TO.StageConstraint
    n::Int
    m::Int
    amplitude_idx::Int
    d2center_idx::Int
    mass::Float64
    max_gradient_coeff::Float64
end

Base.copy(con::AccelerationGradientConstraint) = AccelerationGradientConstraint(
    con.n, con.m, con.amplitude_idx, con.d2center_idx, con.mass, con.max_gradient_coeff,
)

RD.state_dim(con::AccelerationGradientConstraint) = con.n
RD.control_dim(con::AccelerationGradientConstraint) = con.m
Base.length(::AccelerationGradientConstraint) = 2
TO.sense(::AccelerationGradientConstraint) = TO.Inequality()

function TO.evaluate(con::AccelerationGradientConstraint, x::AbstractVector, u::AbstractVector)
    amplitude = x[con.amplitude_idx]
    d2center = u[con.d2center_idx]
    limit = con.max_gradient_coeff * amplitude
    return [
        con.mass * d2center - limit,
        -con.mass * d2center - limit,
    ]
end

function TO.jacobian!(J, con::AccelerationGradientConstraint, x::AbstractVector, u::AbstractVector)
    fill!(J, zero(eltype(J)))
    J[1, con.amplitude_idx] = -con.max_gradient_coeff
    J[2, con.amplitude_idx] = -con.max_gradient_coeff
    J[1, con.n + con.d2center_idx] = con.mass
    J[2, con.n + con.d2center_idx] = -con.mass
    return true
end

# GPU support (included after Model and dynamics definitions)
include(joinpath(@__DIR__, "shuttling_gpu.jl"))

function run_traj(; T_well=1e-3, w0=500.0, N_x=4096, N_t=8000,
                  T_hold=20.0,
                  transport_time=20_000.0,
                  x_target=X_TARGET_NM, amp_target=AMP_TARGET,
                  q_state=1.0, qf_state=1.0, r_alpha=1e-4, r_beta=1e-4, r_time=1e-4,
                  time_optimal=false,
                  verbose=true, save=true, show_progress=true, benchmark=false,
                  iterations_inner=300, iterations_outer=30, n_steps=2,
                  max_iterations=10_000, bp_reg_fp=10.0, dJ_counter_limit=20,
                  bp_reg_type=:control, projected_newton=false, pn_only=false,
                  return_solver=false, gpu=false)
    evolution_time = 2 * T_hold + transport_time
    dt = evolution_time / N_t
    dt_max = 2 * dt
    dt_min = dt / 1e1
    sqrt_dt_max = sqrt(dt_max)
    sqrt_dt_min = sqrt(dt_min)

    N = N_t + 1
    ts = collect(range(0.0, stop=evolution_time, length=N))

    full_L = 2 * (max(abs(x_target), w0) + 2 * w0)
    dx = full_L / N_x
    xs = gen_axis_centered(N_x; dx=dx)
    model = Model(xs, dt; T_well=T_well, w0=w0, time_optimal=time_optimal)
    n, m = size(model)
    t0 = 0.0

    V_initial = zeros(Float64, model.N_x)
    potential_profile!(V_initial, xs, 0.0, 1.0, model.c2_base, model.inv_w0sq)
    vals_initial, vecs_initial = eigensolve_1d(collect(V_initial), collect(xs))
    psi_initial = normalize!(ComplexF64.(vecs_initial[:, 1]))
    x0 = zeros(Float64, n)
    pack_state!(x0, psi_initial, model.state1_idx, model.state2_idx)
    x0[center_idx(model)] = 0.0
    x0[dcenter_idx(model)] = 0.0
    x0[amplitude_idx(model)] = 1.0
    x0[damplitude_idx(model)] = 0.0

    V_final = zeros(Float64, model.N_x)
    potential_profile!(V_final, xs, x_target, amp_target, model.c2_base, model.inv_w0sq)
    vals_final, vecs_final = eigensolve_1d(collect(V_final), collect(xs))
    psi_target = normalize!(ComplexF64.(vecs_final[:, 1]))
    xf = zeros(Float64, n)
    pack_state!(xf, psi_target, model.state1_idx, model.state2_idx)
    xf[center_idx(model)] = x_target
    xf[dcenter_idx(model)] = 0.0
    xf[amplitude_idx(model)] = amp_target
    xf[damplitude_idx(model)] = 0.0

    X0 = [zeros(n) for k = 1:N]
    X0[1] .= x0
    U0 = [zeros(m) for k = 1:N-1]
    t_start = T_hold
    t_stop = ts[end] - T_hold
    for k = 1:N-1
        _, _, alpha = min_jerk_profile(ts[k], t_start, t_stop, x_target)
        U0[k][d2center_idx(model)] = alpha
        U0[k][d2amplitude_idx(model)] = 0.0
        if time_optimal
            U0[k][dt_idx(model)] = sqrt(dt)
        end
        X0[k + 1] .= Altro.discrete_dynamics(EXP, model, X0[k], U0[k], ts[k], dt)
    end

    Q = zeros(Float64, n)
    Q[model.state1_idx] .= q_state
    Q[model.state2_idx] .= q_state

    Qf = zeros(Float64, n)
    Qf[model.state1_idx] .= qf_state
    Qf[model.state2_idx] .= qf_state
    Qf[center_idx(model)] = qf_state
    Qf[dcenter_idx(model)] = qf_state
    Qf[amplitude_idx(model)] = qf_state
    Qf[damplitude_idx(model)] = qf_state

    R = zeros(Float64, m)
    R[d2center_idx(model)] = r_alpha
    R[d2amplitude_idx(model)] = r_beta
    if time_optimal
        R[dt_idx(model)] = r_time
    end
    uref = zeros(Float64, m)
    stage_cost = ShuttlingDiagonalCost(Q, R, xf, uref, false)
    terminal_cost = ShuttlingDiagonalCost(Qf, R, xf, uref, true)
    objective = TO.Objective(stage_cost, terminal_cost, N)

    mass_rb87 = mass_from_alpha(HBAR_SQUARED_BY_2M_RB87)
    max_gradient_coeff = 2 * model.c2_base * Base.exp(-0.5) / w0
    accel_gradient = AccelerationGradientConstraint(1, 1, 1, 1, mass_rb87, max_gradient_coeff)
    accel_gradient = TO.IndexedConstraint(
        n, m, accel_gradient,
        amplitude_idx(model):amplitude_idx(model),
        d2center_idx(model):d2center_idx(model),
    )

    boundary_goal_idxs = [center_idx(model), dcenter_idx(model), amplitude_idx(model), damplitude_idx(model)]
    boundary_goal = TO.GoalConstraint(xf, boundary_goal_idxs)

    constraints = TO.ConstraintList(n, m, N)
    state_bound_idxs = [center_idx(model), amplitude_idx(model)]
    state_upper = TO.LinearConstraint(
        n, m, Matrix{Float64}(I, 2, 2), [x_target, 1.0], TO.Inequality(), state_bound_idxs,
    )
    state_lower = TO.LinearConstraint(
        n, m, -Matrix{Float64}(I, 2, 2), [0.0, 0.0], TO.Inequality(), state_bound_idxs,
    )
    TO.add_constraint!(constraints, state_upper, 1:N)
    TO.add_constraint!(constraints, state_lower, 1:N)
    if time_optimal
        dt_full_idx = [n + dt_idx(model)]
        dt_upper = TO.LinearConstraint(
            n, m, reshape([1.0], 1, 1), [sqrt_dt_max], TO.Inequality(), dt_full_idx,
        )
        dt_lower = TO.LinearConstraint(
            n, m, reshape([-1.0], 1, 1), [-sqrt_dt_min], TO.Inequality(), dt_full_idx,
        )
        TO.add_constraint!(constraints, dt_upper, 1:N-1)
        TO.add_constraint!(constraints, dt_lower, 1:N-1)
    end
    TO.add_constraint!(constraints, accel_gradient, 1:N-1)
    TO.add_constraint!(constraints, boundary_goal, N:N)

    prob = TO.Problem(model, objective, xf, evolution_time;
                      constraints=constraints, t0=t0, x0=x0, N=N, X0=X0, U0=U0,
                      dt=dt, integration=EXP)

    verbose_pn = verbose ? true : false
    verbose_ = verbose ? 2 : 0
    opts = Altro.SolverOptions(
        verbose_pn=verbose_pn, verbose=verbose_,
        iterations_inner=iterations_inner, iterations_outer=iterations_outer,
        n_steps=n_steps, iterations=max_iterations,
        bp_reg_fp=bp_reg_fp, dJ_counter_limit=dJ_counter_limit, bp_reg_type=bp_reg_type,
        projected_newton=projected_newton, memory_efficient=true,
        gpu=gpu,
    )

    # Create GPU caches if gpu=true
    if gpu
        gpu_jac_cache = GpuJacobianCache(model)
        gpu_bp_ws = GpuBackwardPassWorkspace(Float64, n, m)
        opts.gpu_cache = GpuSolverCache(gpu_jac_cache, gpu_bp_ws,
                                         backwardpass_memory_efficient_gpu!)
        if verbose
            println("GPU mode enabled — using decomposed ForwardDiff on CUDA")
        end
    end

    if verbose
        println(@sprintf("shuttling mem-eff problem: N_x=%d, N_t=%d, n=%d, m=%d, dt=%.6f ns",
                         N_x, N_t, n, m, dt))
        println(@sprintf("initial energy = %.6e, target energy = %.6e", vals_initial[1], vals_final[1]))
    end

    solver = if pn_only
        Altro.MemEffProjectedNewtonSolver(prob, opts)
    elseif projected_newton
        ALTROSolver(prob, opts)
    else
        Altro.AugmentedLagrangianSolver(prob, opts)
    end
    if benchmark
        benchmark_result = Altro.benchmark_solve!(solver)
    else
        benchmark_result = nothing
        Altro.solve!(solver)
    end

    if verbose
        println("status: $(solver.stats.status)")
    end

    if benchmark
        return benchmark_result
    end

    acontrols_raw = Altro.controls(solver)
    acontrols_arr = permutedims(reduce(hcat, map(Array, acontrols_raw)), [2, 1])
    astates_raw = TO.states(solver)
    astates_arr = permutedims(reduce(hcat, map(Array, astates_raw)), [2, 1])
    state1_idx_arr = Array(model.state1_idx)
    state2_idx_arr = Array(model.state2_idx)
    controls_idx_arr = Array(model.controls_idx)
    dcontrols_idx_arr = Array(model.dcontrols_idx)
    d2controls_idx_arr = Array(model.d2controls_idx)
    dt_idx_arr = Array(model.dt_idx)
    max_v = TO.max_violation(solver)
    max_v_info = TO.findmax_violation(TO.get_constraints(solver))
    iterations_ = Altro.iterations(solver)
    stats_iterations = collect(solver.stats.iteration[1:iterations_])
    stats_cost = collect(solver.stats.cost[1:iterations_])
    if time_optimal
        ts = vcat(0.0, cumsum(acontrols_arr[:, dt_idx(model)] .^ 2))
    end

    result = Dict(
        "acontrols" => acontrols_arr,
        "astates" => astates_arr,
        "dt" => dt,
        "ts" => ts,
        "xs" => xs,
        "dx" => dx,
        "evolution_time" => evolution_time,
        "transport_time" => transport_time,
        "T_hold" => T_hold,
        "x_target" => x_target,
        "amp_target" => amp_target,
        "N_x" => N_x,
        "N_t" => N_t,
        "T_well" => T_well,
        "w0" => w0,
        "q_state" => q_state,
        "qf_state" => qf_state,
        "r_alpha" => r_alpha,
        "r_beta" => r_beta,
        "r_time" => r_time,
        "state1_idx" => state1_idx_arr,
        "state2_idx" => state2_idx_arr,
        "controls_idx" => controls_idx_arr,
        "dcontrols_idx" => dcontrols_idx_arr,
        "d2controls_idx" => d2controls_idx_arr,
        "dt_idx" => dt_idx_arr,
        "time_optimal" => Integer(time_optimal),
        "memory_efficient" => 1,
        "target_state" => xf,
        "initial_state" => x0,
        "target_energy" => vals_final[1],
        "initial_energy" => vals_initial[1],
        "max_v" => max_v,
        "max_v_info" => max_v_info,
        "iterations" => iterations_,
        "stats_iteration" => stats_iterations,
        "stats_cost" => stats_cost,
        "solver_status" => string(solver.stats.status),
        "save_type" => Int(jl),
    )

    if show_progress
        optimization_plot_file_path = generate_file_path("png", "$(EXPERIMENT_NAME)_loss", SAVE_PATH)
        plot_optimization_loss(stats_iterations, stats_cost, optimization_plot_file_path)
        println("Optimization progress plot saved to $(optimization_plot_file_path)")
        result["optimization_plot_file_path"] = optimization_plot_file_path
    end

    if save
        save_file_path = generate_file_path("h5", EXPERIMENT_NAME, SAVE_PATH)
        println("Saving this optimization to $(save_file_path)")
        h5open(save_file_path, "cw") do save_file
            for key in keys(result)
                write(save_file, key, result[key])
            end
        end
        result["save_file_path"] = save_file_path
    end

    return return_solver ? (result=result, solver=solver) : result
end


function simulate_traj(; T_well=1e-3, w0=500.0, N_x=4096, N_t=8000,
                       T_hold=2000.0,
                       profile=:trapezoid,
                       transport_time=20_000.0,
                       x_target=X_TARGET_NM, amp_target=AMP_TARGET,
                       acc=1.3333e-5, t1=5000.0, t2=0.0, v_peak=nothing,
                       a3=1.0, a4=1.0,
                       time_optimal=false)
    time_optimal && throw(ArgumentError("simulate_traj only supports fixed dt motion profiles"))

    local T_transit::Float64
    local evolution_time::Float64
    local x_target_local::Float64
    local amp_target_local::Float64
    local ts::Vector{Float64}
    local xc_ts::Vector{Float64}
    local vc_ts::Vector{Float64}
    local amp_ts::Vector{Float64}
    local alpha_ts::Vector{Float64}

    if profile == :trapezoid
        if !isnothing(v_peak)
            t1 = ramp_time_for_peak_velocity(Float64(acc), Float64(v_peak))
        end
        T_transit = 2 * t1 + t2
        x_target_local = acc * t1 * (t1 + t2)
        evolution_time = 2 * T_hold + T_transit
        dt = evolution_time / N_t
        ts = collect(dt .* (0:N_t))

        v_max = acc * t1
        function vc_of_t(t)
            tau = t - T_hold
            if tau <= 0
                return 0.0
            elseif tau < t1
                return acc * tau
            elseif tau < t1 + t2
                return v_max
            elseif tau < T_transit
                tau3 = tau - t1 - t2
                return v_max - acc * tau3
            else
                return 0.0
            end
        end
        function xc_of_t(t)
            tau = t - T_hold
            if tau <= 0
                return 0.0
            elseif tau < t1
                return 0.5 * acc * tau^2
            elseif tau < t1 + t2
                tau2 = tau - t1
                return 0.5 * acc * t1^2 + v_max * tau2
            elseif tau < T_transit
                tau3 = tau - t1 - t2
                return 0.5 * acc * t1^2 + v_max * t2 + v_max * tau3 - 0.5 * acc * tau3^2
            else
                return x_target_local
            end
        end
        xc_ts = xc_of_t.(ts)
        vc_ts = vc_of_t.(ts)
        alpha_ts = similar(ts)
        @inbounds for k in eachindex(ts)
            tau = ts[k] - T_hold
            if tau <= 0 || tau >= T_transit
                alpha_ts[k] = 0.0
            elseif tau < t1
                alpha_ts[k] = acc
            elseif tau < t1 + t2
                alpha_ts[k] = 0.0
            else
                alpha_ts[k] = -acc
            end
        end
        amp_target_local = a4
        amp_ts = similar(ts)
        @inbounds for k in eachindex(ts)
            s = x_target_local == 0 ? 0.0 : clamp(xc_ts[k] / x_target_local, 0.0, 1.0)
            amp_ts[k] = a3 + (a4 - a3) * s
        end
    elseif profile == :min_jerk
        x_target_local = x_target
        amp_target_local = amp_target
        evolution_time = 2 * T_hold + transport_time
        dt = evolution_time / N_t
        ts = collect(range(0.0, stop=evolution_time, length=N_t + 1))
        T_transit = transport_time
        t_start = T_hold
        t_stop = ts[end] - T_hold
        xc_ts = zeros(length(ts))
        vc_ts = zeros(length(ts))
        alpha_ts = zeros(length(ts))
        @inbounds for k in eachindex(ts)
            xc_ts[k], vc_ts[k], alpha_ts[k] = min_jerk_profile(ts[k], t_start, t_stop, x_target_local)
        end
        amp_ts = fill(amp_target_local, length(ts))
    else
        throw(ArgumentError("unsupported profile $(profile); use :trapezoid or :min_jerk"))
    end

    N = N_t + 1

    full_L = 2 * (max(abs(x_target_local), w0) + 2 * w0)
    dx = full_L / N_x
    xs = gen_axis_centered(N_x; dx=dx)
    model = Model(xs, dt; T_well=T_well, w0=w0, time_optimal=false)
    n, m = size(model)

    V_initial = zeros(Float64, model.N_x)
    potential_profile!(V_initial, xs, 0.0, amp_ts[1], model.c2_base, model.inv_w0sq)
    vals_initial, vecs_initial = eigensolve_1d(collect(V_initial), collect(xs))
    psi_initial = normalize!(ComplexF64.(vecs_initial[:, 1]))
    x0 = zeros(Float64, n)
    pack_state!(x0, psi_initial, model.state1_idx, model.state2_idx)
    x0[center_idx(model)] = 0.0
    x0[dcenter_idx(model)] = 0.0
    x0[amplitude_idx(model)] = amp_ts[1]
    x0[damplitude_idx(model)] = 0.0

    V_final = zeros(Float64, model.N_x)
    potential_profile!(V_final, xs, x_target_local, amp_target_local, model.c2_base, model.inv_w0sq)
    vals_final, vecs_final = eigensolve_1d(collect(V_final), collect(xs))
    psi_target = normalize!(ComplexF64.(vecs_final[:, 1]))
    xf = zeros(Float64, n)
    pack_state!(xf, psi_target, model.state1_idx, model.state2_idx)
    xf[center_idx(model)] = x_target_local
    xf[dcenter_idx(model)] = 0.0
    xf[amplitude_idx(model)] = amp_target_local

    astates = zeros(Float64, N, n)
    acontrols = zeros(Float64, N - 1, m)
    astates[1, :] .= x0

    @inbounds for k = 1:N
        astates[k, center_idx(model)] = xc_ts[k]
        astates[k, dcenter_idx(model)] = vc_ts[k]
        astates[k, amplitude_idx(model)] = amp_ts[k]
        astates[k, damplitude_idx(model)] = 0.0
    end
    for k = 1:N-1
        acontrols[k, d2center_idx(model)] = alpha_ts[k]
        acontrols[k, d2amplitude_idx(model)] = 0.0
        psi_next = RD.discrete_dynamics(EXP, model, view(astates, k, :), view(acontrols, k, :), ts[k], dt)
        astates[k + 1, model.state1_idx] .= psi_next[model.state1_idx]
        astates[k + 1, model.state2_idx] .= psi_next[model.state2_idx]
    end

    return Dict(
        "acontrols" => acontrols,
        "astates" => astates,
        "dt" => dt,
        "ts" => ts,
        "xs" => xs,
        "dx" => dx,
        "evolution_time" => evolution_time,
        "transport_time" => transport_time,
        "T_transit" => T_transit,
        "T_hold" => T_hold,
        "x_target" => x_target_local,
        "amp_target" => amp_target_local,
        "xc_ts" => xc_ts,
        "vc_ts" => vc_ts,
        "amp_ts" => amp_ts,
        "profile" => String(profile),
        "acc" => acc,
        "t1" => t1,
        "t2" => t2,
        "a3" => a3,
        "a4" => a4,
        "N_x" => N_x,
        "N_t" => N_t,
        "T_well" => T_well,
        "w0" => w0,
        "state1_idx" => Array(model.state1_idx),
        "state2_idx" => Array(model.state2_idx),
        "controls_idx" => Array(model.controls_idx),
        "dcontrols_idx" => Array(model.dcontrols_idx),
        "d2controls_idx" => Array(model.d2controls_idx),
        "dt_idx" => Array(model.dt_idx),
        "time_optimal" => 0,
        "memory_efficient" => 1,
        "target_state" => xf,
        "initial_state" => x0,
        "target_energy" => vals_final[1],
        "initial_energy" => vals_initial[1],
        "solver_status" => "simulation_only",
        "save_type" => Int(jl),
    )
end


function gif_from_result(res;
                         fps=12,
                         frame_stride=10,
                         xlims=nothing,
                         density_ylims=nothing,
                         potential_ylims=nothing,
                         file_name=EXPERIMENT_NAME)
    xs = Vector{Float64}(res["xs"])
    ts = Vector{Float64}(res["ts"])
    astates = res["astates"]
    state1_idx = Vector{Int}(res["state1_idx"])
    state2_idx = Vector{Int}(res["state2_idx"])
    controls_idx = Vector{Int}(res["controls_idx"])
    T_well = Float64(res["T_well"])
    w0 = Float64(res["w0"])
    x_target = Float64(res["x_target"])
    amp_target = Float64(res["amp_target"])
    target_state = Vector{Float64}(res["target_state"])

    target_ψ = unpack_state(target_state, state1_idx, state2_idx)
    target_density = abs2.(target_ψ)
    c2_base = KB * T_well
    inv_w0sq = 2.0 / w0^2

    density_ylims = isnothing(density_ylims) ? (0.0, 1.1 * maximum(target_density)) : density_ylims
    if isnothing(potential_ylims)
        V0 = zeros(length(xs))
        potential_profile!(V0, xs, x_target, amp_target, c2_base, inv_w0sq)
        potential_ylims = (1.1 * minimum(V0), 0.1 * abs(minimum(V0)))
    end

    anim = Plots.@animate for k = 1:frame_stride:length(ts)
        ψ = unpack_state(view(astates, k, :), state1_idx, state2_idx)
        density = abs2.(ψ)
        center = astates[k, controls_idx[1]]
        amplitude = astates[k, controls_idx[2]]
        V = zeros(length(xs))
        potential_profile!(V, xs, center, amplitude, c2_base, inv_w0sq)

        p1 = Plots.plot(xs, density, label="|psi|^2", linewidth=2,
                        ylims=density_ylims,
                        xlabel="x (nm)", ylabel="density",
                        title=@sprintf("t = %.1f ns", ts[k]))
        if !isnothing(xlims)
            Plots.xlims!(p1, xlims)
        end
        Plots.plot!(p1, xs, target_density, label="target", linestyle=:dash, linewidth=2)
        Plots.vline!(p1, [center], label="well center", linestyle=:dot)

        p2 = Plots.plot(xs, V, label="V(x)", linewidth=2,
                        ylims=potential_ylims,
                        xlabel="x (nm)", ylabel="potential (rJ)")
        if !isnothing(xlims)
            Plots.xlims!(p2, xlims)
        end
        Plots.vline!(p2, [center], label="center", linestyle=:dot)

        Plots.plot(p1, p2, layout=(2, 1), dpi=DPI, size=(700, 700))
    end

    plot_file_path = generate_file_path("gif", file_name, SAVE_PATH)
    Plots.gif(anim, plot_file_path, fps=fps)
    return plot_file_path
end


function simulate_with_controls(opt_res::Dict{String,<:Any})
    xs = Vector{Float64}(opt_res["xs"])
    dt = Float64(opt_res["dt"])
    T_well = Float64(opt_res["T_well"])
    w0 = Float64(opt_res["w0"])
    time_optimal = Bool(opt_res["time_optimal"])
    acontrols = Array(opt_res["acontrols"])
    initial_state = Vector{Float64}(opt_res["initial_state"])
    target_state = Vector{Float64}(opt_res["target_state"])

    model = Model(xs, dt; T_well=T_well, w0=w0, time_optimal=time_optimal)
    N = size(acontrols, 1) + 1
    astates = zeros(Float64, N, model.n)
    astates[1, :] .= initial_state
    ts = zeros(Float64, N)
    @inbounds for k = 1:N-1
        astates[k + 1, :] .= RD.discrete_dynamics(EXP, model, view(astates, k, :), view(acontrols, k, :), ts[k], dt)
        dt_local = time_optimal ? acontrols[k, dt_idx(model)]^2 : dt
        ts[k + 1] = ts[k] + dt_local
    end

    sim_res = Dict{String,Any}(opt_res)
    sim_res["astates"] = astates
    sim_res["acontrols"] = acontrols
    sim_res["ts"] = ts
    sim_res["initial_state"] = initial_state
    sim_res["target_state"] = target_state
    sim_res["solver_status"] = "simulation_from_optimized_controls"
    return sim_res
end


function to_gif(; fps=12, frame_stride=10, xlims=nothing, density_ylims=nothing,
                potential_ylims=nothing, kwargs...)
    res = run_traj(verbose=false, save=false, show_progress=false; kwargs...)
    return gif_from_result(res;
                           fps=fps,
                           frame_stride=frame_stride,
                           xlims=xlims,
                           density_ylims=density_ylims,
                           potential_ylims=potential_ylims,
                           file_name=EXPERIMENT_NAME)
end


function simulate_to_gif(; fps=12, frame_stride=10, xlims=nothing, density_ylims=nothing,
                         potential_ylims=nothing, kwargs...)
    res = simulate_traj(; kwargs...)
    return gif_from_result(res;
                           fps=fps,
                           frame_stride=frame_stride,
                           xlims=xlims,
                           density_ylims=density_ylims,
                           potential_ylims=potential_ylims,
                           file_name="$(EXPERIMENT_NAME)_sim")
end


function optimize_simulate_to_gif(; optimize=true,
                                  replay_simulation=true,
                                  make_gif=true,
                                  save_optimization=false,
                                  show_progress=false,
                                  verbose=false,
                                  fps=12,
                                  frame_stride=10,
                                  xlims=nothing,
                                  density_ylims=nothing,
                                  potential_ylims=nothing,
                                  gif_name="$(EXPERIMENT_NAME)_opt_sim",
                                  kwargs...)
    optimize || throw(ArgumentError("optimize_simulate_to_gif requires optimize=true"))

    opt_res = run_traj(verbose=verbose,
                       save=save_optimization,
                       show_progress=show_progress;
                       kwargs...)
    sim_res = replay_simulation ? simulate_with_controls(opt_res) : opt_res

    gif_file_path = nothing
    if make_gif
        gif_file_path = gif_from_result(sim_res;
                                        fps=fps,
                                        frame_stride=frame_stride,
                                        xlims=xlims,
                                        density_ylims=density_ylims,
                                        potential_ylims=potential_ylims,
                                        file_name=gif_name)
    end

    return Dict(
        "optimization" => opt_res,
        "simulation" => sim_res,
        "gif_file_path" => gif_file_path,
    )
end
