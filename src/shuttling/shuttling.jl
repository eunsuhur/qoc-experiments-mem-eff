"""
shuttling.jl
"""

WDIR = joinpath(@__DIR__, "../../")
include(joinpath(WDIR, "src", "qocexperiments.jl"))

using LinearAlgebra

const X_TARGET_NM = 8_000.0
const AMP_TARGET = 1.0
const CONTROL_COUNT = 2

const PE_ROOT = "/Users/eunsuhur/Documents/new vault/Projects/Atom arrays/Shuttling/pe"
const PE_SRC = joinpath(PE_ROOT, "pejl", "src")

# Reuse the PE split-step TDSE utilities used by shuttling-well-sim.jl.
include(joinpath(PE_SRC, "quantum.jl"))

const HBAR = hbar
const HBAR_SQUARED_BY_2M_RB87 = hbarsquaredby2mrb87
const KB = kB

@inline function gen_axis_centered(N::Int; dx::Float64=1.0)
    collect(range(-(N - 1) / 2, (N - 1) / 2, step=1.0) .* dx)
end

@inline function mass_from_alpha(alpha::Float64)
    HBAR^2 / (2 * alpha)
end

function min_jerk_profile(t::Float64, t_start::Float64, t_stop::Float64, distance::Float64)
    if t <= t_start
        return 0.0, 0.0, 0.0
    elseif t >= t_stop
        return distance, 0.0, 0.0
    end

    tau = t - t_start
    tf = t_stop - t_start
    s = tau / tf
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s

    pos = distance * (10 * s3 - 15 * s4 + 6 * s5)
    vel = distance / tf * (30 * s2 - 60 * s3 + 30 * s4)
    acc = distance / tf^2 * (60 * s - 180 * s2 + 120 * s3)
    return pos, vel, acc
end

function potential_profile!(V::AbstractVector, xs::AbstractVector, center, amplitude, c2_base, inv_w0sq)
    coeff = -c2_base * amplitude
    @inbounds @simd for i in eachindex(xs)
        V[i] = coeff * Base.exp(-inv_w0sq * (xs[i] - center)^2)
    end
    return V
end

@inline wavefunction_idx(state1_idx, state2_idx) = state1_idx[1]:state2_idx[end]

function pack_state!(astate::AbstractVector, psi::AbstractVector, state1_idx, state2_idx)
    astate[wavefunction_idx(state1_idx, state2_idx)] .= get_vec_iso(psi)
    return astate
end

function unpack_state(astate::AbstractVector, state1_idx, state2_idx)
    get_vec_uniso(astate[wavefunction_idx(state1_idx, state2_idx)])
end

function plot_optimization_loss(iterations::AbstractVector{<:Real},
                                costs::AbstractVector{<:Real},
                                plot_file_path::AbstractString)
    cost_yscale = all(c -> c > 0, costs) ? :log10 : :identity

    fig = Plots.plot(iterations, costs,
                     linewidth=2, marker=:circle, markersize=3,
                     xlabel="Iteration", ylabel="Loss",
                     title="Optimization Loss", label="loss",
                     yscale=cost_yscale, size=(800, 500), dpi=DPI)
    Plots.savefig(fig, plot_file_path)
    return plot_file_path
end
