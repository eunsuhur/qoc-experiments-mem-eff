"""
shuttling_gpu.jl — GPU support for shuttling dynamics and Jacobian computation.

Provides GPU-accelerated split-step FFT propagation and machine-precision Jacobian
computation via decomposed ForwardDiff (manual chain-rule tracking on CuArrays).

Requires: CUDA.jl
"""

using CUDA
using CUDA.CUFFT
using LinearAlgebra
using StaticArrays: @SVector

# =============================================================================
# GPU Propagation Cache — value-only forward dynamics on GPU
# =============================================================================

mutable struct GpuPropagationCache
    V_val::CuVector{Float64}
    psi_val::CuVector{ComplexF64}
    psik_val::CuVector{ComplexF64}
    exp_T_val::CuVector{ComplexF64}
    xs_gpu::CuVector{Float64}
    kxs_sq_gpu::CuVector{Float64}
    F_single::Any   # CUFFT forward plan (single vector)
    iF_single::Any  # CUFFT inverse plan (single vector)
end

function GpuPropagationCache(model::Model)
    N_x = model.N_x
    psi_val = CUDA.zeros(ComplexF64, N_x)
    psik_val = CUDA.zeros(ComplexF64, N_x)
    GpuPropagationCache(
        CUDA.zeros(Float64, N_x),
        psi_val,
        psik_val,
        CUDA.zeros(ComplexF64, N_x),
        CuVector{Float64}(model.xs),
        CuVector{Float64}(model.kxs_sq),
        plan_fft(psi_val),
        plan_ifft(psik_val),
    )
end

# =============================================================================
# GPU Jacobian Cache — decomposed ForwardDiff on GPU
# =============================================================================

mutable struct GpuJacobianCache
    prop::GpuPropagationCache
    # Partials: N_x × (n+m) complex matrices tracking d(psi)/d(input)
    psi_partials::CuMatrix{ComplexF64}
    psik_partials::CuMatrix{ComplexF64}
    # Temporaries for chain rule
    psi_val_old::CuVector{ComplexF64}
    # Potential derivatives (real-valued, length N_x)
    dV_dcenter::CuVector{Float64}
    dV_damplitude::CuVector{Float64}
    # Output Jacobian on GPU: n × (n+m)
    jac_gpu::CuMatrix{Float64}
    # Batched CUFFT plans for N_x × (n+m) partials matrices
    F_batch::Any
    iF_batch::Any
    # Dimension info
    n::Int
    m::Int
    N_x::Int
    state1_start::Int
    state2_start::Int
    center_col::Int
    amplitude_col::Int
    dcenter_col::Int
    damplitude_col::Int
    d2center_col::Int    # column in z = [astate; acontrol] for d2center control
    d2amplitude_col::Int # column in z for d2amplitude control
end

function GpuJacobianCache(model::Model)
    N_x = model.N_x
    n = model.n
    m = model.m
    nm = n + m

    prop = GpuPropagationCache(model)

    psi_partials = CUDA.zeros(ComplexF64, N_x, nm)
    psik_partials = CUDA.zeros(ComplexF64, N_x, nm)

    F_batch = plan_fft(psi_partials, 1)
    iF_batch = plan_ifft(psik_partials, 1)

    GpuJacobianCache(
        prop,
        psi_partials,
        psik_partials,
        CUDA.zeros(ComplexF64, N_x),          # psi_val_old
        CUDA.zeros(Float64, N_x),             # dV_dcenter
        CUDA.zeros(Float64, N_x),             # dV_damplitude
        CUDA.zeros(Float64, n, nm),            # jac_gpu
        F_batch,
        iF_batch,
        n, m, N_x,
        model.state1_idx[1],                   # state1_start
        model.state2_idx[1],                   # state2_start
        model.controls_idx[1],                 # center_col
        model.controls_idx[2],                 # amplitude_col
        model.dcontrols_idx[1],                # dcenter_col
        model.dcontrols_idx[2],                # damplitude_col
        n + model.d2controls_idx[1],           # d2center_col in augmented z
        n + model.d2controls_idx[2],           # d2amplitude_col in augmented z
    )
end

# =============================================================================
# GPU Backward Pass Workspace
# =============================================================================

struct GpuBackwardPassWorkspace{T}
    # Cost-to-go (alternating pair, indices 1 = curr, 2 = next)
    S_Q::Vector{CuMatrix{T}}     # [2] each n×n
    S_q::Vector{CuVector{T}}     # [2] each n

    # Q-function workspace (on GPU)
    Q_Q::CuMatrix{T}             # n×n
    Q_H::CuMatrix{T}             # m×n
    Q_q::CuVector{T}             # n
    Q_r::CuVector{T}             # m

    # GEMM temporaries
    tmp_nn::CuMatrix{T}          # n×n
    tmp_mn::CuMatrix{T}          # m×n

    # Small CPU buffers for gains computation (m is tiny, typically 2)
    Q_R_cpu::Matrix{T}           # m×m
    Q_H_cpu::Matrix{T}           # m×n
    Q_r_cpu::Vector{T}           # m
    Quu_reg_cpu::Matrix{T}       # m×m
    Qux_reg_cpu::Matrix{T}       # m×n

    # Upload buffers for cost-to-go update
    K_gpu::CuMatrix{T}           # m×n
    d_gpu::CuVector{T}           # m
    tmp_m_gpu::CuVector{T}       # m workspace

    # Pre-allocated GPU buffers for per-iteration cost uploads (avoid repeated allocs)
    cost_q_gpu::CuVector{T}      # n
    cost_Q_diag_gpu::CuVector{T} # n
    cost_r_gpu::CuVector{T}      # m
    Q_R_gpu::CuMatrix{T}         # m×m
    tmp_mn2_gpu::CuMatrix{T}     # m×n (for QRK in ctg)

    # CPU scratch for diagonal extraction (avoid per-iteration array allocation)
    cost_Q_diag_cpu::Vector{T}   # n
end

function GpuBackwardPassWorkspace(::Type{T}, n::Int, m::Int) where T
    GpuBackwardPassWorkspace{T}(
        [CUDA.zeros(T, n, n), CUDA.zeros(T, n, n)],
        [CUDA.zeros(T, n), CUDA.zeros(T, n)],
        CUDA.zeros(T, n, n),
        CUDA.zeros(T, m, n),
        CUDA.zeros(T, n),
        CUDA.zeros(T, m),
        CUDA.zeros(T, n, n),
        CUDA.zeros(T, m, n),
        zeros(T, m, m),
        zeros(T, m, n),
        zeros(T, m),
        zeros(T, m, m),
        zeros(T, m, n),
        CUDA.zeros(T, m, n),
        CUDA.zeros(T, m),
        CUDA.zeros(T, m),
        CUDA.zeros(T, n),        # cost_q_gpu
        CUDA.zeros(T, n),        # cost_Q_diag_gpu
        CUDA.zeros(T, m),        # cost_r_gpu
        CUDA.zeros(T, m, m),     # Q_R_gpu
        CUDA.zeros(T, m, n),     # tmp_mn2_gpu
        zeros(T, n),             # cost_Q_diag_cpu
    )
end

# Wrapper stored in SolverOptions.gpu_cache.
# Includes a callable backward_pass_fn so the Altro module can invoke
# the GPU backward pass without knowing its concrete definition.
struct GpuSolverCache
    jac_cache::GpuJacobianCache
    bp_ws::GpuBackwardPassWorkspace
    backward_pass_fn::Function  # (solver, gpu_cache) -> ΔV
end

# =============================================================================
# CUDA Kernels
# =============================================================================

function _gpu_init_psi_partials_kernel!(psi_partials, N_x, s1_start, s2_start)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N_x
        # dpsi[i]/d(state1[i]) = 1 + 0im  (real part dependence)
        psi_partials[i, s1_start + i - 1] = ComplexF64(1.0, 0.0)
        # dpsi[i]/d(state2[i]) = 0 + 1im  (imaginary part dependence)
        psi_partials[i, s2_start + i - 1] = ComplexF64(0.0, 1.0)
    end
    return nothing
end

function _gpu_add_diagonal_kernel!(A, diag, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        A[i, i] += diag[i]
    end
    return nothing
end

# =============================================================================
# GPU Potential Profile with Derivatives
# =============================================================================

function gpu_potential_profile_with_derivs!(V, dV_dc, dV_da, xs, center, amplitude,
                                            c2_base, inv_w0sq)
    # V[i] = -c2_base * amplitude * exp(-inv_w0sq * (xs[i] - center)^2)
    # dV/damplitude = -c2_base * exp(-inv_w0sq * (xs[i] - center)^2)
    # dV/dcenter = V[i] * 2 * inv_w0sq * (xs[i] - center)
    @. dV_da = -c2_base * exp(-inv_w0sq * (xs - center)^2)
    @. V = amplitude * dV_da
    @. dV_dc = V * (2 * inv_w0sq * (xs - center))
    return nothing
end

# =============================================================================
# GPU Split-Step Operations (value only)
# =============================================================================

function gpu_potential_profile!(V, xs, center, amplitude, c2_base, inv_w0sq)
    coeff = -c2_base * amplitude
    @. V = coeff * exp(-inv_w0sq * (xs - center)^2)
    return nothing
end

function gpu_half_potential_step!(psi, V, dt_local, hbar)
    vfac_i = -dt_local / (2 * hbar)
    @. psi *= exp(Complex(0.0, vfac_i * V))
    return nothing
end

function gpu_kinetic_phase!(exp_T, kxs_sq, dt_local, alpha, hbar)
    tfac_i = -alpha * dt_local / hbar
    @. exp_T = exp(Complex(0.0, tfac_i * kxs_sq))
    return nothing
end

function gpu_split_step_step!(prop::GpuPropagationCache, model::Model,
                              center, amplitude, dt_local)
    gpu_potential_profile!(prop.V_val, prop.xs_gpu, center, amplitude,
                           model.c2_base, model.inv_w0sq)
    gpu_half_potential_step!(prop.psi_val, prop.V_val, dt_local, HBAR)
    gpu_kinetic_phase!(prop.exp_T_val, prop.kxs_sq_gpu, dt_local,
                       HBAR_SQUARED_BY_2M_RB87, HBAR)
    mul!(prop.psik_val, prop.F_single, prop.psi_val)
    prop.psik_val .*= prop.exp_T_val
    mul!(prop.psi_val, prop.iF_single, prop.psik_val)
    gpu_half_potential_step!(prop.psi_val, prop.V_val, dt_local, HBAR)
    return nothing
end

# =============================================================================
# GPU Split-Step with Partials (decomposed ForwardDiff)
# =============================================================================

function gpu_half_potential_step_with_partials!(
    psi_val, psi_partials, psi_val_old, V, dV_dc, dV_da,
    dt_local, hbar, center_col, amplitude_col
)
    vfac = Complex(0.0, -dt_local / (2 * hbar))

    # Save psi_val before modification, then compute exp_V into psi_val_old
    # (reusing psi_val_old buffer — first as snapshot, then as exp_V cache)
    copyto!(psi_val_old, psi_val)

    # Compute exp_V once, reuse for value, partials, and chain rule
    # Store exp(vfac * V) in a temporary (broadcast-fused on GPU)
    # psi_val = psi_val_old * exp_V
    @. psi_val *= exp(vfac * V)

    # Scale ALL partial columns by exp(vfac * V[i])
    # Note: psi_val ./ psi_val_old = exp(vfac*V), but division by zero is dangerous.
    # Instead, just reuse the same broadcast expression — CUDA kernels are fast.
    psi_partials .*= exp.(vfac .* V)

    # Chain rule for center: psi_partials[i, center_col] += psi_val_old[i] * exp(vfac*V[i]) * vfac * dV_dc[i]
    # psi_val_old * exp(vfac*V) = psi_val (after the *= above), so:
    view(psi_partials, :, center_col) .+= psi_val .* vfac .* dV_dc

    # Chain rule for amplitude:
    view(psi_partials, :, amplitude_col) .+= psi_val .* vfac .* dV_da

    return nothing
end

function gpu_kinetic_mult_with_partials!(psik_val, psik_partials, exp_T)
    # Value: psi_k[i] *= exp_T[i]
    @. psik_val *= exp_T
    # Partials: scale all columns (exp_T has no input dependence in non-time-optimal)
    psik_partials .*= exp_T
    return nothing
end

function gpu_fft_partials!(psik_val, psik_partials, psi_val, psi_partials,
                           F_single, F_batch)
    mul!(psik_val, F_single, psi_val)
    mul!(psik_partials, F_batch, psi_partials)
    return nothing
end

function gpu_ifft_partials!(psi_val, psi_partials, psik_val, psik_partials,
                            iF_single, iF_batch)
    mul!(psi_val, iF_single, psik_val)
    mul!(psi_partials, iF_batch, psik_partials)
    return nothing
end

# =============================================================================
# Full GPU Jacobian Computation
# =============================================================================

"""
    gpu_compute_jacobian!(cache::GpuJacobianCache, model::Model, z::AbstractKnotPoint)

Compute the Jacobian of discrete_dynamics w.r.t. [state; control] entirely on GPU
using decomposed ForwardDiff (manual chain-rule tracking on CuArrays).

Result is stored in `cache.jac_gpu` (n × (n+m) CuMatrix{Float64}).
"""
function gpu_compute_jacobian!(cache::GpuJacobianCache, model::Model,
                               z::RD.AbstractKnotPoint)
    astate = RD.state(z)
    acontrol = RD.control(z)
    dt = z.dt
    n, m, N_x = cache.n, cache.m, cache.N_x
    nm = n + m
    prop = cache.prop

    dt_local = model.time_optimal ? acontrol[model.d2controls_idx[end]+1]^2 : dt

    # Extract scalars
    center = astate[model.controls_idx[1]]
    dcenter = astate[model.dcontrols_idx[1]]
    amplitude = astate[model.controls_idx[2]]
    damplitude = astate[model.dcontrols_idx[2]]
    d2center = acontrol[model.d2controls_idx[1]]
    d2amplitude = acontrol[model.d2controls_idx[2]]

    # --- Initialize psi value on GPU ---
    # Unpack state to complex psi on CPU, then upload
    psi_cpu = Vector{ComplexF64}(undef, N_x)
    @inbounds for i in 1:N_x
        psi_cpu[i] = Complex(astate[model.state1_idx[i]], astate[model.state2_idx[i]])
    end
    copyto!(prop.psi_val, psi_cpu)

    # --- Initialize psi_partials on GPU ---
    fill!(cache.psi_partials, zero(ComplexF64))
    threads = min(N_x, 256)
    blocks = cld(N_x, threads)
    @cuda threads=threads blocks=blocks _gpu_init_psi_partials_kernel!(
        cache.psi_partials, N_x, cache.state1_start, cache.state2_start)

    # --- Compute potential with derivatives on GPU ---
    gpu_potential_profile_with_derivs!(
        prop.V_val, cache.dV_dcenter, cache.dV_damplitude,
        prop.xs_gpu, center, amplitude, model.c2_base, model.inv_w0sq)

    # --- First half-potential step ---
    gpu_half_potential_step_with_partials!(
        prop.psi_val, cache.psi_partials, cache.psi_val_old,
        prop.V_val, cache.dV_dcenter, cache.dV_damplitude,
        dt_local, HBAR, cache.center_col, cache.amplitude_col)

    # --- Kinetic phase ---
    gpu_kinetic_phase!(prop.exp_T_val, prop.kxs_sq_gpu, dt_local,
                       HBAR_SQUARED_BY_2M_RB87, HBAR)

    # --- FFT (value + partials) ---
    gpu_fft_partials!(prop.psik_val, cache.psik_partials,
                      prop.psi_val, cache.psi_partials,
                      prop.F_single, cache.F_batch)

    # --- Kinetic multiply (value + partials) ---
    gpu_kinetic_mult_with_partials!(prop.psik_val, cache.psik_partials, prop.exp_T_val)

    # --- IFFT (value + partials) ---
    gpu_ifft_partials!(prop.psi_val, cache.psi_partials,
                       prop.psik_val, cache.psik_partials,
                       prop.iF_single, cache.iF_batch)

    # --- Second half-potential step ---
    gpu_half_potential_step_with_partials!(
        prop.psi_val, cache.psi_partials, cache.psi_val_old,
        prop.V_val, cache.dV_dcenter, cache.dV_damplitude,
        dt_local, HBAR, cache.center_col, cache.amplitude_col)

    # --- Assemble Jacobian on GPU ---
    _gpu_assemble_jacobian!(cache, model, dt_local,
                            center, dcenter, amplitude, damplitude,
                            d2center, d2amplitude)

    return nothing
end

"""
Assemble the full n × (n+m) Jacobian from psi partials and classical update partials.
"""
function _gpu_assemble_jacobian!(cache::GpuJacobianCache, model::Model, dt_local,
                                 center, dcenter, amplitude, damplitude,
                                 d2center, d2amplitude)
    n, m, N_x = cache.n, cache.m, cache.N_x
    nm = n + m
    jac = cache.jac_gpu

    # Zero the Jacobian
    fill!(jac, zero(Float64))

    # Quantum part: rows 1:N_x from real(psi_partials), rows (N_x+1):(2*N_x) from imag
    # jac[state1_idx[i], j] = real(psi_partials[i, j])
    # jac[state2_idx[i], j] = imag(psi_partials[i, j])
    #
    # Since state1_idx = 1:N_x and state2_idx = (N_x+1):(2*N_x):
    #   jac[1:N_x, :] = real.(psi_partials)
    #   jac[(N_x+1):(2*N_x), :] = imag.(psi_partials)
    view(jac, 1:N_x, :) .= real.(cache.psi_partials)
    view(jac, (N_x+1):(2*N_x), :) .= imag.(cache.psi_partials)

    # Classical part: set the sparse entries for the 4 classical rows
    # These are small — compute on CPU and upload
    classical_jac = zeros(Float64, 4, nm)

    ci = model.controls_idx[1]       # center row in astate
    ai = model.controls_idx[2]       # amplitude row
    dci = model.dcontrols_idx[1]     # dcenter row
    dai = model.dcontrols_idx[2]     # damplitude row

    # center_next = center + dt*dcenter + 0.5*dt²*d2center
    classical_jac[1, ci] = 1.0                          # d/d(center)
    classical_jac[1, dci] = dt_local                     # d/d(dcenter)
    classical_jac[1, cache.d2center_col] = 0.5 * dt_local^2  # d/d(d2center)

    # dcenter_next = dcenter + dt*d2center
    classical_jac[2, dci] = 1.0
    classical_jac[2, cache.d2center_col] = dt_local

    # amplitude_next = amplitude + dt*damplitude + 0.5*dt²*d2amplitude
    classical_jac[3, ai] = 1.0
    classical_jac[3, dai] = dt_local
    classical_jac[3, cache.d2amplitude_col] = 0.5 * dt_local^2

    # damplitude_next = damplitude + dt*d2amplitude
    classical_jac[4, dai] = 1.0
    classical_jac[4, cache.d2amplitude_col] = dt_local

    # Upload 4 rows to the appropriate positions
    row_indices = [ci, dci, ai, dai]
    for (local_row, global_row) in enumerate(row_indices)
        copyto!(view(jac, global_row:global_row, :),
                CuMatrix{Float64}(reshape(classical_jac[local_row, :], 1, nm)))
    end

    return nothing
end

# =============================================================================
# GPU value-only dynamics propagation (for forward pass)
# =============================================================================

"""
    gpu_propagate_dynamics!(astate_next, model, astate, acontrol, dt, prop_cache)

GPU-accelerated dynamics propagation (value only, no Jacobian).
"""
function gpu_propagate_dynamics!(astate_next::AbstractVector, model::Model,
                                 astate::AbstractVector, acontrol::AbstractVector,
                                 dt::Real, prop::GpuPropagationCache)
    dt_local = model.time_optimal ? acontrol[model.d2controls_idx[end]+1]^2 : dt
    N_x = model.N_x

    center = astate[model.controls_idx[1]]
    dcenter = astate[model.dcontrols_idx[1]]
    amplitude = astate[model.controls_idx[2]]
    damplitude = astate[model.dcontrols_idx[2]]
    d2center = acontrol[model.d2controls_idx[1]]
    d2amplitude = acontrol[model.d2controls_idx[2]]

    # Upload psi to GPU
    psi_cpu = Vector{ComplexF64}(undef, N_x)
    @inbounds for i in 1:N_x
        psi_cpu[i] = Complex(astate[model.state1_idx[i]], astate[model.state2_idx[i]])
    end
    copyto!(prop.psi_val, psi_cpu)

    # GPU split-step
    gpu_split_step_step!(prop, model, center, amplitude, dt_local)

    # Download psi from GPU
    psi_result = Array(prop.psi_val)
    @inbounds for i in 1:N_x
        astate_next[model.state1_idx[i]] = real(psi_result[i])
        astate_next[model.state2_idx[i]] = imag(psi_result[i])
    end

    # Classical updates (on CPU, trivial)
    astate_next[model.controls_idx[1]] = center + dt_local * dcenter + 0.5 * dt_local^2 * d2center
    astate_next[model.dcontrols_idx[1]] = dcenter + dt_local * d2center
    astate_next[model.controls_idx[2]] = amplitude + dt_local * damplitude + 0.5 * dt_local^2 * d2amplitude
    astate_next[model.dcontrols_idx[2]] = damplitude + dt_local * d2amplitude

    return astate_next
end

# =============================================================================
# GPU Backward Pass Functions (CUBLAS-accelerated)
# =============================================================================

"""
Add a diagonal vector to the diagonal of a GPU matrix: A[i,i] += diag[i]
"""
function gpu_add_diagonal!(A::CuMatrix{T}, diag::CuVector{T}) where T
    n = length(diag)
    threads = min(n, 256)
    blocks = cld(n, threads)
    @cuda threads=threads blocks=blocks _gpu_add_diagonal_kernel!(A, diag, n)
    return nothing
end

"""
GPU Q-function computation using CUBLAS.

Computes the action-value expansion:
  Q.q = fdx' * S_next.q + cost.q
  Q.r = fdu' * S_next.q + cost.r
  Q.Q = fdx' * S_next.Q * fdx + diag(cost.Q)
  Q.R = fdu' * S_next.Q * fdu + diag(cost.R)  (extracted to CPU)
  Q.H = fdu' * S_next.Q * fdx + cost.H        (cost.H = 0 for diagonal cost)
"""
function gpu_calc_Q!(ws::GpuBackwardPassWorkspace, fdx::CuMatrix, fdu::CuMatrix,
                     S_Q_next::CuMatrix, S_q_next::CuVector,
                     cost_r_cpu::Vector, cost_R_diag_cpu::Vector)
    # Q.q = fdx' * S_next.q + cost_q  (cost_q_gpu pre-uploaded by caller)
    mul!(ws.Q_q, fdx', S_q_next)
    ws.Q_q .+= ws.cost_q_gpu

    # Q.r = fdu' * S_next.q + cost_r  (m-dimensional)
    mul!(ws.Q_r, fdu', S_q_next)
    copyto!(ws.cost_r_gpu, cost_r_cpu)
    ws.Q_r .+= ws.cost_r_gpu

    # Q.Q = fdx' * S_next.Q * fdx + diag(cost_Q)
    mul!(ws.tmp_nn, fdx', S_Q_next)       # tmp = fdx' * S_Q  (n×n DGEMM)
    mul!(ws.Q_Q, ws.tmp_nn, fdx)           # Q.Q = tmp * fdx   (n×n DGEMM)
    gpu_add_diagonal!(ws.Q_Q, ws.cost_Q_diag_gpu)

    # Q.H = fdu' * S_next.Q * fdx  (m×n, uses tmp_mn as intermediate)
    mul!(ws.tmp_mn, fdu', S_Q_next)        # tmp = fdu' * S_Q  (m×n DGEMM)
    mul!(ws.Q_H, ws.tmp_mn, fdx)           # Q.H = tmp * fdx   (m×n DGEMM)

    # Q.R on CPU: fdu' * S_next.Q * fdu + diag(cost_R)
    # tmp_mn already = fdu' * S_Q. Compute tmp_mn * fdu → m×m (use pre-allocated Q_R_gpu)
    mul!(ws.Q_R_gpu, ws.tmp_mn, fdu)
    copyto!(ws.Q_R_cpu, Array(ws.Q_R_gpu))
    for i in 1:length(cost_R_diag_cpu)
        ws.Q_R_cpu[i, i] += cost_R_diag_cpu[i]
    end
    # Sync Q_R_gpu with the diagonal-augmented version (needed by gpu_calc_ctg!)
    copyto!(ws.Q_R_gpu, ws.Q_R_cpu)

    # Copy Q.H and Q.r to CPU for gains computation
    copyto!(ws.Q_H_cpu, Array(ws.Q_H))
    copyto!(ws.Q_r_cpu, Array(ws.Q_r))

    return nothing
end

"""
GPU cost-to-go update.

S.q = Q.q + K'*Q.R*d + K'*Q.r + Q.H'*d
S.Q = Q.Q + K'*Q.R*K + K'*Q.H + Q.H'*K   (then symmetrize)
"""
function gpu_calc_ctg!(S_Q::CuMatrix, S_q::CuVector,
                       ws::GpuBackwardPassWorkspace,
                       K_cpu::AbstractMatrix, d_cpu::AbstractVector)
    # Upload K and d to GPU (into pre-allocated buffers)
    copyto!(ws.K_gpu, Matrix(K_cpu))
    copyto!(ws.d_gpu, Vector(d_cpu))

    # --- S.q update ---
    # S.q = Q.q + K'*(Q.R*d) + K'*Q.r + Q.H'*d
    copyto!(S_q, ws.Q_q)
    # Q.R*d on CPU → m vector, upload to tmp_m_gpu
    mul!(ws.tmp_m_gpu, ws.Q_R_gpu, ws.d_gpu)
    # S.q += K' * (Q.R*d)
    mul!(S_q, ws.K_gpu', ws.tmp_m_gpu, 1.0, 1.0)
    # S.q += K' * Q.r
    mul!(S_q, ws.K_gpu', ws.Q_r, 1.0, 1.0)
    # S.q += Q.H' * d
    mul!(S_q, ws.Q_H', ws.d_gpu, 1.0, 1.0)

    # --- S.Q update ---
    # S.Q = Q.Q + K'*Q.R*K + K'*Q.H + Q.H'*K
    copyto!(S_Q, ws.Q_Q)
    # Q.R * K on GPU: (m×m) * (m×n) → m×n, store in tmp_mn2_gpu
    mul!(ws.tmp_mn2_gpu, ws.Q_R_gpu, ws.K_gpu)
    mul!(S_Q, ws.K_gpu', ws.tmp_mn2_gpu, 1.0, 1.0)
    # S.Q += K' * Q.H  (n×m * m×n → n×n)  — note: K_gpu' is n×m, Q_H is m×n
    mul!(S_Q, ws.K_gpu', ws.Q_H, 1.0, 1.0)
    # S.Q += Q.H' * K  (n×m * m×n → n×n)
    mul!(S_Q, ws.Q_H', ws.K_gpu, 1.0, 1.0)

    # Symmetrize: S.Q = 0.5 * (S.Q + S.Q')
    S_Q .= (S_Q .+ S_Q') .* 0.5

    # Compute ΔV terms on CPU
    d_vec = Vector(d_cpu)
    q_r_vec = Array(ws.Q_r)
    t1 = dot(d_vec, q_r_vec)
    t2 = 0.5 * dot(d_vec, ws.Q_R_cpu * d_vec)

    return t1, t2
end

# =============================================================================
# Full GPU Backward Pass
# =============================================================================

"""
    backwardpass_memory_efficient_gpu!(solver, gpu_cache::GpuSolverCache)

GPU-accelerated memory-efficient backward pass for iLQR.
"""
function backwardpass_memory_efficient_gpu!(solver, gpu_cache::GpuSolverCache)
    jac_cache = gpu_cache.jac_cache
    ws = gpu_cache.bp_ws
    model = solver.model
    Z = solver.Z
    K = solver.K
    d = solver.d
    n, m, N = size(solver)

    # Terminal cost expansion (CPU, cheap)
    E_terminal = solver.E[2]
    Altro.stage_cost_expansion!(E_terminal, solver.obj, Z, N, solver.exp_cache)

    # Upload terminal cost-to-go to GPU
    copyto!(ws.S_q[2], Vector(E_terminal.q))
    copyto!(ws.S_Q[2], Matrix(E_terminal.Q))

    ΔV = @SVector zeros(2)
    S_next_idx = 2
    S_curr_idx = 1

    k = N - 1
    while k > 0
        # 1. Compute Jacobian on GPU
        gpu_compute_jacobian!(jac_cache, model, Z[k])
        fdx = view(jac_cache.jac_gpu, :, 1:n)
        fdu = view(jac_cache.jac_gpu, :, (n+1):(n+m))

        # 2. Cost expansion (CPU, diagonal cost is cheap)
        E_stage = solver.E[1]
        Altro.stage_cost_expansion!(E_stage, solver.obj, Z, k, solver.exp_cache)

        # Extract diagonal cost vectors and upload to pre-allocated GPU buffers
        copyto!(ws.cost_q_gpu, Vector(E_stage.q))
        @inbounds for i in 1:n
            ws.cost_Q_diag_cpu[i] = E_stage.Q[i, i]
        end
        copyto!(ws.cost_Q_diag_gpu, ws.cost_Q_diag_cpu)
        cost_r_cpu = Vector(E_stage.r)
        cost_R_diag_cpu = [E_stage.R[i, i] for i in 1:m]

        # 3. Q-function on GPU
        gpu_calc_Q!(ws, fdx, fdu,
                    ws.S_Q[S_next_idx], ws.S_q[S_next_idx],
                    cost_r_cpu, cost_R_diag_cpu)

        # 4. Regularization check (on CPU, m×m)
        if solver.opts.bp_reg
            vals = eigvals(Hermitian(ws.Q_R_cpu .+ solver.ρ[1] * I))
            if minimum(vals) <= 0
                @warn "Backward pass regularized"
                Altro.regularization_update!(solver, :increase)
                k = N - 1
                ΔV = @SVector zeros(2)
                S_next_idx = 2
                S_curr_idx = 1
                # Re-upload terminal cost-to-go
                copyto!(ws.S_q[2], Vector(E_terminal.q))
                copyto!(ws.S_Q[2], Matrix(E_terminal.Q))
                continue
            end
        end

        # 5. Gains on CPU (m=2, trivially fast)
        _gpu_calc_gains_simple!(K[k], d[k], ws.Q_R_cpu, ws.Q_H_cpu, ws.Q_r_cpu,
                                solver.ρ[1], solver.opts.bp_reg_type)

        # 6. Cost-to-go update on GPU
        t1, t2 = gpu_calc_ctg!(ws.S_Q[S_curr_idx], ws.S_q[S_curr_idx],
                                ws, K[k], d[k])
        ΔV += @SVector [t1, t2]

        # Swap
        S_next_idx, S_curr_idx = S_curr_idx, S_next_idx
        k -= 1
    end

    Altro.regularization_update!(solver, :decrease)
    return ΔV
end

"""
Simple gains computation matching the existing _calc_gains! interface.
Writes directly to SizedMatrix K and SizedVector d.
"""
function _gpu_calc_gains_simple!(K, d, Q_R_cpu, Q_H_cpu, Q_r_cpu, ρ, bp_reg_type)
    m = size(Q_R_cpu, 1)
    Quu_reg = copy(Q_R_cpu)
    for j in 1:m
        Quu_reg[j, j] += ρ
    end
    LAPACK.potrf!('U', Quu_reg)
    K_data = copy(Q_H_cpu)
    d_data = copy(Q_r_cpu)
    LAPACK.potrs!('U', Quu_reg, K_data)
    LAPACK.potrs!('U', Quu_reg, d_data)
    K .= -K_data
    d .= -d_data
    return nothing
end
