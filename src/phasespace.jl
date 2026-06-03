function qfunc end
function wignersu2 end
function qfuncsu2 end
function coherentspinstate end
 
function wigner end
 
function wigner(rho::Operator{B,B}, xvec::AbstractVector, pvec::AbstractVector) where {B<:FockBasis}
    _wigner_fock_dm(rho.data, rho.basis_l.N, xvec, pvec)
end
 
function wigner(psi::Ket{B}, xvec::AbstractVector, pvec::AbstractVector) where {B<:FockBasis}
    _wigner_fock_ket(psi.data, psi.basis.N, xvec, pvec)
end
 
 
function _logfact(N::Int)
    lf = zeros(Float64, N+2)
    for i in 1:N+1
        lf[i+1] = lf[i] + log(Float64(i))
    end
    return lf
end
 
function _laguerre(n::Int, k::Int, x::Float64)
    n == 0 && return 1.0
    n == 1 && return Float64(1 + k) - x
    l0, l1 = 1.0, Float64(1 + k) - x
    for j in 2:n
        l2 = ((2j - 1 + k - x) * l1 - (j - 1 + k) * l0) / j
        l0, l1 = l1, l2
    end
    return l1
end
 
function _wigner_fock_dm(ρ::AbstractMatrix, N::Int, xvec, pvec)
    W  = Matrix{Float64}(undef, length(xvec), length(pvec))
    lf = _logfact(N)
    for (ix, x) in enumerate(xvec)
        for (ip, p) in enumerate(pvec)
            β   = complex(Float64(x), Float64(p))
            β2  = abs2(β)
            ef  = exp(-2β2)
            arg = Float64(4β2)
            tot = 0.0
            for m in 0:N
                tot += real(ρ[m+1, m+1]) * (-1)^m * _laguerre(m, 0, arg)
            end
            for m in 0:N-1
                for l in m+1:N
                    k    = l - m
                    scl  = exp(0.5 * (lf[m+1] - lf[l+1]))
                    wml  = (-1)^m * (2*conj(β))^k * scl * _laguerre(m, k, arg)
                    tot += 2 * real(ρ[m+1, l+1] * wml)
                end
            end
            W[ix, ip] = (2/π) * ef * tot
        end
    end
    return W
end
 
function _wigner_fock_ket(c::AbstractVector, N::Int, xvec, pvec)
    W  = Matrix{Float64}(undef, length(xvec), length(pvec))
    lf = _logfact(N)
    for (ix, x) in enumerate(xvec)
        for (ip, p) in enumerate(pvec)
            β   = complex(Float64(x), Float64(p))
            β2  = abs2(β)
            ef  = exp(-2β2)
            arg = Float64(4β2)
            tot = 0.0
            for m in 0:N
                tot += abs2(c[m+1]) * (-1)^m * _laguerre(m, 0, arg)
            end
            for m in 0:N-1
                for l in m+1:N
                    k    = l - m
                    scl  = exp(0.5 * (lf[m+1] - lf[l+1]))
                    wml  = (-1)^m * (2*conj(β))^k * scl * _laguerre(m, k, arg)
                    tot += 2 * real(conj(c[m+1]) * c[l+1] * wml)
                end
            end
            W[ix, ip] = (2/π) * ef * tot
        end
    end
    return W
end