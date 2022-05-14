using LoopVectorization
function cvec_elmul!(cc::AbstractVector{Complex{T}}, ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}) where {T}
    return cvec_elmul!(cc, ca, cb, true)
end
function cvec_elmul!(cc::AbstractVector{Complex{T}}, ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}, alpha::Union{Real,Bool}) where {T}
    c = reinterpret(reshape, T, cc)
    a = reinterpret(reshape, T, ca)
    b = reinterpret(reshape, T, cb)
    re = zero(T)
    im = zero(T)
    @tturbo for i in eachindex(cc, ca, cb)
        re = a[1, i] * b[1, i] - a[2, i] * b[2, i]
        im = a[1, i] * b[2, i] + a[2, i] * b[1, i]
        c[1,i]=re * alpha
        c[2,i]=im * alpha
    end
end

function cvec_elmul!(cc::AbstractVector{Complex{T}}, ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}, calpha::Complex{T}) where {T}
    c = reinterpret(reshape, T, cc)
    a = reinterpret(reshape, T, ca)
    b = reinterpret(reshape, T, cb)
    re = zero(T)
    im = zero(T)
    @tturbo for i in eachindex(cc, ca, cb)
        re = a[1, i] * b[1, i] - a[2, i] * b[2, i]
        im = a[1, i] * b[2, i] + a[2, i] * b[1, i]
        c[1,i]=re * calpha.re - im * calpha.im
        c[2,i]=im * calpha.re + re * calpha.im
    end
end