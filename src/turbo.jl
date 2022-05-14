using LoopVectorization
function cvec_elmul!(cc::AbstractArray{Complex{T},N1}, ca::AbstractArray{Complex{T},N2}, cb::AbstractArray{Complex{T},N3}) where {T,N1,N2,N3}
    return cvec_elmul!(cc, ca, cb, true)
end
function cvec_elmul!(cc::AbstractArray{Complex{T},N1}, ca::AbstractArray{Complex{T},N2}, cb::AbstractArray{Complex{T},N3}, alpha::Union{Real,Bool}) where {T,N1,N2,N3}
    c = reinterpret(reshape, T,  vec(cc))
    a = reinterpret(reshape, T,  vec(ca))
    b = reinterpret(reshape, T,  vec(cb))
    re = zero(T)
    im = zero(T)
    @tturbo for i in eachindex(axes(c,2), axes(a,2), axes(b,2))
        re = a[1, i] * b[1, i] - a[2, i] * b[2, i]
        im = a[1, i] * b[2, i] + a[2, i] * b[1, i]
        c[1,i]=re * alpha
        c[2,i]=im * alpha
    end
end

function cvec_elmul!(cc::AbstractArray{Complex{T},N1}, ca::AbstractArray{Complex{T},N2}, cb::AbstractArray{Complex{T},N3}, calpha::Complex{T}) where {T,N1,N2,N3}
    c = reinterpret(reshape, T,  vec(cc))
    a = reinterpret(reshape, T,  vec(ca))
    b = reinterpret(reshape, T,  vec(cb))
    re = zero(T)
    im = zero(T)
    @tturbo for i in eachindex(axes(c,2), axes(a,2), axes(b,2))
        re = a[1, i] * b[1, i] - a[2, i] * b[2, i]
        im = a[1, i] * b[2, i] + a[2, i] * b[1, i]
        c[1,i]=re * calpha.re - im * calpha.im
        c[2,i]=im * calpha.re + re * calpha.im
    end
end