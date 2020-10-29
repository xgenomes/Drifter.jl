#### Cell lists. Super simple...
@inline _bin_idx(x :: Float64, bin_width :: Float64) = ceil(Int64, x/bin_width)

@inline function _squared_dist(p,n)
    r = 0.0
    for i in 1:length(p)
        r += (p[i]-n[i])^2
    end
    r
end

struct CellList{N}
    cells :: Dict{NTuple{N, Int64}, Vector{NTuple{N, Float64}}}
    radius :: Float64
    function CellList{N}(r) where {N}
        @assert N == 2 || N == 3
        new{N}(Dict(), r)
    end
end

function Base.push!(t :: CellList{N}, p :: NTuple{N, Float64}) where {N}
    k = _bin_idx.(p, t.radius)

    list = if k ∉ keys(t.cells)
        t.cells[k] = NTuple{N, Float64}[]
    else
        t.cells[k]
    end

    push!(list, p)
end

function CellList(points :: Vector{NTuple{N, Float64}}, radius :: Float64) where {N}
    t = CellList{N}(radius)
    for p in points
        push!(t, p)
    end
    t
end

# TODO: Remove duplicate code.
function neighbor(t :: CellList{2}, p :: NTuple{2, Float64})
    r_sq = t.radius*t.radius
    offsets = (0,-1,1)

    bin_idx = _bin_idx.(p, t.radius)

    for o_x in offsets, o_y in offsets
        k = bin_idx .+ (o_x, o_y)
        if k ∈ keys(t.cells)
            for n in t.cells[k]
                if _squared_dist(p,n) <= r_sq
                    return true, n
                end
            end
        end
    end
    return false, (Inf,Inf)
end


function neighbor(t :: CellList{3}, p :: NTuple{3, Float64})
    r_sq = t.radius*t.radius
    offsets = (0,-1,1)

    bin_idx = _bin_idx.(p, t.radius)

    for o_x in offsets, o_y in offsets, o_z in offsets
        k = bin_idx .+ (o_x, o_y, o_z)
        if k ∈ keys(t.cells)
            for n in t.cells[k]
                if _squared_dist(p,n) <= r_sq
                    return true, n
                end
            end
        end
    end
    return false, (Inf,Inf,Inf)
end
