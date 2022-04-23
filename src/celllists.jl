#### Cell lists. Super simple...
@inline _bin_idx(x :: Float64, bin_width :: Float64) = ceil(Int64, x/bin_width) + 1

@inline function _squared_dist(p,n)
    r = 0.0
    for i in 1:length(p)
        r += (p[i]-n[i])^2
    end
    r
end

struct CellList{N}
    cells :: Vector{Vector{NTuple{N, Float64}}}
    radius :: Float64
    indexes :: SparseMatrixCSC{Int, Int}
    function CellList{N}(r, maxx, maxy) where {N}
        @assert N == 2 || N == 3
        new{N}(Vector{NTuple{N, Float64}}[], r, spzeros(Int, _bin_idx(maxx, r) + 1, _bin_idx(maxy, r) + 1))
    end
end

function Base.push!(t :: CellList{N}, p :: NTuple{N, Float64}) where {N}
    k = _bin_idx.(p, t.radius)

    list = if t.indexes[k[1], k[2]] == 0
        t.indexes[k[1], k[2]] = length(t.cells) + 1
        push!(t.cells, NTuple{N, Float64}[])
        t.cells[end]
    else
        t.cells[t.indexes[k[1], k[2]]]
    end

    push!(list, p)
end

function CellList(points :: Vector{NTuple{N, Float64}}, radius :: Float64, maxx :: Float64, maxy :: Float64) where {N}
    t = CellList{N}(radius, maxx, maxy)
    for p in points
        push!(t, p)
    end
    t
end

function Base.haskey(t :: CellList{N}, k :: NTuple{N, Int}) where {N}
    getindex(t.indexes, k...) > 0
end

function Base.getindex(t :: CellList{N}, k :: NTuple{N, Int}) where {N}
    t.cells[getindex(t.indexes, k...)]
end

# TODO: Remove duplicate code.
function neighbor(t :: CellList{2}, p :: NTuple{2, Float64})
    r_sq = t.radius*t.radius
    offsets = (0,-1,1)

    bin_idx = _bin_idx.(p, t.radius)
    (f,closest_p) = (false, (Inf, Inf))
    sq_d_min = r_sq
    for o_x in offsets, o_y in offsets
        k = bin_idx .+ (o_x, o_y)
        if haskey(t, k)
            for n in t[k]
                if _squared_dist(p,n) < sq_d_min
                    f = true
                    sq_d_min = _squared_dist(p,n)
                    closest_p = n
                end
            end
        end
    end
    return f, closest_p
end

function neighbor(t :: CellList{3}, p :: NTuple{3, Float64})
    r_sq = t.radius*t.radius
    offsets = (0,-1,1)

    bin_idx = _bin_idx.(p, t.radius)
    (f,closest_p) = (false, (Inf, Inf, Inf))
    sq_d_min = r_sq
    for o_x in offsets, o_y in offsets, o_z in offsets
        k = bin_idx .+ (o_x, o_y, o_z)
        if haskey(t, k)
            for n in t[k]
                if _squared_dist(p,n) < sq_d_min
                    f = true
                    sq_d_min = _squared_dist(p,n)
                    closest_p = n
                end
            end
        end
    end
    return f, closest_p
end