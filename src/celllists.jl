#### Cell lists. Super simple...
@inline _bin_idx(x :: Float64, bin_width :: Float64) = ceil(Int64, x/bin_width)

struct CellList
    cells :: Dict{NTuple{2, Int64}, Vector{P2D}}
    radius :: Float64
    CellList(r) = new(Dict(), r)
end

function Base.push!(t :: CellList, p :: P2D)
    k = Tuple(_bin_idx.(p, t.radius))
    list = if k ∉ keys(t.cells)
        l = P2D[]
        t.cells[k] = l
    else
        t.cells[k]
    end

    push!(t.cells[k], p)
end

function CellList(points :: Vector{P2D}, radius :: Float64)
    t = CellList(radius)
    for p in points
        push!(t, p)
    end
    t
end

function neighbor(t :: CellList, p :: P2D)
    r_sq = t.radius*t.radius
    offsets = (0,-1,1)

    i_x = _bin_idx(p[1], t.radius)
    i_y = _bin_idx(p[2], t.radius)

    for o_x in offsets, o_y in offsets
        k = (i_x + o_x, i_y + o_y)
        if k ∈ keys(t.cells)
            for n in t.cells[k]
                d = p-n
                if dot(d, d) <= r_sq
                    return true, n
                end
            end
        end
    end
    return false, P2D(Inf,Inf)
end
