using LinearAlgebra, SuiteSparse, SparseArrays, DelimitedFiles;

#a type defined for metabolic networks
struct Network
    S::SparseMatrixCSC{Float64,Int64}
    m::Int64
    n::Int64
    Rev::Array{Int64,1}
    Irr::Array{Int64,1}
    X::SuiteSparse.CHOLMOD.Factor{Float64}
    zeroAcc::Float64
    maxItr::Int64

    function Network(stoichimetry, reversibility)
        S = sparse(stoichimetry);
        m, n = size(S);
        Rev = filter(i -> reversibility[i] == 1, 1:n);
        Irr = symdiff(1:n, Rev);
        X = cholesky(sparse(I, m, m) + S*S');
        zeroAcc = 1e-4;
        maxItr = 1e+5;
        new(S, m, n, Rev, Irr, X, zeroAcc, maxItr);
    end
end

#solving a linear system
solve(model::Network, y::Array{Float64,1}) = map(x -> [y[1:model.n] + model.S'*x; x], [model.X\(y[model.n+1:end] - model.S*y[1:model.n])]);

#conic optimization via operator splitting and homogeneous self-dual embedding
function optimize(model::Network, u::Array{Float64,1}, v::Array{Float64,1}, idx1::Union{Int64, Nothing}, idx2::Union{Int64, Nothing}, forward::Bool)
    itr = 0;
    while itr < 5 || (abs.(model.S*u[1:model.n]-v[model.n+1:end]) .> model.zeroAcc) != falses(model.m) || (abs.(model.S'*u[model.n+1:end] + v[1:model.n]) .> model.zeroAcc) != falses(model.n)
        w = solve(model, u + v);
        #projection onto a cone
        u = w[1] - v;
        u[model.Irr] = u[model.Irr] .* (u[model.Irr] .> 0);
        if idx1 != nothing
          u[idx1] = u[idx1] .* (forward ? (u[idx1] .> 0) : (u[idx1] .< 0));
        end
        if idx2 != nothing
          u[idx2] = 0;
        end
        #one iteration of the optimization algorithm
        v = v - w[1] + u;

        itr += 1;
        if itr > model.maxItr
          error("Err!");
          break;
        end
    end

    return (u, v);
end

#determining the reversibility type of each reaction, removing the blocked ones, and correcting the reversibility vector
function reduceModel(model::Network)
    revType = zeros(model.n);

    (u, v) = optimize(model, ones(model.n + model.m), ones(model.n + model.m), nothing, nothing, true);
    for i in filter(i -> abs(v[i]) > model.zeroAcc, 1:model.n)
        revType[i] = 1;
    end

    for i in model.Rev
        (u, v) = optimize(model, ones(model.n + model.m), ones(model.n + model.m), i, nothing, true);
        if abs(v[i]) > model.zeroAcc
            revType[i] = 2;
        end

        (u, v) = optimize(model, -ones(model.n + model.m), -ones(model.n + model.m), i, nothing, false);
        if abs(v[i]) > model.zeroAcc
            revType[i] = (revType[i] == 2 ? 1 : 3);
        end
    end

    S = model.S;
    revType2 = filter(i -> revType[i] == 2, 1:model.n);
    S[:, revType2] = - S[:, revType2];
    S = S[:, revType .!= 1];

    Rev = ones(model.n);
    for i in model.Irr
        Rev[i] = 0;
    end
    for i in revType2
        Rev[i] = 0;
    end
    revType3 = filter(i -> revType[i] == 3, 1:model.n);
    for i in revType3
        Rev[i] = 0;
    end
    Rev = Rev[revType .!= 1];

    return Network(S, Rev);
end

#finding all the reactions coupled to the given reaction in the argument of the function
function coupled(model::Network, idx1::Union{Int64, Nothing}, forward::Bool)
    U = Array{Float64,2}(undef, model.n + model.m, model.n);
    V = Array{Float64,2}(undef, model.n + model.m, model.n);

    (initU, initV) = optimize(model, (forward ? +1 : -1)*ones(model.n + model.m), (forward ? +1 : -1)*ones(model.n + model.m), idx1, nothing, forward);

    for idx2 = 1:model.n
        (U[:, idx2], V[:, idx2]) = optimize(model, initU, initV, idx1, idx2, forward);
    end

    return (U, V);
end

#finding all the quantitative flux coupling coefficients
function couplingMatrices(model::Network)
    couplingF = Dict{Int64, Tuple{Array{Float64,2},Array{Float64,2}}}();
    couplingB = Dict{Int64, Tuple{Array{Float64,2},Array{Float64,2}}}();

    for i in 1:length(model.Rev)
        couplingF[i] = coupled(model, model.Rev[i], true);
        couplingB[i] = coupled(model, model.Rev[i], false);
    end

    return (couplingF, couplingB);
end

#quantitative flux coupling analysis
function QFCA(model::Network)
    model = reduceModel(model);
    couplings = zeros(model.n, model.n);
    (U, V) = coupled(model, nothing, true);

    for i in 1:model.n, j in 1:model.n
        if abs(V[i, j]) > model.zeroAcc
            if couplings[i, j] == 0
                couplings[i, j] = 3;
                couplings[j, i] = 4;
            else
                couplings[i, j] = 1;
                couplings[j, i] = 1;
            end
        end
        couplings[i, i] = 1;
    end

    @time (couplingF, couplingB) = couplingMatrices(model);

    for i in 1:length(model.Rev)
        Vf = couplingF[i][2];
        Vb = couplingB[i][2];
        idx1 = model.Rev[i];

        for idx2 in model.Rev
            if abs(Vf[idx1, idx2]) > model.zeroAcc && abs(Vb[idx1, idx2]) > model.zeroAcc
                couplings[idx1, idx2] = 1;
                couplings[idx2, idx1] = 1;
            end
        end

        for idx2 in model.Irr
            if abs(Vf[idx1, idx2]) > model.zeroAcc && abs(Vb[idx1, idx2]) > model.zeroAcc
                couplings[idx1, idx2] = 3;
                couplings[idx2, idx1] = 4;
            end
        end
    end

    return (U, V, couplingF, couplingB, couplings);
end

#import the model
S = readdlm("S.csv", header = false);
@assert typeof(S) == Array{Float64,2};
rev = readdlm("rev.csv", header = false);
@assert typeof(rev) == Array{Float64,2};
#initializing a network instance with attributes from the model
model = Network(S, rev);

#finding all the coupling relationships among the reactions
println("QFCA");
@time couplings = QFCA(model)[end];

println("The answer is $(all(readdlm("Ar.csv", header = false) .== couplings) ? "correct" : "wrong").");
using Colors;
palette = distinguishable_colors(5);
map(x -> palette[convert(Int64, x+1)], couplings)