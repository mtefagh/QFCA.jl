#a type defined for metabolic networks
@everywhere immutable Network
    S::SparseMatrixCSC{Float64,Int64}
    m::Int64
    n::Int64
    Rev::Array{Int64,1}
    Irr::Array{Int64,1}
    X::Base.SparseArrays.CHOLMOD.Factor{Float64}
    zeroAcc::Float64
    maxItr::Int64

    function Network(stoichimetry, reversibility)
      S = stoichimetry;
      (m, n) = size(S);
      Rev = find(reversibility .== 1);
      Irr = symdiff(1:n, Rev);
      X = Base.LinAlg.cholfact(speye(m) + S*S');
      zeroAcc = 1e-4;
      maxItr = 1e+4;
      new(S, m, n, Rev, Irr, X, zeroAcc, maxItr);
    end
end

#solving a linear system
function solve(model::Network, y::Array{Float64,1})
    temp = model.X\(y[model.n+1:end] - model.S*y[1:model.n]);
    return [y[1:model.n] + model.S'*temp; temp];
end

#conic optimization via operator splitting and homogeneous self-dual embedding
function optimize(model::Network, u::Array{Float64,1}, v::Array{Float64,1}, idx1::Nullable{Int64}, idx2::Nullable{Int64}, forward::Bool)
    itr = 0;
    while itr < 5 || (abs.(model.S*u[1:model.n]-v[model.n+1:end]) .> model.zeroAcc) != falses(model.m) || (abs.(model.S'*u[model.n+1:end] + v[1:model.n]) .> model.zeroAcc) != falses(model.n)
        w = remotecall_fetch(solve, 1, model, u + v);
        #projection onto a cone
        u = w - v;
        u[model.Irr] = u[model.Irr] .* (u[model.Irr] .> 0);
        if !isnull(idx1)
          u[get(idx1)] = u[get(idx1)] .* (forward ? (u[get(idx1)] .> 0) : (u[get(idx1)] .< 0));
        end
        if !isnull(idx2)
          u[get(idx2)] = 0;
        end
        #one iteration of the optimization algorithm
        v = v - w + u;

        itr += 1;
        if itr > model.maxItr
          warn("Err!");
          break;
        end
    end

    return (u, v);
end

#determining the reversibility type of each reaction, removing the blocked ones, and correcting the reversibility vector
function reduceModel(model::Network)
    revType = zeros(model.n);

    (u, v) = optimize(model, ones(model.n + model.m), ones(model.n + model.m), Nullable{Int64}(), Nullable{Int64}(), true);
    revType[find(abs.(v) .> model.zeroAcc)] = 1;

    for i in model.Rev
        (u, v) = optimize(model, ones(model.n + model.m), ones(model.n + model.m), Nullable(i), Nullable{Int64}(), true);
        if abs(v[i]) .> model.zeroAcc
            revType[i] = 2;
        end

        (u, v) = optimize(model, -ones(model.n + model.m), -ones(model.n + model.m), Nullable(i), Nullable{Int64}(), false);
        if abs(v[i]) > model.zeroAcc
            revType[i] = (revType[i] == 2 ? 1 : 3);
        end
    end

    S = model.S;
    S[:, find(revType .== 2)] = - S[:, find(revType .== 2)];
    S = S[:, find(revType .!= 1)];

    Rev = ones(model.n);
    Rev[model.Irr] = 0;
    Rev[find(revType .== 2)] = 0;
    Rev[find(revType .== 3)] = 0;
    Rev = Rev[find(revType .!= 1)];

    return Network(S, Rev);
end

#finding all the reactions coupled to the given reaction in the argument of the function
function coupled(model::Network, idx1::Nullable{Int64}, forward::Bool)
    U = Array{Float64,2}(model.n + model.m, model.n);
    V = Array{Float64,2}(model.n + model.m, model.n);

    (initU, initV) = optimize(model, (forward ? +1 : -1)*ones(model.n + model.m), (forward ? +1 : -1)*ones(model.n + model.m), idx1, Nullable{Int64}(), forward);

    for idx2 = 1:model.n
        (U[:, idx2], V[:, idx2]) = optimize(model, initU, initV, idx1, Nullable(idx2), forward);
    end

    return (U, V);
end

#finding all the quantitative flux coupling coefficients
function couplingMatrices(model::Network)
    couplingF = Dict{Int64, Tuple{Array{Float64,2},Array{Float64,2}}}();
    couplingB = Dict{Int64, Tuple{Array{Float64,2},Array{Float64,2}}}();

    for i in 1:length(model.Rev)
        couplingF[i] = coupled(model, Nullable(model.Rev[i]), true);
        couplingB[i] = coupled(model, Nullable(model.Rev[i]), false);
    end

    return (couplingF, couplingB);
end

#quantitative flux coupling analysis
function QFCA(model::Network)
    model = reduceModel(model);
    couplings = zeros(model.n, model.n);
    (U, V) = coupled(model, Nullable{Int64}(), true);

    for i in 1:model.n
        for j in 1:model.n
            if abs(V[i, j]) > model.zeroAcc
              if couplings[i, j] == 0
                couplings[i, j] = 3;
                couplings[j, i] = 4;
              else
                couplings[i, j] = 1;
                couplings[j, i] = 1;
              end
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

using COBRA, CSV;
#import the model
model = loadModel("ecoli_core_model.mat", "S", "model");
rev = CSV.read("rev.csv", delim = ' ', header = false);
rev = rev[:, 2];
#initializing a network instance with attributes from the model
model = Network(model.S, rev);

#finding all the coupling relationships among the reactions
info("QFCA");
@time couplings = QFCA(model)[end];

info(all(readdlm("Ar.csv", header = false) .== couplings));