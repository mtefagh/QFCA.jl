#a type defined for metabolic networks
immutable Network
    S::SparseMatrixCSC{Float64}
    m::Int64
    n::Int64
    Rev::Array{Int64,1}
    Irr::Array{Int64,1}
    X::Base.SparseArrays.CHOLMOD.Factor{Float64}
    zeroAcc::Float64
    maxItr::Int64

    function Network(stoichimetry, reversibility)
      S = sparse(stoichimetry);
      (m, n) = size(S);
      Rev = find(reversibility .== 1);
      Irr = symdiff(1:n, Rev);
      X = Base.LinAlg.cholfact(speye(m) + S*S');
      zeroAcc = 1e-3;
      maxItr = 1e+4;
      new(S, m, n, Rev, Irr, X, zeroAcc, maxItr);
    end
end

#solving a linear system
function solve(model::Network, y::Array{Float64,1})
    temp = model.X\(y[model.n+1:end] - model.S*y[1:model.n]);
    return [y[1:model.n] + model.S'*temp; temp];
end

#projection onto a cone
function P(model::Network, x::Array{Float64,1}, idx1::Nullable{Int64}, idx2::Nullable{Int64}, forward::Bool)
    x[model.Irr] = x[model.Irr] .* (x[model.Irr] .> 0);
    if !isnull(idx1)
      x[get(idx1)] = x[get(idx1)] .* (forward ? (x[get(idx1)] .> 0) : (x[get(idx1)] .< 0));
    end
    if !isnull(idx2)
      x[get(idx2)] = 0;
    end
    return x;
end

#one iteration of the optimization algorithm
function iterate(model::Network, u::Array{Float64,1}, v::Array{Float64,1}, idx1::Nullable{Int64}, idx2::Nullable{Int64}, forward::Bool)
    w = solve(model, u + v);
    u = P(model, w - v, idx1, idx2, forward);
    v = v - w + u;
    return (u, v);
end

#conic optimization via operator splitting and homogeneous self-dual embedding
function opt(model::Network, u::Array{Float64,1}, v::Array{Float64,1}, idx1::Nullable{Int64}, idx2::Nullable{Int64}, forward::Bool)
    for counter in 1:5
      (u, v) = iterate(model, u, v, idx1, idx2, forward);
    end
    itr = 5;

    while (abs(model.S*u[1:model.n]-v[model.n+1:end]) .> model.zeroAcc) != falses(model.m) || (abs(model.S'*u[model.n+1:end] + v[1:model.n]) .> model.zeroAcc) != falses(model.n)
        (u, v) = iterate(model, u, v, idx1, idx2, forward);

        itr += 1;
        if itr > model.maxItr
          println("Err!");
          break;
        end
    end

    return (u, v);
end

#determining the reversibility type of each reaction, removing the blocked ones, and correcting the reversibility vector
function reduceModel(model::Network)
    revType = zeros(model.n);

    (u, v) = opt(model, ones(model.n + model.m), ones(model.n + model.m), Nullable{Int64}(), Nullable{Int64}(), true);
    revType[find(abs(v) .> model.zeroAcc)] = 1;

    for i in model.Rev
        (u, v) = opt(model, ones(model.n + model.m), ones(model.n + model.m), Nullable(i), Nullable{Int64}(), true);
        if abs(v[i]) .> model.zeroAcc
            revType[i] = 2;
        end

        (u, v) = opt(model, -ones(model.n + model.m), -ones(model.n + model.m), Nullable(i), Nullable{Int64}(), false);
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

    (m, n) = size(S);
    couplings = zeros(n, n);
    reacNum = 1:n;
    

    return (Network(S, Rev), couplings, reacNum);
end

#finding all the reactions coupled to the given reaction in the argument of the function
function coupled(model::Network, idx1::Nullable{Int64}, forward::Bool)
    U = SharedArray{Float64,2}(zeros(model.n + model.m, model.n));
    V = SharedArray{Float64,2}(zeros(model.n + model.m, model.n));

    (initU, initV) = opt(model, (forward ? +1 : -1)*ones(model.n + model.m), (forward ? +1 : -1)*ones(model.n + model.m), idx1, Nullable{Int64}(), forward);

    @sync @parallel for idx2 = 1:model.n
        (U[:, idx2], V[:, idx2]) = opt(model, initU, initV, idx1, Nullable(idx2), forward);
    end

    return (U, V);
end

#finding all the quantitative flux coupling coefficients
function couplingMatrices(model::Network, forward::Bool)
    couplingMatrices = Dict{Int64, Tuple{Array{Float64,2},Array{Float64,2}}}();

    if forward
        couplingMatrices[0] = coupled(model, Nullable{Int64}(), forward);
    end

    for idx in model.Rev
        couplingMatrices[idx] = coupled(model, Nullable(idx), forward);
    end

    return couplingMatrices;
end

#quantitative flux coupling analysis
function QFCA(model::Network)
    couplings = zeros(model.n, model.n);

    couplingF = couplingMatrices(model, true);
    couplingB = couplingMatrices(model, false);

    V = couplingF[0][2];

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

    for idx1 in model.Rev
        Vf = couplingF[idx1][2];
        Vb = couplingB[idx1][2];

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

    return (couplingF, couplingB, couplings);
end

#quantitative metabolite coupling analysis
function QMCA(model::Network)
    S = transpose(model.S);
    Rev = zeros(model.m);
    return QFCA(Network(S, Rev));
end

using COBRA, DataFrames;
#import the model
model = loadModel("ecoli_core_model.mat", "S", "model");
#initializing a network instance with attributes from the model
model = Network(model.S, 1*(model.lb .!= zeros(size(model.S)[2])));
#finding all the coupling relationships among the metabolites
@time couplingsM = QMCA(model)[end];
#reduce the model
#@time (model, couplings, reacNum) = reduceModel(model);
#finding all the coupling relationships among the reactions
@time couplings[reacNum, reacNum] = QFCA(model)[end];

#exporting the results
writetable("couplingsM.csv", DataFrame(couplingsM));
writetable("couplings.csv", DataFrame(couplings));