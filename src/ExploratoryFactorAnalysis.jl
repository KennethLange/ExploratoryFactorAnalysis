__precompile__()

module ExploratoryFactorAnalysis

using LinearAlgebra, StatsBase, Distributions, Arpack
using MultivariateStats, KrylovKit, Printf, Random, DataFrames

export ExploratoryPCA,
       FactorAnalysisGN, 
       FactorAnalysisMM, 
       FactorAnalysisPartial, 
       FactorAnalysisFull,
       FactorAnalysisLAD,
       LoadingsUpdate,
       GenerateRandomData,
       TestsBenchmark,
       TestsAccuracy

"""Forms r principal components of the data matrix X."""
function ExploratoryPCA(X::Matrix, r::Int)
  X = X .- mean(X, dims=1) # centered data
  S = cov(X) # sample covariance matrix
  (lambda, V) = eigs(S, r, which=:LR) # Arpack eigendecomposition
  V = real(V) # convert to real
  (lambda, V) = EigenRefinement(S, V) # corrected eigendecomposition
  perm  = sortperm(lambda, rev = true)
  lambda = lambda[perm] # ordered eigenvalues
  V = V[:, perm] # ordered eigenvectors
  PC = X * V # principal components 
  return (PC, lambda, V)
end

"""Projects the symmetric matrix S onto the closest positive 
semidefinite matrix of rank r or less by full eigen-decomposition."""
function PositiveDefiniteProjection(S::Matrix{T}, r::Int) where T <: Real
#
  (lambda, V) = eigen(S)
  (n, p) = size(S)
  L = zeros(T, p, 0) # factor loading matrix
  for j = (n - r + 1):n
    if lambda[j] > zero(T)
      L = [L sqrt(lambda[j]) * V[:, j]]
    end   
  end
  return L
end

# function LoadingsUpdate(S::Matrix{T}, r::Int) where T <: Real
#   (lambda, V) = eigs(S, r, which=:LR) # Arpack call
#   V = real(V) # convert to real
#   (lambda, V) = EigenRefinement(S, V)
#   D = Diagonal(sqrt.(max.(lambda[1:r], zero(T))))
#   return V[:, 1:r] * D
# end

"""Updates factor loadings."""
function LoadingsUpdate(S::Matrix{T}, r::Int; EigenMethod = "KrylovKit", Refine = true) where T <: Real
#
  if EigenMethod == "KrylovKit"
    vals, vecs, _ = eigsolve(S, r, :LR; issymmetric=true)
    r_found = min(length(vals), r)
    lambda = real.(vals[1:r_found])
    V = hcat(vecs[1:r_found]...)
    D = Diagonal(sqrt.(max.(lambda, zero(T))))
    return V * D
  else
    (lambda, V) = eigs(S, r, which=:LR) # Arpack call
     V = real(V) # convert to real
    if Refine
      (lambda, V) = EigenRefinement(S, V)
    end
    D = Diagonal(sqrt.(max.(lambda[1:r], zero(T))))
    return V[:, 1:r] * D
  end
end

"""Refines r approximate eigenvectors of the symmetric matrix S."""
function EigenRefinement(S::Matrix{T}, V::Matrix{T}) where T <: Real
  (p, r) =  size(V)
  foreach(normalize!, eachcol(V)) # normalize eigenvectors
  (lambda, E) = (zeros(T, r), zeros(T, r, r))
  VtSV = transpose(V) * ( S * V)
  VtV = transpose(V) * V
  lambda = diag(VtSV) # eigenvalues
  for i = 1:r
    for j = 1:(i - 1)
      E[i, j] = VtSV[i, j] - lambda[j] * VtV[i, j]
      E[i, j] = E[i, j] /(lambda[j] - lambda[i])
      E[j, i] = VtSV[j, i] - lambda[i] * VtV[j, i]
      E[j, i] = E[j, i] / (lambda[i] - lambda[j])
    end
  end
  V = V * (I + E) # improved eigenvectors
  foreach(normalize!, eachcol(V)) # normalize eigenvectors
  return (lambda, V)
end

"""Evaluates the proximal map (soft thresholding) of the 
absolute value |y|."""
function ProxAbsolute(y::T, t::T) where T <: Real
  return sign(y) * max(zero(T), abs(y) - t)
end

"""Performs rank r factor analysis on S by MM Gauss-Newton updates."""
function FactorAnalysisMM(S::Matrix{T}, r::Int) where T <: Real
  (p, iters) = (size(S, 1), 0) # size of sample covariance matrix
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  L = randn(p, r) # random initialization of loadings
  LtL = zeros(T, r, r)
#   println(0,"  ",norm(S - L * L' - Diagonal(d)))
  for iter = 1:500
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(L[i, :])^2, zero(T))
    end
    mul!(LtL, transpose(L), L) # prepare to update loadings
    L .= L .+ (S * L - L * LtL - Diagonal(d) * L) / (2LtL)
#     println(iter,"  ",norm(S - L * L' - Diagonal(d)))
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  SVD = svd(L) # orthogonalize the columns of L
  L = SVD.U * Diagonal(SVD.S)
  return (L, d, iters)
end

"""Performs rank r factor analysis on S by standard Gauss-Newton 
updates."""
function FactorAnalysisGN(S::Matrix{T}, r::Int) where T <: Real
  (p, iters) = (size(S, 1), 0) # size of sample covariance matrix
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  L = randn(T, p, r) # random initialization of loadings
  LtL = zeros(T, r, r)
  Y = zeros(T, r, r)
  Z = zeros(T, p, r)
#   println(0,"  ",norm(S - L * L' - Diagonal(d)))
  for iter = 1:500
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(L[i, :])^2, zero(T))
    end
    mul!(LtL, transpose(L), L) # prepare to update loadings
    Y = L / LtL
    mul!(Z, S, Y)
    Z .= Z .- Diagonal(d) * Y
    L .= Z .- L * ((transpose(Y) * Z - I) ./ 2) # loadings update
#     println(iter,"  ",norm(S - L * L' - Diagonal(d)))
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  SVD = svd(L) # orthogonalize the columns of L
  L = SVD.U * Diagonal(SVD.S)
  return (L, d, iters)
end

"""Performs rank r factor analysis on S by exact factor loading 
updates with a partial eigen-decomposition."""
function FactorAnalysisPartial(S::Matrix{T}, r::Int; 
  EigenMethod = "KrylofKit") where T <: Real
#
  (p, iters) = (size(S, 1), 0)
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  L = randn(p, r) # random factor loadings
#   println(0,"  ",norm(S - L * L' - Diagonal(d)))
  for iter = 1:500
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(L[i, :])^2, zero(T))
    end
    for i = 1:p # adjust diagonal sample variances
      S[i, i] = S[i, i] - d[i]
    end
    if EigenMethod != "KrylofKit"
      L = LoadingsUpdate(S, r, EigenMethod = "Arpack") # update loadings
    else
      L = LoadingsUpdate(S, r, EigenMethod = "KrylofKit")
    end
    for i = 1:p # restore sample variances
      S[i, i] = S[i, i] + d[i]
    end 
#     println(iter,"  ",norm(S - L * L' - Diagonal(d)))
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  return (L, d, iters)
end

"""Performs rank r factor analysis on S by exact factor loading 
updates with a full eigen-decomposition."""
function FactorAnalysisFull(S::Matrix{T}, r::Int) where T <: Real
  (p, iters) = (size(S, 1), 0)
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  L = randn(p, r) # random factor loadings
#   println(0,"  ",norm(S - L * L' - Diagonal(d)))
  for iter = 1:500
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(L[i, :])^2, zero(T))
    end
    for i = 1:p # adjust diagonal sample variances
      S[i, i] = S[i, i] - d[i]
    end
    L = PositiveDefiniteProjection(S, r) # update loadings
    for i = 1:p # restore sample variances
      S[i, i] = S[i, i] + d[i]
    end 
#     println(iter,"  ",norm(S - L * L' - Diagonal(d)))
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  return (L, d, iters)
end

"""Performs rank r factor analysis on S by Gauss-Newton updates
and an approximate least absolute deviation loss."""
function FactorAnalysisLAD(S::Matrix{T}, r::Int, mu::T) where T <: Real
  (p, iters) = (size(S, 1), 0)
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  L = randn(p, r) # factor loadings
  LtL = zeros(T, r, r)
  Y = zeros(T, r, r)
  Z = zeros(T, p, r)
  R = similar(S)
#   println(0,"  ",norm(S - L * L', 1))
  for iter = 1:500
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(L[i, :])^2, zero(T))
    end
    D = Diagonal(d)
    mul!(R, L, transpose(L))
    for j = 1:p
      for i = 1:p
        pr = ProxAbsolute(S[i, j] - R[i, j] - D[i, j], mu)
        R[i, j] = S[i, j] - D[i, j] - pr
      end
    end
    mul!(LtL, transpose(L), L) # prepare to update loadings
    Y = L / LtL
    mul!(Z, S, Y)
    Z .= Z .- Diagonal(d) * Y
    L .= Z .- L * ((transpose(Y) * Z - I) ./ 2) # Gauss-Newton update
#     println(iter,"  ",norm(S - L * L' - D, 1))
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  SVD = svd(L) # orthogonalize the columns of L
  L = SVD.U * Diagonal(SVD.S)
  return (L, d, iters)
end

"""Generates random factor analysis data."""
function GenerateRandomData(n, p, r)
  avg = randn(p) # means
  L = randn(p, r) # factor loading
  d = rand(p) # specific variances
  D = Diagonal(sqrt.(d))
  X = zeros(n, p)
  for i = 1:n
    X[i, :] = L * randn(r) + D * randn(p)
  end
  S = cov(X)
  Y = (X .- mean(X, dims = 2))'  
  return (S, Y)
end

"""Runs test problems of rank r and Moreau constant mu."""
function TestsBenchmark(mu)
  n = 4000 # cases
  for p in [500] # predictors
    for r in [5] # rank
      (S, Y) = GenerateRandomData(n, p, r) # covariance matrix and data
      (L1, d1, iters1) = FactorAnalysisGN(S, r)
      (L2, d2, iters2) = FactorAnalysisMM(S, r);      
      (L2, d3, iters3) = FactorAnalysisPartial(S, r);
      (L3, d4, iters4) = FactorAnalysisFull(S, r);
      (L4, d5, iters5) = FactorAnalysisLAD(S, r, mu);
    end
  end
  trials = 5
  time = zeros(trials, 7)
  avg = zeros(7)
  results_df = DataFrame(
    PR_Tuple      = String[],
    GN_Time       = Float64[],
    Avg_Iters     = Float64[],
    Ratio_MM      = Float64[],
    Ratio_Partial = Float64[],
    Ratio_Full    = Float64[],
    Ratio_Robust  = Float64[],
    Ratio_EM      = Float64[],
    Ratio_CM      = Float64[]
  )
  @printf("%-32s | %s\n", " ", "      Relative Time Ratio ( vs GN )")
  println("-"^94)
  @printf("%-12s | %-8s | %-6s | %-7s | %-7s | %-7s | %-7s | %-7s | %-7s\n", 
    "(p, r)", "GN", "Iters",
    "MM", "Partial", "Full", "Robust", "EM", "CM")
  println("-"^98)
    
  for r in [5, 10, 100, 200] # rank
    for p in [250, 500, 1000, 2000, 4000, 8000] # predictors    
      iters = 0
      for trial = 1:trials
        (S, Y) = GenerateRandomData(n, p, r); # covariance matrix and data
        time[trial, 1] = @elapsed (L1, d1, iters1) = FactorAnalysisGN(S, r)
        iters = iters + iters1
        time[trial, 2] = @elapsed (L2, d2, iters2) = FactorAnalysisMM(S, r)
        if r <= 5 
          time[trial, 3] = @elapsed (L3, d3, iters3) = FactorAnalysisPartial(S, r)
        end
        if r <= 5 && p <= 500
          time[trial, 4] = @elapsed (L4, d4, iters4) = FactorAnalysisFull(S, r)
        end
        if r <= 5
          time[trial, 5] = @elapsed (L5, d5, iters5) = FactorAnalysisLAD(S, r, mu)
        end
       if p <= 500 && r <= 5
         time[trial, 6] = @elapsed M1 = fit(FactorAnalysis, Y, method = :em, maxoutdim = r)
         time[trial, 7] = @elapsed M2 = fit(FactorAnalysis, Y, method = :cm, maxoutdim = r)
        end
      end
      for i = 1:7 # average Gauss-Newton time and ratio times
        avg[i] = mean(time[:, i])
        if i > 1
          avg[i] = avg[i] / avg[1]
        end
      end            
      push!(results_df, (
          string((p, r)), # PR_Tuple
          avg[1],         # GN_Time
          iters / trials, # Avg_Iters
          avg[2],         # Ratio_MM
          avg[3],         # Ratio_Partial
          avg[4],         # Ratio_Full
          avg[5],         # Ratio_Robust
          avg[6],         # Ratio_EM
          avg[7]          # Ratio_CM
      ))            
    @printf("%-12s | %-8.4g | %-6.1f | %-7.5g | %-7.5g | %-7.5g | %-7.5g | %-7.5g | %-7.5g\n",
        string((p, r)),
        avg[1],
        iters / trials,
        avg[2], avg[3], avg[4],
        avg[5], avg[6], avg[7]
      )    
      fill!(time, 0.0)
    end
  end
  return results_df
end

"""Runs test problems of rank r and Moreau constant mu."""
function TestsAccuracy(r, mu; EigenMethod = "Arpack")
  results_df = DataFrame(
    NP_Tuple      = String[],
    GN_Error      = Float64[],
    GN_Iters      = Int[],
    Partial_Error = Float64[],
    Partial_Iters = Int[]
  )
  @printf("%-12s | %-25s | %-6s | %-25s | %-6s\n",
    "(n, p)", "GN", "Iters", "Partial", "Iters")
  println("-"^90)
  for n in [500] # cases
    for p in [6, 10, 25, 50, 100, 250, 500] # predictors
      (S, Y) = GenerateRandomData(n, p, r); # covariance matrix and data
      (L1, d1, iters1) = FactorAnalysisGN(S, r);
      (L2, d2, iters2) = FactorAnalysisPartial(S, r, EigenMethod);
      gn_error = norm(S - L1 * L1' - Diagonal(d1))
      partial_error = norm(S - L2 * L2' - Diagonal(d2))
      @printf("%-12s | %-25s | %-6s | %-25s | %-6s\n",
        string((n, p)), gn_error, iters1, partial_error, iters2
      )
      push!(results_df, (
        string((n, p)),  # NP_Tuple
        gn_error,        # GN_Error
        iters1,          # GN_Iters
        partial_error,   # Partial_Error
        iters2           # Partial_Iters
      ))            
    end
  end
  return results_df
end

end