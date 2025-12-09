using LinearAlgebra, StatsBase, Distributions, Arpack, MultivariateStats

"""Projects the symmetric matrix S onto the closest positive 
semidefinite matrix of rank r or less by partial eigen-decomposition."""
function LoadingsUpdate(S::Matrix{T}, r::Int) where T <: Real
  (lambda, V) = eigs(S, r, which=:LR) # Arpack call
  V = real(V) # convert to real
  lambda = real(lambda)
#   (lambda, V) = EigenRefinement(S, V)
  D = Diagonal(sqrt.(max.(lambda[1:r], zero(T))))
  return V[:, 1:r] * D
end

"""Performs rank r factor analysis on S by Gauss-Newton updates."""
function FactorAnalysisGN(S::Matrix{T}, r::Int) where T <: Real
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
    L = L + (S * L - L * LtL - Diagonal(d) * L) / (2LtL)
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
function FactorAnalysisPartial(S::Matrix{T}, r::Int) where T <: Real
  (p, iters) = (size(S, 1), 0)
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  L = randn(p, r) # random factor loadings
#   println(0,"  ",norm(S - L * L' - Diagonal(d)))
  for iter = 1:100
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(L[i, :])^2, zero(T))
    end
    for i = 1:p # adjust diagonal sample variances
      S[i, i] = S[i, i] - d[i]
    end
    L = LoadingsUpdate(S, r) # update loadings
    for i = 1:p # restore sample variances
      S[i, i] = S[i, i] + d[i]
    end 
#     println(iter,"  ",norm(S - L * L' - Diagonal(d)))
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  return (L, d, iters)
end

"""Generates random factor analysis data."""
function GenerateRandomData(n, p, r)
  avg = randn(p) # means
  L = randn(p, r) # factor loading
  d = rand(p) # specific variances
  Cov = L * L' + Diagonal(d) # overall covariance
  dist = MvNormal(avg, Cov) # multivariate distribution
  X = zeros(n, p)
  X .= rand(dist, n)' # random Gaussian sample
  S = cov(X) # sample covariance matrix
  Y = (X .- mean(X, dims = 2))'  
  return (S, Y)
end

"""Runs test problems of rank r and Moreau constant mu."""
function RunTests(r, mu)
  for n in [500] # cases
    for p in [6, 10, 25, 50, 100, 250, 500] # predictors
      println("(n ,p) = ",(n, p))
      (S, Y) = GenerateRandomData(n, p, r); # covariance matrix and data
      (L1, d1, iters1) = FactorAnalysisGN(S, r);
      println("GN      ",norm(S - L1 * L1' - Diagonal(d1)),"  ",iters1)
      (L2, d2, iters2) = FactorAnalysisPartial(S, r);
      println("Partial ",norm(S - L2 * L2' - Diagonal(d2)),"  ",iters1)
    end
  end
end

(r, mu) = (5, 1.0); # factors, Moreau envelope constant
RunTests(r, mu)
