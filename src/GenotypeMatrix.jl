using SnpArrays, DataFrames, LinearAlgebra, MultivariateStats, StatsBase 

"""Performs rank r factor analysis on S by MM Gauss-Newton updates."""
function FactorAnalysisMM(S::Matrix{T}, r::Int) where T <: Real
  (p, iters) = (size(S, 1), 0) # size of sample covariance matrix
  (d, old_d, conv) = (zeros(T, p), zeros(T, p), 1.0e-8)
  F = randn(p, r) # random initialization of loadings
  FtF = zeros(T, r, r) # stores F transpose * F
  for iter = 1:1000
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(F[i, :])^2, zero(T))
    end
    mul!(FtF, transpose(F), F) # prepare to update loadings
    F .= F .+ (S * F - F * FtF - Diagonal(d) * F) / (2FtF)
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  SVD = svd(F) # orthogonalize the columns of F
  F = SVD.U * Diagonal(SVD.S)
  return (F, d, iters)
end

"""Performs rank r factor analysis on S by standard Gauss-Newton 
updates."""
function FactorAnalysisGN(S::Matrix{T}, r::Int) where T <: Real
  (p, iters) = (size(S, 1), 0) # size of sample covariance matrix
  (d, old_d, conv, ep) = (zeros(T, p), zeros(T, p), 1.0e-8, 10.0)
  F = randn(T, p, r) # random initialization of loadings
  FtF = zeros(T, r, r) # stores F transpose * F
  Y = zeros(T, r, r)
  Z = zeros(T, p, r)
  for iter = 1:1000
    iters = iters + 1
    for i = 1:p # update specific variances
      d[i] = max(S[i, i] - norm(F[i, :])^2, zero(T))
    end
    mul!(FtF, transpose(F), F) # prepare to update loadings
    Y = F / FtF
    mul!(Z, S, Y)
    Z .= Z .- Diagonal(d) * Y
    F .= Z .- F * ((transpose(Y) * Z - I) ./ 2) # loadings update
    if norm(d - old_d) < conv * norm(old_d) && iter > 1 break end
    old_d .= d
  end
  SVD = svd(F) # orthogonalize the columns of F
  F = SVD.U * Diagonal(SVD.S)
  return (F, d, iters)
end

# Fetch path to packaged example dataset.
bed_file = SnpArrays.datadir("EUR_subset.bed")
# bed_file = SnpArrays.datadir("mouse.bed") 

# Parse the files into a compressed SnpArray object with
# rows as people and columns as SNP genotypes.
snp_data = SnpArray(bed_file);

# Compute the genotype relationship matrix
G = grm(snp_data, method = :GRM);

r = 5; # number of factors
@time M1 = fit(FactorAnalysis, G, method = :em, maxoutdim = r);
println("EM algorithm ",norm(G - projection(M1) * projection(M1)' - cov(M1)))
#@time M2 = fit(FactorAnalysis, G, method = :cm, maxoutdim = r)
#println("CM algorithm ",norm(G - projection(M2) * projection(M2)' - cov(M2)))
for i = 1:10
  @time (F, d, iters) = FactorAnalysisMM(G, r);
  println(i," ","MM algorithm ",iters," ",norm(G - F * F' - Diagonal(d)))
end
for i = 1:10
  @time (F, d, iters) = FactorAnalysisGN(G, r);
  println(i," ","GN algorithm ",iters," ",norm(G - F * F' - Diagonal(d)))
end



