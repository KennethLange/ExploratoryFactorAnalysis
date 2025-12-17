                                              Abstract of Submitted Paper

After more than a century, factor analysis remains one of the most popular tools in applied statistics. Exploratory factor analysis tends to be driven by iterated principal axis factorization. Confirmatory factor analysis tends to rely on maximum likelihood estimation. This paper demonstrates the computational superiority of the Gauss-Newton method in minimizing the principal axis loss. On sample problems Gauss-Newton is more than three orders of magnitude faster than popular implementations of maximum likelihood factor analysis. As part of this comparison,  we derive an alternative Gauss-Newton method that leverages the MM principle of optimization. We also explore a simple perturbation correction that dramatically improves the accuracy of the factor loading matrices derived from approximate spectral decompositions. Finally, we suggest a robust version of iterated principal axis factorization that leverages the MM principle, block descent, and the Gauss-Newton method. These innovations are implemented in Julia code and applied to a sequence of representative problems.

ExploratoryFactorAnalysis.jl is a Julia module for performing Exploratory Factor Analysis (EFA). The module implements various advanced optimization algorithms, including Majorization-Minimization (MM), Gauss-Newton (GN), corrected  Eigen-decomposition, and robust Least Absolute Deviation (LAD) regression. The two programs Uncorrected.jl and FactorAnalysis.jl, generate Tables 1 and 2 of the paper, respectively.

# 🚀 Installation
This module relies on several standard Julia packages for scientific computing. The module itself must be loaded manually into your Julia environment.

### Step 1: 
**Install Dependencies** Please ensure all necessary packages are installed by running the following commands in the Julia 
```REPL:Julia
using Pkg
Pkg.add(["LinearAlgebra", "StatsBase", "Distributions", "MultivariateStats", "KrylovKit", "Printf", "Random", "DataFrames"])
```

### Step 2: 
**Load the Module** Save the provided code as a file named ExploratoryFactorAnalysis.jl. Then load it into your Julia session or script using the command:
```REPL:Julia
using Pkg
Pkg.add(url="https://github.com/KennethLange/ExploratoryFactorAnalysis.git")
using ExploratoryFactorAnalysis
```

# 🛠️ Core Functionality and Usage
The primary goal of EFA in this module is to estimate the factor loadings matrix $\mathbf{L}$ and specific variances vector $\mathbf{d}$ corresponding to the sample covariance matrix $\mathbf{S}$. This goal is accomplished by minimizing the norm of the residual error $\mathbf{S} - \mathbf{L}\mathbf{L}^T - \text{diag}(\mathbf{d})$.

### 1. Data Generation
Use the GenerateRandomData function to create synthetic data for testing the factor analysis methods. 

**Function Description**: GenerateRandomData(n, p, r) generates factor analysis data with $n$ cases, $p$ predictors, and a rank (number of factors) $r$. The function returns the sample covariance matrix $\mathbf{S}$ and the transposed, centered data matrix $\mathbf{Y}$.

**Example**
```REPL:Julia
using LinearAlgebra, Random
n = 1000  # Number of cases
p = 50    # Number of predictors
r = 5     # Number of factors/rank
(S, Y) = GenerateRandomData(n, p, r)
```

### 2. Factor Analysis Solvers
The module provides several optimization routines for estimating the Factor Model:

2.1 **FactorAnalysisGN** solves the Factor Analysis model using Gauss-Newton (GN) updates.

```REPL:julia
(S, Y) = GenerateRandomData(n, p, r); # covariance matrix and data
(L1, d1, iters1) = FactorAnalysisGN(S, r)
println("GN      ",norm(S - L1 * L1' - Diagonal(d1)),"  ",iters1)
```

2.2 **FactorAnalysisMM** uses Majorization-Minimization (MM) coupled with Gauss-Newton-like updates

```REPL:julia
(L2, d2, iters2) = FactorAnalysisMM(S, r)
println("MM      ",norm(S - L2 * L2' - Diagonal(d2)),"  ",iters2)
```

2.3 **FactorAnalysisPartial** extracts the partial eigen-decomposition (via KrylovKit.jl) of the adjusted covariance matrix. This function is efficient for large $p$ (predictors) and small $r$ (factors).

```REPL:julia
(L3, d3, iters3) = FactorAnalysisPartial(S, r)
println("Partial ",norm(S - L3 * L3' - Diagonal(d3)),"  ",iters3)
```

2.4 **FactorAnalysisFull** computes a full eigen-decomposition of the adjusted covariance matrix. This method offers better precision at a higher computational cost for large matrices.

```REPL:julia
(L4, d4, iters4) = FactorAnalysisFull(S, r)
println("Full    ",norm(S - L4 * L4' - Diagonal(d4)),"  ",iters4)
```

2.5 **FactorAnalysisLAD** implements robust factor analysis by minimizing a least absolute deviation (LAD) loss based on the Moreau Proximal Map (ProxAbsolute).

```REPL:julia
mu = 1.0 # Moreau Constant
(L5, d5, iters5) = FactorAnalysisLAD(S, r, mu)
println("Full    ",norm(S - L5 * L5' - Diagonal(d5)),"  ",iters5)
```

2.6 **External EM/CM** algorithms

```REPL:julia
using MultivariateStats, Distributions
M1 = fit(FactorAnalysis, Y, method = :em, maxoutdim = r)
println("EM algorithm ",norm(S - projection(M1) * projection(M1)' - cov(M1)))
M2 = fit(FactorAnalysis, Y, method = :cm, maxoutdim = r)
println("CM algorithm ",norm(S - projection(M2) * projection(M2)' - cov(M2)))
``` 

### 3. Auxiliary and Core Functions

3.1 **ExploratoryPCA**: 
Computes the first $r$ principal components of the data matrix $\mathbf{X}$.

3.2 **LoadingsUpdate**: 
Projects the adjusted covariance matrix onto the closest positive semidefinite matrix of rank $r$ or less based on a partial eigen-decomposition.

3.3 **PositiveDefiniteProjection**: 
Same as above, but based on a full eigen-decomposition.

3.4 **EigenRefinement**: 
Refines approximate eigenvectors and eigenvalues delivered by partial eigen-decomposition.

3.5 **ProxAbsolute**: 
Evaluates the proximal map (soft thresholding) of the absolute value function. This function essential for LAD estimation. 

3.6 **Benchmarking and Accuracy Testing**
The module provides two utility functions to compare the performance and accuracy of the implemented solvers:FunctionPurpose

##### 4 Running the code

4.1 Run tests comparing the fit ($\| \mathbf{S} - \mathbf{L}\mathbf{L}^T - \text{diag}(\mathbf{d}) \|_F$) and iteration counts of the GN and Partial methods.
```REPL:Julia
using Pkg
Pkg.add(url="https://github.com/KennethLange/ExploratoryFactorAnalysis.git")
using ExploratoryFactorAnalysis
(r, mu) = (5, 1.0) # rank and Moreau Constant
accuracy_results  = TestsAccuracy(r, mu) # generates Table 1 of paper

4.2 Run extensive tests comparing the runtime ratios and average iterations of the GN, MM, Partial, Full, LAD, and standard EM/CM (from MultivariateStats.jl) methods across various dimensions ($p, r$).
```REPL:Julia
using Pkg
Pkg.add(url="https://github.com/KennethLange/ExploratoryFactorAnalysis.git")
using ExploratoryFactorAnalysis
mu = 1.0 # Moreau Constant
benchmark_results = TestsBenchmark(mu) # generates Table 2 of paper


