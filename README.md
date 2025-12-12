                                              Abstract of Submitted Paper

After more than a century, factor analysis remains one of the most popular tools in applied statistics. Exploratory factor analysis tends to be driven by iterated principal axis factorization. Confirmatory factor analysis tends to rely on maximum likelihood estimation. This paper demonstrates the computational superiority of the Gauss-Newton method in minimizing the principal axis loss. On sample problems Gauss-Newton is more than three orders of magnitude faster than popular implementations of maximum likelihood factor analysis. As part of this comparison,  we derive an alternative Gauss-Newton method that leverages the MM principle of optimization. We also explore a simple perturbation correction that dramatically improves the accuracy of the factor loading matrices derived from approximate spectral decompositions. Finally, we suggest a robust version of iterated principal axis factorization that leverages the MM principle, block descent, and the Gauss-Newton method. These innovations are implemented in Julia code and applied to a sequence of representative problems.

The two programs,Uncorrected.jl and FactorAnalysis.jl, generate Tables 1 and 2 of the submitted paper, respectively.

ExploratoryFactorAnalysis.jl is a Julia module designed for performing Exploratory Factor Analysis (EFA). It implements various advanced optimization algorithms, including Majorization-Minimization (MM), Gauss-Newton (GN), exact Eigendecomposition-based updates, and a robust approach using the Least Absolute Deviation (LAD) loss.


# 🚀 Installation
This module relies on several standard Julia packages for scientific computing. 
Since this is provided as a single module file, you will need to load it manually into your Julia environment.

### Step 1: 
**Install Dependencies** Please ensure all necessary packages are installed by running the following commands in the Julia 
```REPL:Julia
using Pkg
Pkg.add(["LinearAlgebra", "StatsBase", "Distributions", "MultivariateStats", "KrylovKit", "Printf", "Random", "DataFrames"])
```

### Step 2: 
**Load the Module** Save the provided code as a file named ExploratoryFactorAnalysis.jl. Then, load it into your Julia session or script using the include command:
```REPL:Julia
Pkg.add(url="https://github.com/KennethLange/ExploratoryFactorAnalysis.git")
using ExploratoryFactorAnalysis
using LinearAlgebra, Random, MultivariateStats, Distributions
```

# 🛠️ Core Functionality and Usage
The primary goal of EFA in this module is to estimate the factor loadings matrix $\mathbf{L}$ and specific variances vector $\mathbf{d}$ from the sample covariance matrix $\mathbf{S}$, minimizing the error such that $\mathbf{S} \approx \mathbf{L}\mathbf{L}^T + \text{diag}(\mathbf{d})$.

### 1. Data Generation
Use the GenerateRandomData function to create synthetic data for testing the factor analysis methods. 

**Function Description**: GenerateRandomData(n, p, r) generates factor analysis data with $n$ cases, $p$ predictors, and an inherent rank (number of factors) of $r$. Returns the sample covariance matrix $\mathbf{S}$ and the transposed, centered data matrix $\mathbf{Y}$.

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

2.3 **FactorAnalysisPartial** uses iterative updates based on Partial Eigendecomposition (via KrylovKit.jl) of the adjusted covariance matrix.Efficient for large $p$ (predictors) and small $r$ (factors).

```REPL:julia
(L3, d3, iters3) = FactorAnalysisPartial(S, r)
println("Partial ",norm(S - L3 * L3' - Diagonal(d3)),"  ",iters3)
```

2.4 **FactorAnalysisFull** iterative updates based on Full Eigendecomposition of the adjusted covariance matrix.High precision, but higher computational cost for large matrices.

```REPL:julia
(L4, d4, iters4) = FactorAnalysisFull(S, r)
println("Full    ",norm(S - L4 * L4' - Diagonal(d4)),"  ",iters4)
```

2.5 **FactorAnalysisLAD** uses robust factor analysis by minimizing a least absolute deviation (LAD) loss, using the Moreau Proximal Map (ProxAbsolute) for approximation

```REPL:julia
mu = 1.0 # Moreau Constant
(L5, d5, iters5) = FactorAnalysisLAD(S, r, mu)
println("Full    ",norm(S - L5 * L5' - Diagonal(d5)),"  ",iters5)
```

2.6 **External EM/CM** 

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
Projects the adjusted covariance matrix onto the closest positive semidefinite matrix of rank $r$ or less, using partial eigendecomposition.

3.3 **PositiveDefiniteProjection**: 
Same as above, but uses full eigendecomposition.

3.4 **EigenRefinement**: 
Refines approximate eigenvectors and eigenvalues, improving the accuracy of partial eigendecomposition results.

3.5 **ProxAbsolute**: 
Evaluates the proximal map (soft thresholding) of the absolute value, essential for the LAD loss function.4. 

3.6 **Benchmarking and Accuracy Testing**
The module provides two utility functions to compare the performance and accuracy of the implemented solvers:FunctionPurpose

##### 3.6.1 TestsAccuracy(r, mu)
Runs tests comparing the model fit error (e.g., $\| \mathbf{S} - \mathbf{L}\mathbf{L}^T - \text{diag}(\mathbf{d}) \|$) and iteration counts of the GN and Partial methods.
```REPL:Julia
(r, mu) = (5, 1.0) # Moreau Constant
accuracy_results  = TestsAccuracy(r, mu)
```

```markdown
```console
$ REPL:Julia
julia> TestsAccuracy(5, 1.0)
(n, p)       | GN                        | Iters  | Partial                   | Iters 
------------------------------------------------------------------------------------------
(500, 6)     | 1.8735076413684065e-8     | 74     | 4.0939445820677676e-8     | 105   
(500, 10)    | 0.11338150668913886       | 500    | 0.11338097682528249       | 500   
(500, 25)    | 0.47112005821092495       | 54     | 0.47112005821092384       | 55    
(500, 50)    | 0.9377191041250191        | 28     | 0.9377191041250214        | 26    
(500, 100)   | 2.154818068360006         | 19     | 2.1548180683600084        | 17    
(500, 250)   | 5.200811636827077         | 14     | 5.200811636827078         | 12    
(500, 500)   | 10.639303361606174        | 13     | 10.63930336160618         | 10   
```

##### 3.6.2 TestsBenchmark(mu)
Runs extensive tests comparing the runtime ratios and average iterations of the GN, MM, Partial, Full, LAD, and standard EM/CM (from MultivariateStats.jl) methods across various dimensions ($p, r$).
```REPL:Julia
mu = 1.0 # Moreau Constant
benchmark_results = TestsBenchmark(mu)
```
