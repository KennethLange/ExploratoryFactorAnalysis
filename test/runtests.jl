using ExploratoryFactorAnalysis
using Test

@testset "ExploratoryFactorAnalysis Tests" begin
    
    (r, mu) = (5, 1.0); # factors, Moreau envelope constant

    @testset "Accuracy Tests" begin
        println("\n=== Running Accuracy Tests ===")
        df = TestsAccuracy(r, mu)
        display(df)
    end

    @testset "Benchmark Tests" begin
        println("\n=== Running Benchmarks ===")
        df = TestsBenchmark(mu)
        display(df)
    end

end

