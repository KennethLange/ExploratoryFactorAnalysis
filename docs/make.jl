using ExploratoryFactorAnalysis
using Documenter

DocMeta.setdocmeta!(ExploratoryFactorAnalysis, :DocTestSetup, :(using ExploratoryFactorAnalysis); recursive=true)

makedocs(;
    modules=[ExploratoryFactorAnalysis],
    authors="Xunjian-Li <11930699@mail.sustech.edu.cn> and contributors",
    sitename="ExploratoryFactorAnalysis.jl",
    format=Documenter.HTML(;
        canonical="https://Xunjian-Li.github.io/ExploratoryFactorAnalysis.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Xunjian-Li/ExploratoryFactorAnalysis.jl",
    devbranch="main",
)
