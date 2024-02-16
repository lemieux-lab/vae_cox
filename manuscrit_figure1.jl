include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
outpath, session_id = set_dirs() ;
## import Leucegene, BRCA

##### Different datasets - dim-redux.
## Leucegene AML / BRCA - RDM / LSC17 / PCA 

##### Explored Features 
## CLINF ON/OFF
## Modeltype DNN/Cox-ridge
## Dim redux size 0-ngenes

##### HOW 
## 5-fold cross-validated c-index (95% CI)
## Learning curves => overfitting
## Survival curves

#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
BRCA_data = load_tcga_datasets(tcga_datasets_list)["BRCA"]
DataDict = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5"))
df = gather_params("RES/")
df = df[ismissing.(df.dim_redux_type) .== 0,:]

fig =  Figure(size= (612,1024));
dataset_name = "BRCA"
data_df = df[df[:,"dataset"] .== dataset_name,:]
data_df = sort(data_df, ["insize"])
ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
ax1 = Axis(fig[1,1],
    xticks = (log10.(ticks),string.(ticks)),
    #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
    title = "Survival prediction in $dataset_name with CPH-DNN & COX-ridge \n by input size with random and directed dimensionality reductions",
    xlabel = "Input size",
    ylabel = "concordance index")
    #limits = (nothing, nothing, 0.45, 0.75))
lines!(ax1,log10.(ticks[[1,end]]),[0.5,0.5],linetype = "dashed")
scatter!(ax1,log10.(data_df[data_df[:,"nb_clinf"] .!= 0 ,"insize"]),data_df[data_df[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], label = "CPHDNN 8 CF")
scatter!(ax1,log10.(data_df[data_df[:,"nb_clinf"] .== 0 ,"insize"]),data_df[data_df[:,"nb_clinf"] .== 0,"cph_test_c_ind"], label = "CPHDNN No CF")
axislegend(ax1, position = :rb)

dataset_name = "LgnAML"
data_df = df[df[:,"dataset"] .== dataset_name,:]
data_df = sort(data_df, ["insize"])
dataset_name = unique(data_df[:,"dataset"])[1]
ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
ax2 = Axis(fig[2,1],
    xticks = (log10.(ticks),string.(ticks)),
    #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
    title = "Survival prediction in $dataset_name with CPH-DNN & COX-ridge \n by input size with random and directed dimensionality reductions",
    xlabel = "Input size",
    ylabel = "concordance index")
    #limits = (nothing, nothing, 0.45, 0.75))
lines!(ax2,log10.(ticks[[1,end]]),[0.5,0.5],linetype = "dashed")
scatter!(ax2,log10.(data_df[data_df[:,"nb_clinf"] .!= 0 ,"insize"]),data_df[data_df[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], label = "CPHDNN 8 CF")
scatter!(ax2,log10.(data_df[data_df[:,"nb_clinf"] .== 0 ,"insize"]),data_df[data_df[:,"nb_clinf"] .== 0,"cph_test_c_ind"], label = "CPHDNN No CF")
axislegend(ax2, position = :rb)

CairoMakie.save("figures/dimension_sweep_lgnaml_coxridge_cphdnn.svg",fig)
CairoMakie.save("figures/dimension_sweep_lgnaml_coxridge_cphdnn.png",fig)
CairoMakie.save("figures/dimension_sweep_lgnaml_coxridge_cphdnn.pdf",fig)

fig