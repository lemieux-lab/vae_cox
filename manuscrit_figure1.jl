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
function SMA(DATA_X, DATA_Y; n=10, k=10)
    means_y = []
    means_x = []
    Y_infos =  DATA_Y[sortperm(DATA_X)]
    X_infos = sort(DATA_X)
    for X_id in 1:Int(ceil(length(DATA_X) / n )):length(DATA_X)
        x_id_min = max(X_id - k, 1)
        x_id_max = min(X_id + k, length(DATA_X))
        sma = mean(Y_infos[x_id_min:x_id_max])
        push!(means_y, sma)
        push!(means_x, X_infos[X_id])
    end
    return float.(means_x),float.(means_y)
end 

#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
BRCA_data = load_tcga_datasets(tcga_datasets_list)["BRCA"]
DataDict = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5"))
df = gather_params("RES/")
df = df[ismissing.(df.dim_redux_type) .== 0,:]

fig =  Figure(size= (1024,1024));
dataset_name = "BRCA"
data_df = df[df[:,"dataset"] .== dataset_name,:]
data_df = sort(data_df, ["insize"])
ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
ax1 = Axis(fig[1,1],
    xticks = (log10.(ticks),string.(ticks)),
    #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
    title = "Survival prediction in $dataset_name with CPH-DNN & COX-ridge \n by input size with random and directed dimensionality reductions",
    xlabel = "Input size",
    ylabel = "concordance index",
    limits = (nothing, nothing, 0.4, 0.75))
lines!(ax1,log10.(ticks[[1,end]]),[0.5,0.5],linetype = "dashed")
RDM = data_df[data_df[:, "dim_redux_type"] .== "RDM",:]

cphdnn = RDM[RDM[:,"model_type"] .== "cphdnn",:]
coxridge = RDM[RDM[:,"model_type"] .== "cox_ridge",:]

## BRCA DNN CLINF + GE RDM DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .!= 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax1, Xs, Ys, color= "blue", alpha = 0.5)
lines!(ax1, SMA(Xs, Ys, k=7)..., color = "blue", label = "CPHDNN 16 CF", linewidth=3, linestyle =:dash)

## BRCA DNN NO CLINF - GE RDM DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .== 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax1, Xs, Ys, color = "blue")
lines!(ax1, SMA(Xs, Ys, k=7)..., color = "blue", label = "CPHDNN 0 CF", linewidth=3)

## BRCA RIDGE-CPH CLINF + GE RDM DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax1, Xs, Ys, color = "orange", alpha = 0.5)
lines!(ax1, SMA(Xs, Ys, k=7)..., color = "orange", label = "Cox-ridge 16 CF", linewidth=3, linestyle=:dash)


## BRCA RIDGE-CPH NO CLINF - GE RDM DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax1, Xs, Ys, color = "orange")
lines!(ax1, SMA(Xs, Ys, k=7)..., color = "orange", label = "Cox-ridge 0 CF", linewidth=3)

Pca = data_df[data_df[:, "dim_redux_type"] .== "PCA",:]
cphdnn = Pca[Pca[:,"model_type"] .== "cphdnn",:]
coxridge = Pca[Pca[:,"model_type"] .== "cox_ridge",:]
ticks = [minimum(Pca[:,"insize"]),100,1000,maximum(Pca[:,"insize"])]
ax3 = Axis(fig[1,2],
    xticks = (log10.(ticks),string.(ticks)),
    #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
    title = "Survival prediction in $dataset_name with CPH-DNN & COX-ridge \n by input size with directed dimensionality reductions (PCA / UMAP)",
    xlabel = "Input size",
    ylabel = "concordance index",
    limits = (nothing, nothing, 0.4, 0.75))
lines!(ax3,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)

## BRCA DNN CLINF + GE PCA DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .!= 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax3, Xs, Ys, color= "blue", alpha = 0.5)
lines!(ax3, SMA(Xs, Ys, k=7)..., color = "blue", label = "CPHDNN 16 CF", linewidth=3, linestyle=:dash)

## BRCA DNN CLINF + GE PCA DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .== 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax3, Xs, Ys, color= "blue")
lines!(ax3, SMA(Xs, Ys, k=7)..., color = "blue", label = "CPHDNN 0 CF", linewidth=3)

## BRCA Cox-Rdige NO CLINF - GE PCA DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax3, Xs, Ys, color = "orange", alpha=0.5)
lines!(ax3, SMA(Xs, Ys, k=7)..., color = "orange", label = "Cox-ridge 16 CF", linewidth=3, linestyle=:dash)

## BRCA DNN NO CLINF - GE PCA DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax3, Xs, Ys, color= "orange")
lines!(ax3, SMA(Xs, Ys, k=7)..., color = "orange", label = "Cox-ridge 0 CF",linewidth=3)

axislegend(ax3, position = :rb)    
fig
# scatter!(ax1,log10.(coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], label = "Cox-ridge 8 CF")
# scatter!(ax1,log10.(coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"], label = "Cox-ridge 0 CF")
# fig
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
    ylabel = "concordance index",
    limits = (nothing, nothing, 0.35, 0.75))
lines!(ax2,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)
RDM = data_df[data_df[:, "dim_redux_type"] .== "RDM",:]

cphdnn = RDM[RDM[:,"model_type"] .== "cphdnn",:]
coxridge = RDM[RDM[:,"model_type"] .== "cox_ridge",:]

## LGN CPHDNN CLINF + GE DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .!= 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax2, Xs, Ys, color= "blue", alpha =0.5)
lines!(ax2, SMA(Xs, Ys)..., color = "blue", label = "CPHDNN 8 CF", linewidth=3, linestyle=:dash)

## LGN CPHDNN NO CLINF - GE DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .== 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax2, Xs, Ys, color = "blue")
lines!(ax2, SMA(Xs, Ys)..., color = "blue", label = "CPHDNN 0 CF", linewidth=3)

## LGN Cox-ridge CLINF + GE DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax2, Xs, Ys, color= "orange", alpha = 0.5)
lines!(ax2, SMA(Xs, Ys)..., color = "orange", label = "Cox-ridge 8 CF", linewidth=3, linestyle=:dash)

## LGN Cox-ridge NO CLINF + GE DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax2, Xs, Ys, color= "orange")
lines!(ax2, SMA(Xs, Ys)..., color = "orange", label = "Cox-ridge 0 CF", linewidth=3)

fig
Pca = data_df[data_df[:, "dim_redux_type"] .== "PCA",:]
cphdnn = Pca[Pca[:,"model_type"] .== "cphdnn",:]
coxridge = Pca[Pca[:,"model_type"] .== "cox_ridge",:]
ticks = [minimum(Pca[:,"insize"]),100,1000,maximum(Pca[:,"insize"])]
ax4 = Axis(fig[2,2],
    xticks = (log10.(ticks),string.(ticks)),
    #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
    title = "Survival prediction in $dataset_name with CPH-DNN & COX-ridge \n by input size with directed dimensionality reductions (PCA / UMAP)",
    xlabel = "Input size",
    ylabel = "concordance index",
    limits = (nothing, nothing, 0.35, 0.75))
lines!(ax4,log10.(ticks[[1,end]]),[0.5,0.5],linestyle = :dash)
fig
## BRCA DNN CLINF + GE PCA DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .!= 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax4, Xs, Ys, color= "blue", alpha = 0.5)
lines!(ax4, SMA(Xs, Ys, k=7)..., color = "blue", label = "CPHDNN 16 CF", linewidth=3, linestyle=:dash)
fig
## BRCA DNN CLINF + GE PCA DIM SWEEP  
Xs, Ys = log10.(cphdnn[cphdnn[:,"nb_clinf"] .== 0 ,"insize"]), cphdnn[cphdnn[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax4, Xs, Ys, color= "blue")
lines!(ax4, SMA(Xs, Ys, k=7)..., color = "blue", label = "CPHDNN 0 CF", linewidth=3)

## BRCA Cox-Rdige NO CLINF - GE PCA DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"]
scatter!(ax4, Xs, Ys, color = "orange", alpha=0.5)
lines!(ax4, SMA(Xs, Ys, k=7)..., color = "orange",label = "Cox-ridge 16 CF", linewidth=3, linestyle=:dash)

## BRCA DNN NO CLINF - GE PCA DIM SWEEP  
Xs, Ys = log10.(coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"]), coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"]
scatter!(ax4, Xs, Ys, color= "orange")
lines!(ax4, SMA(Xs, Ys, k=7)..., color = "orange", label = "Cox-ridge 0 CF", linewidth=3)

axislegend(ax4, position = :rb)  
fig

CairoMakie.save("figures/dimension_sweep_lgnaml_brca_coxridge_cphdnn.svg",fig)
CairoMakie.save("figures/dimension_sweep_lgnaml_brca_coxridge_cphdnn.png",fig)
CairoMakie.save("figures/dimension_sweep_lgnaml_brca_coxridge_cphdnn.pdf",fig)

fig