include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
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

# LGN-AML CF (no GE) + Cox-Ridge
# LGN-AML CF (no GE) + CPH-DNN 
# LGN-AML GE (no CF) + Cox-Ridge
# LGN-AML GE (no CF) + CPH-DNN 

tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
BRCA_data = load_tcga_datasets(tcga_datasets_list)["BRCA"]
DataDict = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5"))
DATA = DataDict["dataset"] 
keep1 = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep2 = [gene in BRCA_data["dataset"].genes[keep1] for gene in DATA.genes]

### Leucegene data processing  
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
push!(lnames, "Age")
clinical_factors = hcat(CF_bin, lgn_CF[:,"Age_at_diagnosis"])
dataset_name = "LgnAML"
dim_redux_type = "RDM"
#DATA = DataDict["dataset"] 
#keep = [occursin("protein_coding", bt) for bt in DATA.biotypes]
ngenes = sum(keep2)
println("$dataset_name nb genes : $(size(DATA.genes)[1])")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")

results_df = []
CDS_data = Matrix(DATA.data[:,keep2])
cphdnn_c_ind = evaluate_cphdnn(clinical_factors,dataset_name,0, dim_redux_type,size(clinical_factors)[2], nepochs= 10_000, cph_wd =1e-2) 
coxridge_c_ind = evaluate_coxridge(clinical_factors,dataset_name,0, dim_redux_type,size(clinical_factors)[2], hlsize = 0, nepochs= 10_000, cph_nb_hl = 0, cph_lr = 1e-4, cph_wd = 1e-4, modeltype = "cox_ridge") 
push!(results_df, ("cphdnn_clinf", cphdnn_c_ind))
push!(results_df, ("coxridge_clinf", coxridge_c_ind))

final_data = hcat(CDS_data, clinical_factors)
cphdnn_c_ind = evaluate_cphdnn(final_data,dataset_name,size(CDS_data)[2], dim_redux_type,size(clinical_factors)[2], nepochs=500, cph_wd =1e-2) 
coxridge_c_ind = evaluate_coxridge(final_data,dataset_name,size(CDS_data)[2], dim_redux_type,size(clinical_factors)[2], hlsize = 0, nepochs= 10_000, cph_nb_hl = 0, cph_lr = 1e-6, cph_wd = 1e-2, modeltype = "cox_ridge") 
push!(results_df, ("cphdnn_clinf_GE", cphdnn_c_ind))
push!(results_df, ("coxridge_clinf_GE", coxridge_c_ind))

cphdnn_c_ind = evaluate_cphdnn(CDS_data,dataset_name,size(CDS_data)[2], dim_redux_type,0, nepochs=500, cph_wd =1e-2) 
coxridge_c_ind = evaluate_coxridge(CDS_data,dataset_name,size(CDS_data)[2], dim_redux_type,0, hlsize = 0, nepochs= 10_000, cph_nb_hl = 0, cph_lr = 1e-6, cph_wd = 1e-2, modeltype = "cox_ridge") 
push!(results_df, ("cphdnn_GE", cphdnn_c_ind))
push!(results_df, ("coxridge_GE", coxridge_c_ind))

models_df = DataFrame(Dict(results_df))
fig = Figure(size = (1024,512));
ax1 = Axis(fig[1,1],
    title = "LgnAML",
    xticks = (collect(1:6),[join(x[2:end],"-") for x in split.(names(models_df),"_")]))
for (i,colname) in enumerate(names(models_df))
    modeltype = split(colname,"_")[1]
    col = modeltype == "cphdnn" ? "blue" : "orange"
    boxplot!(ax1, ones(size(models_df)[1]) * i, Float32.(models_df[:,colname]), color = col, label = modeltype)
    text!(ax1, i -0.25,median(models_df[:,colname]), text= string(round(median(models_df[:,colname]), digits = 3)))
end 
axislegend(ax, position = :rb)
fig 

CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn.svg",fig)
CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn.png",fig)
CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn.pdf",fig)
DataDict = BRCA_data
DATA = BRCA_data["dataset"]
keep = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
clinical_factors = Matrix(CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame))

dataset_name = "BRCA"
dim_redux_type = "RDM"
#DATA = DataDict["dataset"] 
#keep = [occursin("protein_coding", bt) for bt in DATA.biotypes]
ngenes = sum(keep)
println("$dataset_name nb genes : $(size(DATA.genes)[1])")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")

results_df = []
CDS_data = Matrix(DATA.data[:,keep])
cphdnn_c_ind = evaluate_cphdnn(clinical_factors,dataset_name,0, dim_redux_type,size(clinical_factors)[2], nepochs= 10_000, cph_wd =1e-2) 
coxridge_c_ind = evaluate_coxridge(clinical_factors,dataset_name,0, dim_redux_type,size(clinical_factors)[2], hlsize = 0, nepochs= 10_000, cph_nb_hl = 0, cph_lr = 1e-4, cph_wd = 1e-4, modeltype = "cox_ridge") 
push!(results_df, ("cphdnn_clinf", cphdnn_c_ind));
push!(results_df, ("coxridge_clinf", coxridge_c_ind));

final_data = hcat(CDS_data, clinical_factors)
cphdnn_c_ind = evaluate_cphdnn(final_data,dataset_name,size(CDS_data)[2], dim_redux_type,size(clinical_factors)[2], nepochs=500, cph_wd =1e-2) 
coxridge_c_ind = evaluate_coxridge(final_data,dataset_name,size(CDS_data)[2], dim_redux_type,size(clinical_factors)[2], hlsize = 0, nepochs= 10_000, cph_nb_hl = 0, cph_lr = 1e-6, cph_wd = 1e-2, modeltype = "cox_ridge") 
push!(results_df, ("cphdnn_clinf_GE", cphdnn_c_ind));
push!(results_df, ("coxridge_clinf_GE", coxridge_c_ind));

cphdnn_c_ind = evaluate_cphdnn(CDS_data,dataset_name,size(CDS_data)[2], dim_redux_type,0, nepochs=500, cph_wd =1e-2) 
coxridge_c_ind = evaluate_coxridge(CDS_data,dataset_name,size(CDS_data)[2], dim_redux_type,0, hlsize = 0, nepochs= 10_000, cph_nb_hl = 0, cph_lr = 1e-6, cph_wd = 1e-2, modeltype = "cox_ridge") 
push!(results_df, ("cphdnn_GE", cphdnn_c_ind));
push!(results_df, ("coxridge_GE", coxridge_c_ind));

models_df = DataFrame(Dict(results_df));

ax2 = Axis(fig[1,2],
    title = "BRCA",
    xticks = (collect(1:6),[join(x[2:end],"-") for x in split.(names(models_df),"_")]))
for (i,colname) in enumerate(names(models_df))
    modeltype = split(colname,"_")[1]
    col = modeltype == "cphdnn" ? "blue" : "orange"
    boxplot!(ax2, ones(size(models_df)[1]) * i, Float32.(models_df[:,colname]), color = col, label = modeltype)
    text!(ax2, i -0.25,median(models_df[:,colname]), text= string(round(median(models_df[:,colname]), digits = 3)))
end 
fig 

CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn.svg",fig)
CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn.png",fig)
CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn.pdf",fig)

res_df = gather_params("RES/2024-02-21T22:27:41.302")
res_df[:,"log_insize"] .= log10.(res_df[:,"insize"])
p = AlgebraOfGraphics.data(res_df) * mapping(:nb_clinf, :cph_test_c_ind, row = :dataset, col = :model_type, color = :)
draw(p)
fig = Figure(size = (512,512));
ax1 = Axis(fig[1,1],
    title = "LgnAML",
    xticks = (collect(1:6),[join(x[2:end],"-") for x in split.(names(models_df),"_")]))
for (i,colname) in enumerate(names(models_df))
    modeltype = split(colname,"_")[1]
    col = modeltype == "cphdnn" ? "blue" : "orange"
    boxplot!(ax1, ones(size(models_df)[1]) * i, Float32.(models_df[:,colname]), color = col, label = modeltype)
    text!(ax1, i -0.25,median(models_df[:,colname]), text= string(round(median(models_df[:,colname]), digits = 3)))
end 
axislegend(ax, position = :rb)
fig 




function make_plot()
    #tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
    tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
    BRCA_data = load_tcga_datasets(tcga_datasets_list)["BRCA"]
    DataDict = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5"))
    df = gather_params("RES/")
    df = df[ismissing.(df.dim_redux_type) .== 0,:]

    fig =  Figure(size= (1024,1024));
    RDM = df[df[:,"dim_redux_type"] .== "RDM",:]
    dataset_name = "BRCA"
    data_df = RDM[RDM[:,"dataset"] .== dataset_name,:]
    data_df = sort(data_df, ["insize"])

    fig, ax1 = add_multi_scatter!(fig, 1, 1, data_df; SMA_K=10, SMA_N=10)

    dataset_name = "LgnAML"
    data_df = RDM[RDM[:,"dataset"] .== dataset_name,:]
    data_df = sort(data_df, ["insize"])

    fig, ax2 = add_multi_scatter!(fig, 2, 1, data_df; SMA_K=10, SMA_N=10)

    Pca = df[df[:, "dim_redux_type"] .== "PCA",:]
    dataset_name = "BRCA"
    data_df = Pca[Pca[:,"dataset"] .== dataset_name,:]
    data_df = sort(data_df, ["insize"])

    fig, ax3 = add_multi_scatter!(fig, 1, 2, data_df; SMA_K=10, SMA_N=10)

    dataset_name = "BRCA"
    data_df = Pca[Pca[:,"dataset"] .== dataset_name,:]
    data_df = sort(data_df, ["insize"])

    fig, ax4 = add_multi_scatter!(fig, 2, 2, data_df; SMA_K=10, SMA_N=10)

    axislegend(ax3, position = :rb)    

    CairoMakie.save("figures/dimension_sweep_lgnaml_brca_coxridge_cphdnn.svg",fig)
    CairoMakie.save("figures/dimension_sweep_lgnaml_brca_coxridge_cphdnn.png",fig)
    CairoMakie.save("figures/dimension_sweep_lgnaml_brca_coxridge_cphdnn.pdf",fig)

    return fig
end 
