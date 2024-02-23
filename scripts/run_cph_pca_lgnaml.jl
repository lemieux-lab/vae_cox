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
#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
#DataDict = load_tcga_datasets(tcga_datasets_list)["BRCA"]
#clinical_factors = Matrix(CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame))
# clinical_factors = Matrix(tmp_df)
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
BRCA_data = load_tcga_datasets(tcga_datasets_list)["BRCA"]
DataDict = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5"))
DATA = DataDict["dataset"] 
keep1 = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep2 = [gene in BRCA_data["dataset"].genes[keep1] for gene in DATA.genes]
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
push!(lnames, "Age")
clinical_factors = hcat(CF_bin, lgn_CF[:,"Age_at_diagnosis"])

dataset_name = "LgnAML"
dim_redux_type = "PCA"
DATA = DataDict["dataset"] 
#sum(DATA.samples .== brca_CF.case_id)
#sum(DATA.surve .== brca_CF.surve)
ngenes = sum(keep2)
println("$dataset_name nb genes : $(size(DATA.genes)[1])")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")


dim_redux_size = 17
evaluate_cphdnn_pca(CDS_data, clinical_factors, dataset_name, dim_redux_size, dim_redux_type, nepochs= 5_000, cph_lr = 1e-5, cph_wd = 0.0001)
evaluate_coxridge(clinical_factors, dataset_name,  0, dim_redux_type, size(clinical_factors)[2];hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 0.0001, modeltype = "cox_ridge")

dim_redux_list = shuffle([1,2,3,4,5,10,15,20,25,50,75,100,125,150,175,200])
CDS_data = Matrix(DATA.data[:,keep2])
for dim_redux_size in dim_redux_list
    # with clinical features 
    evaluate_cphdnn_pca(CDS_data, clinical_factors, dataset_name, dim_redux_size, dim_redux_type;  nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 0.0001)
    evaluate_coxridge_pca(CDS_data, clinical_factors, dataset_name, dim_redux_size, dim_redux_type;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 0.0001, modeltype = "cox_ridge")
    # without clinical features 
    evaluate_cphdnn_pca(CDS_data, clinical_factors[:,[]], dataset_name, dim_redux_size, dim_redux_type)
    evaluate_coxridge_pca(CDS_data, clinical_factors[:,[]], dataset_name, dim_redux_size, dim_redux_type;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 0.0001, modeltype = "cox_ridge")
end