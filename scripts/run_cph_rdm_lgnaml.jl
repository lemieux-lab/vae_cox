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

#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
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

dim_redux_list = shuffle([0,1,2,3,4,5,10,15,20,25,50,75,100,125,250,375,500,1_000,1_500,2_000,2_500,3000,5_000,6000,7000,8000,9000,10_000,11_000,12000,13000,14000,15_000,16_000,17_000,18_000,19_000,ngenes])

CDS_data = Matrix(DATA.data[:,keep2])
evaluate_cphdnn(clinical_factors,dataset_name,0, dim_redux_type,size(clinical_factors)[2], nepochs=20_000, cph_wd =1e-4) 
final_data = hcat(CDS_data, clinical_factors)
evaluate_cphdnn(final_data,dataset_name,size(CDS_data)[2], dim_redux_type,size(clinical_factors)[2], nepochs=500, cph_wd =1e-2) 
evaluate_coxridge(final_data,dataset_name,size(CDS_data)[2], dim_redux_type,size(clinical_factors)[2], hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 1e-4, modeltype = "cox_ridge") 

# select and process using dim. redux
for dim_redux_size in dim_redux_list

    #dim_redux_size = dim_redux_list[3]
    
    dredux_data = Matrix(CDS_data[:,sample(collect(1:size(CDS_data)[2]), dim_redux_size, replace = false)])
    nb_clinf = size(clinical_factors)[2]
    final_data = hcat(dredux_data, clinical_factors)
    #final_data = clinical_factors
    evaluate_coxridge(final_data, dataset_name,  dim_redux_size, dim_redux_type, nb_clinf;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 1e-4, modeltype = "cox_ridge")
    evaluate_cphdnn(final_data,dataset_name,dim_redux_size, dim_redux_type,  nb_clinf )    
    if dim_redux_size != 0
        final_data = Matrix(CDS_data[:,sample(collect(1:size(CDS_data)[2]), dim_redux_size, replace = false)])
        evaluate_coxridge(final_data, dataset_name,  dim_redux_size, dim_redux_type, 0;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-3, cph_wd = 1e-4, modeltype = "cox_ridge")
        evaluate_cphdnn(final_data,dataset_name,dim_redux_size, dim_redux_type,  0)       
    end 
end
