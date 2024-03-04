include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
outpath, session_id = set_dirs("RES") ;
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
TCGA_datasets = load_tcga_datasets(tcga_datasets_list);
BRCA_data = TCGA_datasets["BRCA"]
LGG_data = TCGA_datasets["LGG"]
OV_data = TCGA_datasets["OV"]

LGNAML_data = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5")) 
keep_tcga_cds = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep_lgnaml_common = [gene in BRCA_data["dataset"].genes[keep_tcga_cds] for gene in LGNAML_data["dataset"].genes];

BRCA_data["CDS"] = keep_tcga_cds
BRCA_data["CF"] = zeros(size(BRCA_data["dataset"].data)[1],0)  
LGG_data["CDS"] = keep_tcga_cds  
LGG_data["CF"] = zeros(size(LGG_data["dataset"].data)[1],0) 
OV_data["CDS"] = keep_tcga_cds  
OV_data["CF"] = zeros(size(OV_data["dataset"].data)[1],0) 

LGNAML_data["CDS"] = keep_lgnaml_common 
LGNAML_data["CF"] = zeros(size(LGNAML_data["dataset"].data)[1],0) 

### Use LGG, BRCA, AML, OV 
### Use random vs pca 
### Use CPHDNN, Cox-ridge 
### 10 replicates
nepochs = 5000
DS_list = shuffle([BRCA_data, LGNAML_data, LGG_data, OV_data])
### EVAL BRCA CDS 
ngenes = sum(BRCA_data["CDS"])
CDS_data = BRCA_data["dataset"].data[:,BRCA_data["CDS"]]
evaluate_cphdnn_rdm(BRCA_data, 8000, nepochs =nepochs, cph_wd =1e-2, dim_redux_type="STD");
evaluate_coxridge_rdm(BRCA_data, ngenes, nepochs =nepochs, cph_lr = 1e-5);
### EVAL BRCA PCA 
evaluate_cphdnn_pca(BRCA_data, size(BRCA_data["dataset"].samples)[1], nepochs=10000, cph_wd= 1e-2);
evaluate_coxridge_pca(BRCA_data, size(BRCA_data["dataset"].samples)[1], nepochs=nepochs, cph_lr = 1e-4);
### EVAL BRCA PAM50 
PAM50 = CSV.read("Data/GDC_processed/PAM50_genes_processed.csv", DataFrame)
BRCA_data["CDS"] =  [gene in PAM50[:,"alt_name"] for gene in BRCA_data["dataset"].genes];
BRCA_data["CF"] = zeros(size(BRCA_data["dataset"].data)[1],0); 
evaluate_coxridge_rdm(BRCA_data, sum(BRCA_data["CDS"]); dim_redux_type = "PAM50", nepochs =nepochs, cph_lr = 1e-3);
evaluate_cphdnn_rdm(BRCA_data, sum(BRCA_data["CDS"]); dim_redux_type = "PAM50", nepochs =nepochs);

### EVAL BRCA CLIN F 
clinical_factors = Matrix(CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame))
BRCA_data["CF"] = clinical_factors
evaluate_coxridge_rdm(BRCA_data, 0; dim_redux_type = "CLINF", nepochs =nepochs, cph_lr = 1e-3);
evaluate_cphdnn_rdm(BRCA_data, 0; dim_redux_type = "CLINF", nepochs =nepochs);


### EVAL LGNAML CDS 
ngenes = sum(LGNAML_data["CDS"])
evaluate_cphdnn_rdm(LGNAML_data, ngenes, nepochs =nepochs, cph_wd =1e-2);
evaluate_coxridge_rdm(LGNAML_data, ngenes, nepochs =nepochs, cph_lr = 1e-5);
### EVAL LGNAML PCA 
evaluate_cphdnn_pca(LGNAML_data, size(LGNAML_data["dataset"].samples)[1], nepochs=nepochs, cph_wd= 1e-2);
evaluate_coxridge_pca(LGNAML_data, size(LGNAML_data["dataset"].samples)[1], nepochs=nepochs, cph_lr = 1e-4);
### EVAL LGNAML LSC17 
LSC17 = CSV.read("Data/SIGNATURES/LSC17.csv", DataFrame);
LGNAML_data["CDS"] =  [gene in LSC17[:,"alt_name"] for gene in LGNAML_data["dataset"].genes];
ngenes = sum(LGNAML_data["CDS"])
LGNAML_data["CF"] = zeros(size(LGNAML_data["dataset"].data)[1],0) 
evaluate_coxridge_rdm(LGNAML_data, ngenes; dim_redux_type = "LSC17", nepochs =nepochs, cph_lr = 1e-3);
evaluate_cphdnn_rdm(LGNAML_data, ngenes; dim_redux_type = "LSC17", nepochs =nepochs);
### EVAL LGNAML CLIN F
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
push!(lnames, "Age")
clinical_factors = hcat(CF_bin, lgn_CF[:,"Age_at_diagnosis"])
LGNAML_data["CF"] = clinical_factors
evaluate_coxridge_rdm(LGNAML_data, 0; dim_redux_type = "CLINF", nepochs =nepochs, cph_lr = 1e-3);
evaluate_cphdnn_rdm(LGNAML_data, 0; dim_redux_type = "CLINF", nepochs =nepochs);



