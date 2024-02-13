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
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
function numerise_labels(DF, colnames)
    MMs = []
    level_names = []
    for colname in colnames
        levels = unique(DF[:, colname])
        if length(levels) > 2
            levels_dict = Dict([(lab, idx) for (idx, lab) in enumerate(levels)])
            MM = zeros(size(DF)[1], size(levels)[1])
            for level in levels
                MM[DF[:,colname].==level, levels_dict[level]] .= 1
            end
            push!(level_names, ["$(colname)_$(level)" for level in levels])
        else 
            MM = zeros(size(DF)[1],1)
            MM[DF[:,colname] .== levels[2]] .= 1
            push!(level_names, "$(colname)_$(levels[2])")
        end 

        push!(MMs, MM)
        
    end 
    return hcat(MMs...), vcat(level_names...) 
end
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
push!(lnames, "Age")
clinical_factors = hcat(CF_bin, lgn_CF[:,"Age_at_diagnosis"])
# DataDict = BRCA_data

# dataset_name = "BRCA"
dataset_name = "LgnAML"
dim_redux_type = "RDM"
DATA = DataDict["dataset"] 

# keep = [occursin("protein_coding", bt) for bt in DATA.biotypes]
# ngenes = sum(keep)
println("$dataset_name nb genes : $(size(DATA.genes)[1])")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")
dim_redux_list = [0,1,2,3,4,5,10,15,20,25,50,75,100,125,250,375,500,625,ngenes]
hlsize = 512
cph_nb_hl, modeltype, nepochs= 2,  "cphdnn", 5000
# for dim size in list do 
# select and process using dim. redux
LSC17 = CSV.read("Data/SIGNATURES/LSC17.csv", DataFrame)
keep = [x in LSC17[:,"alt_name"] for x in DATA.genes ]
keep = [occursin("protein_coding",x) for x in DATA.biotypes]
dredux_data = Matrix(DATA.data[:,keep])
dim_redux_size =0# size(dredux_data)[2]
nb_clinf = size(clinical_factors)[2]
final_data = hcat(dredux_data, clinical_factors)
final_data = clinical_factors

base_params = Dict(
        ## run infos 
        "session_id" => session_id, "nfolds" =>5, "model_id" =>  "$(bytes2hex(sha256("$(now())$(dataset_name)"))[1:Int(floor(end/3))])",
        "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
        "dim_redux_type"=>dim_redux_type, "dataset" => dataset_name,
        ## optim infos 
        "nepochs" => nepochs, "ae_lr" =>0, "cph_lr" => 1e-6, "ae_wd" => 0, "cph_wd" => 0.1,
        ## model infos
        "model_type"=> modeltype, "ae_nb_hls" => 2,
        "enc_nb_hl" => 0, "enc_hl_size"=> 0,
        "venc_nb_hl" => 0, "venc_hl_size"=> 0,  "dec_nb_hl" => 0 , "dec_hl_size"=> 0,
        "cph_nb_hl" => cph_nb_hl, "cph_hl_size" => hlsize
        ## metrics
)

# init params dict
DataDict["params"] = deepcopy(base_params)
DataDict["params"]["dim_redux"] = dim_redux_size
DataDict["params"]["nb_clinf"] = nb_clinf
DataDict["params"]["dataset"] = dataset_name
DataDict["params"]["nsamples"] = size(DATA.samples)[1]
DataDict["params"]["nsamples_test"] = Int(round(size(DATA.samples)[1] / base_params["nfolds"]))
DataDict["params"]["ngenes"] = size(DATA.genes[keep])[1]
DataDict["params"]["nsamples_train"] = size(DATA.samples)[1] - Int(round(size(DATA.samples)[1] / base_params["nfolds"]))
DataDict["params"]["model_title"] = "$(dataset_name)_$(modeltype)_$(dim_redux_type)_$(nb_clinf)CF"
DataDict["params"]["insize"] = DataDict["params"]["dim_redux"] + DataDict["params"]["nb_clinf"]
# split folds 
folds = split_train_test(final_data, DATA.survt, DATA.surve, DATA.samples;nfolds =DataDict["params"]["nfolds"])
OUTS_TST = []
Y_T_TST = []
Y_E_TST = []
# for fold in folds do train
for fold in folds
    test_samples = DATA.samples[fold["test_ids"]]
    train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
    DataDict["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
    "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)

    ## init model 
    cph_opt = Flux.ADAM(base_params["cph_lr"]) 
    cph_wd = base_params["cph_wd"]
    cphdnn = gpu(Chain(Dense(DataDict["params"]["insize"],base_params["cph_hl_size"], leakyrelu), 
    Dense(base_params["cph_hl_size"],base_params["cph_hl_size"], leakyrelu), 
    Dense(base_params["cph_hl_size"], 1, bias = false)))
    # cphdnn = gpu(Chain(Dense(DataDict["params"]["insize"], 1, bias = false)))
    for i in 1:DataDict["params"]["nepochs"]
        cph_ps = Flux.params(cphdnn)
        cph_gs = gradient(cph_ps) do 
            cox_nll_vec(cphdnn, DataDict["data_prep"]["train_x"], DataDict["data_prep"]["train_y_e"], DataDict["data_prep"]["NE_frac_tr"]) + l2_penalty(cphdnn) * DataDict["params"]["cph_wd"] 
        end 
        #meta_eval(cphdnn, tcga_datasets, base_params, verbose = i, verbose_step = 1)
        OUTS_tst = cphdnn(DataDict["data_prep"]["test_x"])
        OUTS_tr = cphdnn(DataDict["data_prep"]["train_x"])
        cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DataDict["data_prep"]["test_y_t"], DataDict["data_prep"]["test_y_e"], -1 * OUTS_tst)
        cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DataDict["data_prep"]["train_y_t"],DataDict["data_prep"]["train_y_e"], -1 * OUTS_tr)
        DataDict["params"]["cph_train_c_ind"] = cind_tr
        DataDict["params"]["cph_tst_c_ind"] = cind_test
        DataDict["params"]["step"] = i
        
        Flux.update!(cph_opt,cph_ps, cph_gs)
        if i % 1000 ==  0 || i == 1
            println("FOLD $(fold["foldn"]) $dataset_name - $i : TRAIN c-ind: $(round(cind_tr, digits = 3))\tTEST c-ind: $(round(cind_test,digits =5))")
        end 
    end
    push!(OUTS_TST, vec(cpu(cphdnn(DataDict["data_prep"]["test_x"]))))
    push!(Y_T_TST, vec(cpu(DataDict["data_prep"]["test_y_t"])))
    push!(Y_E_TST, vec(cpu(DataDict["data_prep"]["test_y_e"])))
end 
OUTS = vcat(OUTS_TST...);
Y_T = vcat(Y_T_TST...);
Y_E = vcat(Y_E_TST...);
c_inds = [];
bootstrap_n = 1000
for i in 1:bootstrap_n
    sample_ids = sample(collect(1:size(OUTS)[1]), size(OUTS)[1], replace = true)
    cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(Y_T[sample_ids],Y_E[sample_ids], -1 * OUTS[sample_ids])
    push!(c_inds, cind_test)
end 
sorted_cinds = sort(c_inds)
lo_ci = round(sorted_cinds[Int(floor(bootstrap_n * 0.025 ))], digits = 3)
up_ci = round(sorted_cinds[Int(floor(bootstrap_n * 0.975 ))], digits = 3)
med_c_ind = round(median(sorted_cinds), digits = 3)
println("TEST bootstrap c-index : $(med_c_ind) ($up_ci - $lo_ci 95% CI)")
# merge results, obtain risk scores, test data.
OUTS_TST[2]