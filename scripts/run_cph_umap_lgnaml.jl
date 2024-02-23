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
DATA = DataDict["dataset"] 
keep1 = [occursin("protein_coding", bt) for bt in BRCA_data["dataset"].biotypes]
keep2 = [gene in BRCA_data["dataset"].genes[keep1] for gene in DATA.genes]
lgn_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
CF_bin, lnames = numerise_labels(lgn_CF, ["Sex","Cytogenetic risk", "NPM1 mutation", "IDH1-R132 mutation", "FLT3-ITD mutation", ])
push!(lnames, "Age")
clinical_factors = hcat(CF_bin, lgn_CF[:,"Age_at_diagnosis"])

dataset_name = "LgnAML"
dim_redux_type = "UMAP"
DATA = DataDict["dataset"] 
#sum(DATA.samples .== brca_CF.case_id)
#sum(DATA.surve .== brca_CF.surve)
keep = [occursin("protein_coding", bt) for bt in DATA.biotypes]
ngenes = sum(keep)
println("$dataset_name nb genes : $(size(DATA.genes)[1])")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")

dim_redux_list = shuffle([2,4,8,16,32,64,128,200])
CDS_data = Matrix(DATA.data[:,keep])
    
# folds = split_train_test(CDS_data, DATA.survt, DATA.surve, DATA.samples;nfolds =5)
# train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(folds[1])
# Umap_model = UMAP_(cpu(train_x), dim_redux_size)        
# X_tr_umap = UMAP.transform(Umap_model, cpu(train_x))
# X_tst_umap = UMAP.transform(Umap_model, cpu(test_x))

# @time Umap_model = UMAP_(cpu(train_x), 200);        
# X_tr_umap = UMAP.transform(Umap_model, cpu(train_x))
# X_tst_umap = UMAP.transform(Umap_model, cpu(test_x))


# ttr_x = gpu(Matrix(hcat(X_tr_umap', clinical_factors[folds[1]["train_ids"],:])'))
# ttst_x = gpu(Matrix(hcat(X_tst_umap', clinical_factors[folds[1]["test_ids"],:])'))

# test_x = gpu(Matrix(hcat(predict(M, cpu(test_x))', clinical_factors[fold["test_ids"],:])'))

# folds[1]
function evaluate_cphdnn_umap(CDS_data,clinical_factors,dataset_name,dim_redux_size, dim_redux_type;hlsize = 512, nepochs= 5_000, cph_nb_hl = 2, cph_lr = 1e-6, cph_wd = 0.1,  modeltype = "cphdnn")
    nb_clinf = size(clinical_factors)[2]

    base_params = Dict(
            ## run infos 
            "session_id" => session_id, "nfolds" =>5, "modelid" =>  "$(bytes2hex(sha256("$(now())$(dataset_name)"))[1:Int(floor(end/3))])",
            "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
            "dim_redux_type"=>dim_redux_type, "dataset" => dataset_name,
            ## optim infos 
            "nepochs" => nepochs, "cph_lr" => cph_lr, "cph_wd" => cph_wd,
            ## model infos
            "model_type"=> modeltype, 
            "cph_nb_hl" => cph_nb_hl, "cph_hl_size" => hlsize
            ## metrics
    )
     # init params dict
     DataDict["params"] = deepcopy(base_params)
     DataDict["params"]["dim_redux"] = dim_redux_size
     DataDict["params"]["nb_clinf"] = nb_clinf
     DataDict["params"]["dataset"] = dataset_name
     DataDict["params"]["nsamples"] = size(CDS_data)[1]
     DataDict["params"]["nsamples_test"] = Int(round(size(CDS_data)[1] / base_params["nfolds"]))
     DataDict["params"]["ngenes"] = size(CDS_data)[2] 
     DataDict["params"]["nsamples_train"] = size(CDS_data)[1] - Int(round(size(CDS_data)[1] / base_params["nfolds"]))
     DataDict["params"]["model_title"] = "$(dataset_name)_$(modeltype)_$(dim_redux_type)_$(dim_redux_size)_$(nb_clinf)CF"
     DataDict["params"]["insize"] = DataDict["params"]["dim_redux"] + DataDict["params"]["nb_clinf"]
     # split folds 
    folds = split_train_test(CDS_data, DATA.survt, DATA.surve, DATA.samples;nfolds =DataDict["params"]["nfolds"])
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    # for fold in folds do train
    for fold in folds
        test_samples = DATA.samples[fold["test_ids"]]
        train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
        
        Umap_model = UMAP_(cpu(train_x), dim_redux_size)    
                
        X_tr_umap = UMAP.transform(Umap_model, cpu(train_x))
        X_tst_umap = UMAP.transform(Umap_model, cpu(test_x))


        ttr_x = gpu(Matrix(hcat(X_tr_umap', clinical_factors[folds[1]["train_ids"],:])'))
        ttst_x = gpu(Matrix(hcat(X_tst_umap', clinical_factors[folds[1]["test_ids"],:])'))
            
        DataDict["data_prep"] = Dict("train_x"=>ttr_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>ttst_x,
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
            push!(train_cinds, cind_tr)
            push!(test_cinds, cind_test)
            Flux.update!(cph_opt,cph_ps, cph_gs)
            if i % 1000 ==  0 || i == 1
                println("$dataset_name $session_id - $modeltype - $dim_redux_type - $dim_redux_size ($nb_clinf) FOLD $(fold["foldn"]) - $i : TRAIN c-ind: $(round(cind_tr, digits = 3))\tTEST c-ind: $(round(cind_test,digits =5))")
            end 
        end
        OUTS_tr = cphdnn(DataDict["data_prep"]["train_x"])
        cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DataDict["data_prep"]["train_y_t"],DataDict["data_prep"]["train_y_e"], -1 * OUTS_tr)    
        push!(OUTS_TST, vec(cpu(cphdnn(DataDict["data_prep"]["test_x"]))))
        push!(Y_T_TST, vec(cpu(DataDict["data_prep"]["test_y_t"])))
        push!(Y_E_TST, vec(cpu(DataDict["data_prep"]["test_y_e"])))
        DataDict["params"]["nparams"] = sum([*(size(x.weight)...) for x in cphdnn]) +  DataDict["params"]["cph_nb_hl"] * DataDict["params"]["cph_hl_size"]
    end 
    med_c_ind, lo_ci, up_ci = bootstrap_c_ind(OUTS_TST, Y_T_TST, Y_E_TST)
    println("TEST bootstrap c-index : $(med_c_ind) ($up_ci - $lo_ci 95% CI)")
    DataDict["params"]["cph_tst_c_ind_med"] = med_c_ind
    DataDict["params"]["cph_tst_c_ind_up_ci"] = lo_ci
    DataDict["params"]["cph_tst_c_ind_lo_ci"] = up_ci
    DataDict["params"]["cph_train_c_ind"] = mean(train_cinds)
    DataDict["params"]["cph_test_c_ind"] = mean(test_cinds)
    DataDict["params"]["model_cv_complete"] = true
    model_params_path = "$(DataDict["params"]["session_id"])/$(DataDict["params"]["model_title"])_$(DataDict["params"]["modelid"])"
    mkdir("RES/$model_params_path")
    bson("RES/$model_params_path/params.bson",DataDict["params"])
end 

function evaluate_coxridge_umap(CDS_data,clinical_factors,dataset_name,dim_redux_size, dim_redux_type;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-6, cph_wd = 0.1, modeltype = "cox_ridge")
    nb_clinf = size(clinical_factors)[2]
    base_params = Dict(
            ## run infos 
            "session_id" => session_id, "nfolds" =>5, "modelid" =>  "$(bytes2hex(sha256("$(now())$(dataset_name)"))[1:Int(floor(end/3))])",
            "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())",
            "dim_redux_type"=>dim_redux_type, "dataset" => dataset_name,
            ## optim infos 
            "nepochs" => nepochs, "cph_lr" => cph_lr, "cph_wd" => cph_wd,
            ## model infos
            "model_type"=> modeltype, 
            "cph_nb_hl" => cph_nb_hl, "cph_hl_size" => hlsize
            ## metrics
    )
     # init params dict
     DataDict["params"] = deepcopy(base_params)
     DataDict["params"]["dim_redux"] = dim_redux_size
     DataDict["params"]["nb_clinf"] = nb_clinf
     DataDict["params"]["dataset"] = dataset_name
     DataDict["params"]["nsamples"] = size(CDS_data)[1]
     DataDict["params"]["nsamples_test"] = Int(round(size(CDS_data)[1] / base_params["nfolds"]))
     DataDict["params"]["ngenes"] = size(CDS_data)[2] 
     DataDict["params"]["nsamples_train"] = size(CDS_data)[1] - Int(round(size(CDS_data)[1] / base_params["nfolds"]))
     DataDict["params"]["model_title"] = "$(dataset_name)_$(modeltype)_$(dim_redux_type)_$(dim_redux_size)_$(nb_clinf)CF"
     DataDict["params"]["insize"] = DataDict["params"]["dim_redux"] + DataDict["params"]["nb_clinf"]
     # split folds 
    folds = split_train_test(CDS_data, DATA.survt, DATA.surve, DATA.samples;nfolds =DataDict["params"]["nfolds"])
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    # for fold in folds do train
    for fold in folds
        test_samples = DATA.samples[fold["test_ids"]]
        Umap_model = UMAP_(cpu(train_x), dim_redux_size)    
                
        X_tr_umap = UMAP.transform(Umap_model, cpu(train_x))
        X_tst_umap = UMAP.transform(Umap_model, cpu(test_x))


        ttr_x = gpu(Matrix(hcat(X_tr_umap', clinical_factors[folds[1]["train_ids"],:])'))
        ttst_x = gpu(Matrix(hcat(X_tst_umap', clinical_factors[folds[1]["test_ids"],:])'))
            
        DataDict["data_prep"] = Dict("train_x"=>ttr_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>ttst_x,
        "test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)

        ## init model 
        cph_opt = Flux.ADAM(base_params["cph_lr"]) 
        cph_wd = base_params["cph_wd"]
        cox_ridge = gpu(Chain(Dense(DataDict["params"]["insize"], 1, bias = false)))
        for i in 1:DataDict["params"]["nepochs"]
            cph_ps = Flux.params(cox_ridge)
            cph_gs = gradient(cph_ps) do 
                cox_nll_vec(cox_ridge, DataDict["data_prep"]["train_x"], DataDict["data_prep"]["train_y_e"], DataDict["data_prep"]["NE_frac_tr"]) + l2_penalty(cox_ridge) * cph_wd
            end 
            #meta_eval(cphdnn, tcga_datasets, base_params, verbose = i, verbose_step = 1)
            OUTS_tst = cox_ridge(DataDict["data_prep"]["test_x"])
            OUTS_tr = cox_ridge(DataDict["data_prep"]["train_x"])
            cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DataDict["data_prep"]["test_y_t"], DataDict["data_prep"]["test_y_e"], -1 * OUTS_tst)
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DataDict["data_prep"]["train_y_t"],DataDict["data_prep"]["train_y_e"], -1 * OUTS_tr)
            push!(train_cinds, cind_tr)
            push!(test_cinds, cind_test)
            Flux.update!(cph_opt,cph_ps, cph_gs)
            if i % 1000 ==  0 || i == 1
                println("$dataset_name $session_id - $modeltype -  $dim_redux_type - $dim_redux_size ($nb_clinf) FOLD $(fold["foldn"]) - $i : TRAIN c-ind: $(round(cind_tr, digits = 3))\tTEST c-ind: $(round(cind_test,digits =5))")
            end 
        end
        OUTS_tr = cox_ridge(DataDict["data_prep"]["train_x"])
        cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DataDict["data_prep"]["train_y_t"],DataDict["data_prep"]["train_y_e"], -1 * OUTS_tr)    
        push!(OUTS_TST, vec(cpu(cox_ridge(DataDict["data_prep"]["test_x"]))))
        push!(Y_T_TST, vec(cpu(DataDict["data_prep"]["test_y_t"])))
        push!(Y_E_TST, vec(cpu(DataDict["data_prep"]["test_y_e"])))
        DataDict["params"]["nparams"] = sum([*(size(x.weight)...) for x in cox_ridge]) +  DataDict["params"]["cph_nb_hl"] * DataDict["params"]["cph_hl_size"]
    end 
    med_c_ind, lo_ci, up_ci = bootstrap_c_ind(OUTS_TST, Y_T_TST, Y_E_TST)
    println("TEST bootstrap c-index : $(med_c_ind) ($up_ci - $lo_ci 95% CI)")
    DataDict["params"]["cph_tst_c_ind_med"] = med_c_ind
    DataDict["params"]["cph_tst_c_ind_up_ci"] = lo_ci
    DataDict["params"]["cph_tst_c_ind_lo_ci"] = up_ci
    DataDict["params"]["cph_train_c_ind"] = mean(train_cinds)
    DataDict["params"]["cph_test_c_ind"] = mean(test_cinds)
    DataDict["params"]["model_cv_complete"] = true
    model_params_path = "$(DataDict["params"]["session_id"])/$(DataDict["params"]["model_title"])_$(DataDict["params"]["modelid"])"
    mkdir("RES/$model_params_path")
    bson("RES/$model_params_path/params.bson",DataDict["params"])
end

CDS_data = Matrix(DATA.data[:,keep])
for dim_redux_size in dim_redux_list
    # with clinical features 
    evaluate_cphdnn_umap(CDS_data, clinical_factors, dataset_name, dim_redux_size, dim_redux_type, nepochs=10_000, cph_lr = 1e-3)
    evaluate_coxridge_umap(CDS_data, clinical_factors, dataset_name, dim_redux_size, dim_redux_type;cph_lr =1e-3)
    # without clinical features 
    evaluate_cphdnn_umap(CDS_data, clinical_factors[:,[]], dataset_name, dim_redux_size, dim_redux_type, nepochs=10_000, cph_lr = 1e-3)
    evaluate_coxridge_umap(CDS_data, clinical_factors[:,[]], dataset_name, dim_redux_size, dim_redux_type;cph_lr =1e-3)
end