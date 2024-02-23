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
DataDict = load_tcga_datasets(tcga_datasets_list)["BRCA"]
### Leucegene data processing  
# brca_CF_raw = CSV.read("Data/GDC_processed/TCGA_clinical_features_survival_pam50.csv", DataFrame)
# brca_CF_raw = CSV.read("Data/GDC_raw/TCGA_BRCA_survival_pam50.csv", DataFrame)[:,[1,2,7,8]]
# basepath = "Data/GDC_raw"
# FILES = "$basepath/TCGA_BRCA_clinical.json"
# J = JSON.parsefile(FILES)
# submitids, case_ids, surves, survts = [], [],[],[]
# stages, ethns = [], []
# for (i,F) in enumerate(J)
#     if "demographic" in keys(F) 
#     case_id = F["case_id"]
#     submitter_ID = split(F["demographic"]["submitter_id"],"_")[1]
#     surve = F["demographic"]["vital_status"] == "Dead" ? 1 : 0 # status 
#     if surve == 1 && "days_to_death" in keys(F["demographic"])
#         survt = Int(F["demographic"]["days_to_death"]) 
#         push!(submitids, submitter_ID)
#         push!(case_ids, case_id)
#         push!(surves, surve)
#         push!(survts, survt)
#         push!(ethns, F["demographic"]["race"])
#         if "ajcc_pathologic_stage" in keys(F["diagnoses"][1])
#             push!(stages, F["diagnoses"][1]["ajcc_pathologic_stage"])
#         else 
#             push!(stages, "missing")
            
#         end 
#         #println("$i $(case_id) $submitter_ID $surve $survt")
#     elseif surve == 0
#         survt = Int(F["diagnoses"][1]["days_to_last_follow_up"])
#         push!(submitids, submitter_ID)
#         push!(case_ids, case_id)
#         push!(surves, surve)
#         push!(survts, survt)
#         push!(ethns, F["demographic"]["race"])
#         if "ajcc_pathologic_stage" in keys(F["diagnoses"][1])
#         push!(stages, F["diagnoses"][1]["ajcc_pathologic_stage"])
#         else 
#         push!(stages,"missing")
        
#         end 
#     end
#     end 
# end 
# CLIN_df = DataFrame(:case_id=>case_ids, :submitid=>submitids,:survt=>Array{Int}(survts),:surve=>Array{Int}(surves), :stage=>stages, :ethn=>ethns)
# CLIN_df_2 = sort(DataFrame(["case_id"=>CLIN_df.case_id,
# "ethnicity"=> CLIN_df.ethn, 
# "submitid"=> CLIN_df.submitid, 
# "stage_i"=> Array{Int}([x in ["Stage I", "Stage IA", "Stage IB", "Stage IC"] for x in CLIN_df.stage]),
# "stage_ii"=>  Array{Int}([x in ["Stage II", "Stage IIA", "Stage IIB", "Stage IIC"] for x in CLIN_df.stage]),
# "stage_iii"=>  Array{Int}([x in ["Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC"] for x in CLIN_df.stage]),
# "stage_iv"=>  Array{Int}([x in ["Stage IV"] for x in CLIN_df.stage]) ]), "case_id")
# brca_CF = sort(innerjoin(brca_CF_raw, CLIN_df_2, on = ("barcode"=>"submitid")), "case_id")

# tmp_df = brca_CF[:,["age", "stage_i", "stage_ii","stage_iii", "stage_iv"]]
# CF_bin, lnames =  numerise_labels(brca_CF, ["clinical_data_PAM50MRNA"])
# clinical_factors = float.(hcat(Matrix(tmp_df), CF_bin))
# colnames = vcat(["age", "stage_i", "stage_ii","stage_iii", "stage_iv"], lnames)
# CF_bin, lnames =  numerise_labels(brca_CF, ["ethnicity"])
# clinical_factors = hcat(clinical_factors, CF_bin)
# colnames = vcat(colnames, lnames)
# CSV.write("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame(Dict([(colname, clinical_factors[:,i]) for (i, colname) in enumerate(colnames)])))
clinical_factors = Matrix(CSV.read("Data/GDC_processed/TCGA_BRCA_clinical_bin.csv", DataFrame))
# clinical_factors = Matrix(tmp_df)
dataset_name = "BRCA"
dim_redux_type = "RDM"
DATA = DataDict["dataset"] 
#sum(DATA.samples .== brca_CF.case_id)
#sum(DATA.surve .== brca_CF.surve)
keep = [occursin("protein_coding", bt) for bt in DATA.biotypes]
ngenes = sum(keep)
println("$dataset_name nb genes : $(size(DATA.genes)[1])")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")

dim_redux_list = shuffle([0,1,2,3,4,5,10,15,20,25,50,75,100,125,250,375,500,1_000,1_500,2_000,2_500,3000,5_000,6000,7000,8000,9000,10_000,11_000,12000,13000,14000,ngenes])
function evaluate_cphdnn(final_data, dataset_name,  dim_redux_size, dim_redux_type, nb_clinf;hlsize = 512, nepochs= 5_000, cph_nb_hl = 2, cph_lr = 1e-6, cph_wd = 0.1,  modeltype = "cphdnn")
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
     DataDict["params"]["nsamples"] = size(final_data)[1]
     DataDict["params"]["nsamples_test"] = Int(round(size(final_data)[1] / base_params["nfolds"]))
     DataDict["params"]["ngenes"] = size(final_data)[2] - nb_clinf
     DataDict["params"]["nsamples_train"] = size(final_data)[1] - Int(round(size(final_data)[1] / base_params["nfolds"]))
     DataDict["params"]["model_title"] = "$(dataset_name)_$(modeltype)_$(dim_redux_type)_$(dim_redux_size)_$(nb_clinf)CF"
     DataDict["params"]["insize"] = DataDict["params"]["dim_redux"] + DataDict["params"]["nb_clinf"]
     # split folds 
    folds = split_train_test(final_data, DATA.survt, DATA.surve, DATA.samples;nfolds =DataDict["params"]["nfolds"])
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
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
            push!(train_cinds, cind_tr)
            push!(test_cinds, cind_test)
            Flux.update!(cph_opt,cph_ps, cph_gs)
            if i % 1000 ==  0 || i == 1
                println("$dataset_name $session_id - $dim_redux_size FOLD $(fold["foldn"]) $dataset_name - $i : TRAIN c-ind: $(round(cind_tr, digits = 3))\tTEST c-ind: $(round(cind_test,digits =5))")
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

function evaluate_coxridge(final_data, dataset_name,  dim_redux_size, dim_redux_type, nb_clinf;hlsize = 0, nepochs= 5_000, cph_nb_hl = 0, cph_lr = 1e-6, cph_wd = 0.1, modeltype = "cox_ridge")
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
     DataDict["params"]["nsamples"] = size(final_data)[1]
     DataDict["params"]["nsamples_test"] = Int(round(size(final_data)[1] / base_params["nfolds"]))
     DataDict["params"]["ngenes"] = size(final_data)[2] - nb_clinf
     DataDict["params"]["nsamples_train"] = size(final_data)[1] - Int(round(size(final_data)[1] / base_params["nfolds"]))
     DataDict["params"]["model_title"] = "$(dataset_name)_$(modeltype)_$(dim_redux_type)_$(dim_redux_size)_$(nb_clinf)CF"
     DataDict["params"]["insize"] = DataDict["params"]["dim_redux"] + DataDict["params"]["nb_clinf"]
     # split folds 
    folds = split_train_test(final_data, DATA.survt, DATA.surve, DATA.samples;nfolds =DataDict["params"]["nfolds"])
    OUTS_TST, Y_T_TST, Y_E_TST, train_cinds, test_cinds = [], [], [], [],[]
    # for fold in folds do train
    for fold in folds
        test_samples = DATA.samples[fold["test_ids"]]
        train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
        DataDict["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
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
                println("$dataset_name $session_id - $dim_redux_size FOLD $(fold["foldn"]) $dataset_name - $i : TRAIN c-ind: $(round(cind_tr, digits = 3))\tTEST c-ind: $(round(cind_test,digits =5))")
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

for dim_redux_size in dim_redux_list

    #dim_redux_size = dim_redux_list[3]
    CDS_data = Matrix(DATA.data[:,keep])
    dredux_data = Matrix(CDS_data[:,sample(collect(1:size(CDS_data)[2]), dim_redux_size, replace = false)])
    nb_clinf = size(clinical_factors)[2]
    final_data = hcat(dredux_data, clinical_factors)
    evaluate_cphdnn(final_data,dataset_name,dim_redux_size, dim_redux_type,  nb_clinf )    
    evaluate_coxridge(final_data,dataset_name,dim_redux_size, dim_redux_type,  nb_clinf )    
    if dim_redux_size != 0
        final_data = Matrix(CDS_data[:,sample(collect(1:size(CDS_data)[2]), dim_redux_size, replace = false)])
        evaluate_coxridge(final_data,dataset_name,dim_redux_size, dim_redux_type,  0) 
        evaluate_cphdnn(final_data,dataset_name,dim_redux_size, dim_redux_type,  0)       
    end 
end