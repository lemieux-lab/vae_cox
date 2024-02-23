include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
#device!()
outpath, session_id = set_dirs() ;
# loading datasets in RAM 
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
for iter_id in 1:50
cph_nb_hl, hlsize,modeltype,nepochs= 2, 512, "direct_cox_ridge", 5_000

base_params = Dict(
        ## run infos 
        "session_id" => session_id, "nfolds" =>5,
        "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", "model_title"=>"direct_coxridge_with_tcga_datasets",
        ## optim infos 
        "nepochs" => nepochs, "ae_lr" =>0, "cph_lr" => 1e-6, "ae_wd" => 0, "cph_wd" => 0.1,
        ## model infos
        "model_type"=> modeltype, "dim_redux" => hlsize, "ae_nb_hls" => 2,
        "enc_nb_hl" => 0, "enc_hl_size"=> 0,
        "venc_nb_hl" => 0, "venc_hl_size"=> 0,  "dec_nb_hl" => 0 , "dec_hl_size"=> 0,
        "nb_clinf" => 0, "cph_nb_hl" => 0, "cph_hl_size" => hlsize
        ## metrics
)
tcga_datasets = load_tcga_datasets(tcga_datasets_list)
data_prep!(tcga_datasets, base_params)

## init model 

# cphdnn = gpu(Chain(Dense(tcga_datasets["BRCA"]["params"]["insize"],base_params["cph_hl_size"], leakyrelu), 
# Dense(base_params["cph_hl_size"],base_params["cph_hl_size"], leakyrelu), 
# Dense(base_params["cph_hl_size"], 1, bias = false)))

for DS_name in keys(tcga_datasets) 
    tcga_datasets[DS_name]["model"] = gpu(Chain(Dense(tcga_datasets["BRCA"]["params"]["insize"], 1, bias = false)))
    tcga_datasets[DS_name]["opt"] = Flux.ADAM(base_params["cph_lr"]) 
    tcga_datasets[DS_name]["params"]["modelid"] =  "$(bytes2hex(sha256("$(now())$(DS_name)"))[1:Int(floor(end/3))])"
    #println("$(bytes2hex(sha256("$(now())$(DS_name)"))[1:Int(floor(end/3))])")
end 

# prepping datasets and loading to GPU 
### Replicate 50 times?

function eval_direct(DS;verbose = 1,verbose_step =1)
    model = DS["model"]
    OUTS_tst = model(DS["data_prep"]["test_x"])
    OUTS_tr = model(DS["data_prep"]["train_x"])
    cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS["data_prep"]["test_y_t"], DS["data_prep"]["test_y_e"], -1 * OUTS_tst)
    cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DS["data_prep"]["train_y_t"],DS["data_prep"]["train_y_e"], -1 * OUTS_tr)
    DS["params"]["cph_tst_c_ind"] = cind_test
    DS["params"]["cph_train_c_ind"] = cind_tr
    DS["params"]["step"] = verbose
    return cind_test, cind_tr
end 
function cox_nll_direct_train(DataSet)
    model = DataSet["model"]
    opt = DataSet["opt"]
    params_dict = DataSet["params"]
    step = params_dict["step"]
    wd = params_dict["cph_wd"]
    ps = Flux.params(model)
    gs = gradient(ps) do 
        cox_nll_vec(model, DataSet["data_prep"]["train_x"], DataSet["data_prep"]["train_y_e"], DataSet["data_prep"]["NE_frac_tr"]) + l2_penalty(model) * wd
    end 
    cind_test, cind_tr = eval_direct(DataSet, verbose = step, verbose_step = 1)
    
    Flux.update!(opt,ps,gs)
    return cind_test, cind_tr 
end 

for i in 1:base_params["nepochs"]
    TESTDict = Dict()
    TRAINDict = Dict()
    for DS_name in keys(tcga_datasets)
        DataSet = tcga_datasets[DS_name]
        DataSet["params"]["step"] = i
        cind_test, cind_tr = cox_nll_direct_train(DataSet)
        TESTDict[DS_name] = round(cind_test, digits = 3)
        TRAINDict[DS_name] = round(cind_tr, digits = 3)
        
    end 
    TESTDict["step"] = i
    TRAINDict["step"] = i
    if i % 100 == 0 || i == 1
        println(DataFrame(TRAINDict))
        println(DataFrame(TESTDict))
    end    
end
dump_results(tcga_datasets["BRCA"]["model"], tcga_datasets)
end