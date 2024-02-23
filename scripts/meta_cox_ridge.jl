include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
#device!()
outpath, session_id = set_dirs() ;
# loading datasets in RAM 
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]

cph_nb_hl, hlsize,modeltype,nepochs= 2, 512, "meta_cox_ridge", 5_000
base_params = Dict(
        ## run infos 
        "session_id" => session_id, "nfolds" =>5,"modelid"=> "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
        "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", "model_title"=>"meta_cox_ridge_with_tcga_datasets",
        ## optim infos 
        "nepochs" => nepochs, "ae_lr" =>0, "cph_lr" => 1e-6, "ae_wd" => 0, "cph_wd" => 0.1,
        ## model infos
        "model_type"=> modeltype, "dim_redux" => hlsize, "ae_nb_hls" => 2,
        "enc_nb_hl" => 0, "enc_hl_size"=> 0,
        "venc_nb_hl" => 0, "venc_hl_size"=> 0,  "dec_nb_hl" => 0 , "dec_hl_size"=> 0,
        "nb_clinf" => 0, "cph_nb_hl" => cph_nb_hl, "cph_hl_size" => hlsize
        ## metrics
)
tcga_datasets = load_tcga_datasets(tcga_datasets_list)
data_prep!(tcga_datasets, base_params)

## init model 
cph_opt = Flux.ADAM(base_params["cph_lr"]) 
cph_wd = base_params["cph_wd"]
cphdnn = gpu(Chain(Dense(tcga_datasets["BRCA"]["params"]["insize"], 1, bias = false)))

# prepping datasets and loading to GPU 
### Replicate 50 times?

for i in 1:base_params["nepochs"]
    cph_ps = Flux.params(cphdnn)
    cph_gs = gradient(cph_ps) do 
        meta_cox_nll_train(cphdnn, tcga_datasets) + l2_penalty(cphdnn) * base_params["cph_wd"] 
    end 
    meta_eval(cphdnn, tcga_datasets, base_params, verbose = i, verbose_step = 1)
    
    Flux.update!(cph_opt,cph_ps, cph_gs)
    end
dump_results(cphdnn, tcga_datasets)