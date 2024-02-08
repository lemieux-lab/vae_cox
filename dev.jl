
include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
outpath, session_id = set_dirs() ;
## import Leucegene, BRCA

#tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_datasets") ]
tcga_datasets_list = ["Data/TCGA_datasets/$(x)" for x in readdir("Data/TCGA_test") ]
BRCA_data = load_tcga_datasets(tcga_datasets_list)["BRCA"]
LgnAML_data = Dict("name"=>"LgnAML","dataset" => MLSurvDataset("Data/LEUCEGENE/LGN_AML_tpm_n300_btypes_labels_surv.h5"))
## 
#device!()
# loading datasets in RAM 
for (dataset_name,DataDict) in zip(["BRCA", "Lgn-AML"], [BRCA_data, LgnAML_data])

hlsize = 512
cph_nb_hl, modeltype,nepochs= 2,  "cphdnn_v6", 5_000
base_params = Dict(
        ## run infos 
        "session_id" => session_id, "nfolds" =>5,"modelid"=> "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
        "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", "model_title"=>"cphdnn_lgn_brca",
        ## optim infos 
        "nepochs" => nepochs, "ae_lr" =>0, "cph_lr" => 1e-6, "ae_wd" => 0, "cph_wd" => 0.1,
        ## model infos
        "model_type"=> modeltype, "dim_redux" => hlsize, "ae_nb_hls" => 2,
        "enc_nb_hl" => 0, "enc_hl_size"=> 0,
        "venc_nb_hl" => 0, "venc_hl_size"=> 0,  "dec_nb_hl" => 0 , "dec_hl_size"=> 0,
        "nb_clinf" => 0, "cph_nb_hl" => cph_nb_hl, "cph_hl_size" => hlsize
        ## metrics
)

DATA = DataDict["dataset"]
keep = [occursin("protein_coding", bt) for bt in DATA.biotypes]
println("$dataset_name nb genes : $(sum(keep))")
println("$dataset_name nb patients : $(size(DATA.samples)[1])")
println("$dataset_name % uncensored : $(round(mean(DATA.surve .!= 0), digits=3) * 100)%")

DataDict["params"] = deepcopy(base_params)
DataDict["params"]["dataset"] = dataset_name
DataDict["params"]["nsamples"] = size(DATA.samples)[1]
DataDict["params"]["nsamples_test"] = Int(round(size(DATA.samples)[1] / base_params["nfolds"]))
DataDict["params"]["ngenes"] = size(DATA.genes[keep])[1]
DataDict["params"]["nsamples_train"] = size(DATA.samples)[1] - Int(round(size(DATA.samples)[1] / base_params["nfolds"]))
DataDict["params"]["insize"] = size(DATA.genes[keep])[1]
        

## init model 
cph_opt = Flux.ADAM(base_params["cph_lr"]) 
cph_wd = base_params["cph_wd"]
cphdnn = gpu(Chain(Dense(DataDict["params"]["insize"],base_params["cph_hl_size"], leakyrelu), 
Dense(base_params["cph_hl_size"],base_params["cph_hl_size"], leakyrelu), 
Dense(base_params["cph_hl_size"], 1, bias = false)))

# prepping datasets and loading to GPU 
# split train test
folds = split_train_test(Matrix(DATA.data[:,keep]), DATA.survt, DATA.surve, DATA.samples;nfolds =5)
fold = folds[1]
# format input data  
test_samples = DATA.samples[fold["test_ids"]]
train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst = format_train_test(fold)
DataDict["data_prep"] = Dict("train_x"=>train_x, "train_y_t"=>train_y_t,"train_y_e"=>train_y_e,"NE_frac_tr"=>NE_frac_tr, "test_x"=>test_x,
"test_y_t"=> test_y_t, "test_y_e"=>test_y_e, "NE_frac_tst"=> NE_frac_tst)
### Replicate 50 times?

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

    DataDict["params"]["cph_tst_c_ind"] = cind_test
    DataDict["params"]["cph_train_c_ind"] = cind_tr
    DataDict["params"]["step"] = i

    Flux.update!(cph_opt,cph_ps, cph_gs)
    if i % 100 ==  0 || i == 1
        println("$dataset_name - $i : TRAIN c-ind: $(round(cind_tr, digits = 3))\tTEST c-ind: $(round(cind_test,digits =3))")
    end 
end
model = cphdnn
DataDict["params"]["nparams"] = sum([*(size(x.weight)...) for x in model]) +  DataDict["params"]["cph_nb_hl"] * DataDict["params"]["cph_hl_size"]
        
#dump_results(cphdnn, tcga_datasets)
model_params_path = "$(DataDict["params"]["session_id"])/$(DataDict["params"]["model_type"])_$(DataDict["params"]["dataset"])_$(DataDict["params"]["modelid"])"
mkdir("RES/$model_params_path")
bson("RES/$model_params_path/params.bson",DataDict["params"])
end
# OUTS_tst = cphdnn(brca_test_x)
# scores = sort(vec(cpu(OUTS_tst)))
# ranks = collect(1:size(OUTS_tst)[2])
# fig = Figure(resolution = (512,512));
# ax = Axis(fig[1,1], xlabel = "score ranks", ylabel ="score");
# scatter!(fig[1,1], ranks, scores);
# fig
### 2) fine-tuning on BRCA (train) and testing on BRCA (test)

# train_x, train_y_t, train_y_e, NE_frac_tr, test_samples, test_x, test_y_t, test_y_e, NE_frac_tst, params_dict = data_prep(LAML; nepochs = 10000, dim_redux = 1000, dataset = "LAML", modeltype = "vae_cox_transfer_v1")
# vae_train_x = gpu(Matrix(TCGA.data[[i for (i,x) in enumerate(TCGA.samples) if !(x in test_samples)],keep]));
# venc = VariationalEncoder(size(train_x)[1], params_dict["dim_redux"], 2000);
# vdec = Decoder(size(train_x)[1], params_dict["dim_redux"], 2000);
# vae_opt = Flux.ADAM(1e-5) ## opt VAE
# vae_wd = params_dict["ae_wd"]
# # training VAE only first
# mbsize = 600
# for i in 1:3000
#     ### SGD system
#     X_ = vae_train_x[shuffle(collect(1:size(vae_train_x)[1]))[1:mbsize],:]'
#     vae_ps = Flux.params(venc, vdec)
#     vae_gs = gradient(vae_ps) do 
#         VAE_lossf(venc, vdec, X_) + l2_penalty(venc) * vae_wd + l2_penalty(vdec) * vae_wd
#         #VAE_COX_loss(model["venc"], model["cph"], train_x, train_y_e, NE_frac_tr) + l2_penalty(model["venc"]) * wd + l2_penalty(model["cph"]) * wd 
#     end 
#     vae_loss =  round(VAE_lossf(venc, vdec, X_) + l2_penalty(venc) * vae_wd + l2_penalty(vdec) * vae_wd, digits = 3)  
#     VAE_test = round(my_cor(vec(test_x), vec(MyReconstruct(venc, vdec, test_x)[end])),digits = 3)

#     # cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, -1 * OUTS_tr)
#     # cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, -1 *OUTS_tst)
    
#     if i%100 == 0 || i == 1
#     println("$i TRAIN vae_loss : $vae_loss  \t TEST ae-corr: $VAE_test")
#     end
#     #Flux.update!(opt1,ps1, gs1)
#     Flux.update!(vae_opt,vae_ps, vae_gs)
#     params_dict["vae_cor"] = VAE_test
#     # params_dict["cph_tst_c_ind"] = cind_test
#     # params_dict["cph_train_c_ind"] = cind_tr
#     params_dict["step"] = i 
# end
# function VAE_COX_loss(VENC::VariationalEncoder, CPH, X, Y_e, NE_frac;device = gpu)
#     mu, log_sigma = VENC(X)
#     #z = mu + device(randn(Float32, size(log_sigma))) .* exp.(log_sigma)
#     outs = vec(CPH(mu))
#     hazard_ratios = exp.(outs)
#     log_risk = log.(cumsum(hazard_ratios))
#     uncensored_likelihood = outs .- log_risk
#     censored_likelihood = uncensored_likelihood .* Y_e'
#     #neg_likelihood = - sum(censored_likelihood) / sum(e .== 1)
#     neg_likelihood = - sum(censored_likelihood) * NE_frac
#     return neg_likelihood
# end 


# # training VENC-CPHDNN 

# # training VENC-CPHDNN 
# for i in 1:params_dict["nepochs"]
#     cph_ps  = Flux.params(venc, cphdnn)
#     cph_gs= gradient(cph_ps) do 
#         VAE_COX_loss(venc, cphdnn, train_x, train_y_e, NE_frac_tr) + l2_penalty(cphdnn) * cph_wd + l2_penalty(venc) * vae_wd
#         #VAE_COX_loss(model["venc"], model["cph"], train_x, train_y_e, NE_frac_tr) + l2_penalty(model["venc"]) * wd + l2_penalty(model["cph"]) * wd 
#     end 
#     vaecox_loss =  round(VAE_COX_loss(venc, cphdnn, train_x, train_y_e, NE_frac_tr) + l2_penalty(cphdnn) * cph_wd + l2_penalty(venc) * vae_wd, digits = 3)  
#     OUTS_tr = cphdnn(venc(train_x)[1])
#     OUTS_tst = cphdnn(venc(test_x)[1])
    
#     vae_loss =  round(VAE_lossf(venc, vdec, train_x) + l2_penalty(venc) * cph_wd + l2_penalty(vdec) * vae_wd, digits = 3)  
#     VAE_test = round(my_cor(vec(test_x), vec(MyReconstruct(venc, vdec, test_x)[end])),digits = 3)

#     cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, -1 * OUTS_tr)
#     cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, -1 *OUTS_tst)
    
#     if i%100 ==0 || i == 1
#     println("$i TRAIN $vaecox_loss cind: $(round(cind_tr, digits = 3)) \t TEST ae-corr: $VAE_test \tcind: $(round(cind_test, digits = 3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
#     end 
#     Flux.update!(cph_opt,cph_ps, cph_gs)
#     #Flux.update!(opt2,ps2, gs2)
#     params_dict["cph_tst_c_ind"] = cind_test
#     params_dict["cph_train_c_ind"] = cind_tr
#     params_dict["step"] = i 
# end
# model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
# mkdir("RES/$model_params_path")
# bson("RES/$model_params_path/params.bson",params_dict)

#end 
### c-index 

### save and can be loaded in bson

### Random hyper-params 
### CPH-DNN training
### c-index  
### Venc-CPH-DNN fine-tuning
### c-index 
