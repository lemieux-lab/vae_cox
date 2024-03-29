
include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
device!()
outpath, session_id = set_dirs() ;
LGG = MLSurvDataset("Data/TCGA_LGG_tpm_n513_btypes_labels_surv.h5")
OV = MLSurvDataset("Data/TCGA_OV_tpm_n420_btypes_labels_surv.h5")
BRCA = MLSurvDataset("Data/TCGA_BRCA_tpm_n1049_btypes_labels_surv.h5")
LAML = MLSurvDataset("Data/LGN_AML_tpm_n300_btypes_labels_surv.h5") 
TCGA = MLDataset("Data/TCGA_tpm_n10384_btypes_labels.h5")


### pre-training on TCGA
DATA = TCGA
nfolds = 5 
nepochs = 10000
dim_redux= 1000
dataset="BRCA"
keep = [occursin("protein_coding", bt) for bt in TCGA.biotypes]

### VAE - params
params_dict = Dict(
    ## run infos 
    "session_id" => session_id, "nfolds" =>5,  "modelid" => "$(bytes2hex(sha256("$(now())"))[1:Int(floor(end/3))])",
    "machine_id"=>strip(read(`hostname`, String)), "device" => "$(device())", "model_title"=>"AECPHDNN",
    ## data infos 
    "dataset" => dataset, "nsamples" => size(DATA.samples)[1],
    "nsamples_test" => Int(round(size(DATA.samples)[1] / nfolds)), "ngenes" => size(DATA.genes[keep])[1],
    "nsamples_train" => size(DATA.samples)[1] - Int(round(size(DATA.samples)[1] / nfolds)),
    ## optim infos 
    "nepochs" => nepochs, "ae_lr" =>1e-6, "cph_lr" => 1e-5, "ae_wd" => 1e-6, "cph_wd" => 1e-4,
    ## model infos
    "model_type"=> "vae_cox_transfer_v1", "dim_redux" => dim_redux, "ae_nb_hls" => 2,
    "enc_nb_hl" => 2, "enc_hl_size"=> 128,
    "venc_nb_hl" => 2, "venc_hl_size"=> 128,  "dec_nb_hl" => 2 , "dec_hl_size"=> 128,
    "nb_clinf" => 0, "cph_nb_hl" => 2, "cph_hl_size" => 64, 
    "insize" => size(DATA.genes[keep])[1],
    ## metrics
    "training_complete" => false
)


### Bottleneck 1-insize 

function VariationalEncoder(input_dim::Int, latent_dim::Int, hidden_dim::Int;device = gpu) 
    return VariationalEncoder(
    device(Chain(Dense(input_dim, hidden_dim, leakyrelu), Dense(hidden_dim,hidden_dim, leakyrelu))),   # linear
    device(Dense(hidden_dim, latent_dim)),        # mu
    device(Dense(hidden_dim, latent_dim)))        # log sigma
end 
function l2_penalty(model::VariationalEncoder)
    l2_sum = 0
    for wm in model.linear
        l2_sum += sum(abs2, wm.weight)
    end 
    l2_sum += sum(abs2, model.mu.weight)
    l2_sum += sum(abs2, model.log_sigma.weight)
    return l2_sum
end
for iter_id in 1:50
train_x, train_y_t, train_y_e, NE_frac_tr, test_samples, test_x, test_y_t, test_y_e, NE_frac_tst, params_dict = data_prep(LAML; nepochs = 10000, dim_redux = 1000, dataset = "LAML", modeltype = "vae_cox_transfer_v1")
vae_train_x = gpu(Matrix(TCGA.data[[i for (i,x) in enumerate(TCGA.samples) if !(x in test_samples)],keep]));
venc = VariationalEncoder(size(train_x)[1], params_dict["dim_redux"], 2000);
vdec = Decoder(size(train_x)[1], params_dict["dim_redux"], 2000);
vae_opt = Flux.ADAM(1e-5) ## opt VAE
vae_wd = params_dict["ae_wd"]
# training VAE only first
mbsize = 600
for i in 1:3000
    ### SGD system
    X_ = vae_train_x[shuffle(collect(1:size(vae_train_x)[1]))[1:mbsize],:]'
    vae_ps = Flux.params(venc, vdec)
    vae_gs = gradient(vae_ps) do 
        VAE_lossf(venc, vdec, X_) + l2_penalty(venc) * vae_wd + l2_penalty(vdec) * vae_wd
        #VAE_COX_loss(model["venc"], model["cph"], train_x, train_y_e, NE_frac_tr) + l2_penalty(model["venc"]) * wd + l2_penalty(model["cph"]) * wd 
    end 
    vae_loss =  round(VAE_lossf(venc, vdec, X_) + l2_penalty(venc) * vae_wd + l2_penalty(vdec) * vae_wd, digits = 3)  
    VAE_test = round(my_cor(vec(test_x), vec(MyReconstruct(venc, vdec, test_x)[end])),digits = 3)

    # cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, -1 * OUTS_tr)
    # cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, -1 *OUTS_tst)
    
    if i%100 == 0 || i == 1
    println("$i TRAIN vae_loss : $vae_loss  \t TEST ae-corr: $VAE_test")
    end
    #Flux.update!(opt1,ps1, gs1)
    Flux.update!(vae_opt,vae_ps, vae_gs)
    params_dict["vae_cor"] = VAE_test
    # params_dict["cph_tst_c_ind"] = cind_test
    # params_dict["cph_train_c_ind"] = cind_tr
    params_dict["step"] = i 
end
function VAE_COX_loss(VENC::VariationalEncoder, CPH, X, Y_e, NE_frac;device = gpu)
    mu, log_sigma = VENC(X)
    #z = mu + device(randn(Float32, size(log_sigma))) .* exp.(log_sigma)
    outs = vec(CPH(mu))
    hazard_ratios = exp.(outs)
    log_risk = log.(cumsum(hazard_ratios))
    uncensored_likelihood = outs .- log_risk
    censored_likelihood = uncensored_likelihood .* Y_e'
    #neg_likelihood = - sum(censored_likelihood) / sum(e .== 1)
    neg_likelihood = - sum(censored_likelihood) * NE_frac
    return neg_likelihood
end 

cph_opt = Flux.ADAM(1e-6) ## opt VAE
cph_wd = params_dict["cph_wd"]
cphdnn = gpu(Chain(Dense(params_dict["dim_redux"],512, leakyrelu), Dense(512,512, leakyrelu), Dense(512, 1, bias = false)))

# training VENC-CPHDNN 

# training VENC-CPHDNN 
for i in 1:params_dict["nepochs"]
    cph_ps  = Flux.params(venc, cphdnn)
    cph_gs= gradient(cph_ps) do 
        VAE_COX_loss(venc, cphdnn, train_x, train_y_e, NE_frac_tr) + l2_penalty(cphdnn) * cph_wd + l2_penalty(venc) * vae_wd
        #VAE_COX_loss(model["venc"], model["cph"], train_x, train_y_e, NE_frac_tr) + l2_penalty(model["venc"]) * wd + l2_penalty(model["cph"]) * wd 
    end 
    vaecox_loss =  round(VAE_COX_loss(venc, cphdnn, train_x, train_y_e, NE_frac_tr) + l2_penalty(cphdnn) * cph_wd + l2_penalty(venc) * vae_wd, digits = 3)  
    OUTS_tr = cphdnn(venc(train_x)[1])
    OUTS_tst = cphdnn(venc(test_x)[1])
    
    vae_loss =  round(VAE_lossf(venc, vdec, train_x) + l2_penalty(venc) * cph_wd + l2_penalty(vdec) * vae_wd, digits = 3)  
    VAE_test = round(my_cor(vec(test_x), vec(MyReconstruct(venc, vdec, test_x)[end])),digits = 3)

    cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, -1 * OUTS_tr)
    cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, -1 *OUTS_tst)
    
    if i%100 ==0 || i == 1
    println("$i TRAIN $vaecox_loss cind: $(round(cind_tr, digits = 3)) \t TEST ae-corr: $VAE_test \tcind: $(round(cind_test, digits = 3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
    end 
    Flux.update!(cph_opt,cph_ps, cph_gs)
    #Flux.update!(opt2,ps2, gs2)
    params_dict["cph_tst_c_ind"] = cind_test
    params_dict["cph_train_c_ind"] = cind_tr
    params_dict["step"] = i 
end
model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
mkdir("RES/$model_params_path")
bson("RES/$model_params_path/params.bson",params_dict)

end 

