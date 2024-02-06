#### MEta learning functions

function dump_results(model, DataSets)
    for DS_name in keys(DataSets)
        params_dict = DataSets[DS_name]["params"]
        params_dict["nparams"] = sum([*(size(x.weight)...) for x in model]) +  params_dict["cph_nb_hl"] * params_dict["cph_hl_size"]
        model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["dataset"])_$(params_dict["modelid"])"
        if !("$(params_dict["model_type"])_$(params_dict["dataset"])_$(params_dict["modelid"])" in readdir("RES/$(params_dict["session_id"])")) 
            mkdir("RES/$model_params_path") 
        end 
        bson("RES/$model_params_path/params.bson",params_dict)
    end 
end

function meta_cox_nll_train(model, DS)
    psum = 0
    for DS_name in keys(DS)
        psum += cox_nll_vec(model, DS[DS_name]["data_prep"]["train_x"], 
            DS[DS_name]["data_prep"]["train_y_e"], 
            DS[DS_name]["data_prep"]["NE_frac_tr"]) 
    end 
    return psum 
end 
    
function meta_eval(model, DS, base_params;verbose = 1, verbose_step = 10)
    # loss train 
    lossval_combined = meta_cox_nll_train(model, DS) + l2_penalty(model) * base_params["cph_wd"]
    TESTDict = Dict()
    TRAINDict = Dict()
    for DS_name in keys(DS)
        OUTS_tst = model(DS[DS_name]["data_prep"]["test_x"])
        OUTS_tr = model(DS[DS_name]["data_prep"]["train_x"])
        cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(DS[DS_name]["data_prep"]["test_y_t"], DS[DS_name]["data_prep"]["test_y_e"], -1 * OUTS_tst)
        cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(DS[DS_name]["data_prep"]["train_y_t"],DS[DS_name]["data_prep"]["train_y_e"], -1 * OUTS_tr)
    
        TESTDict[DS_name] = round(cind_test, digits = 3)
        TRAINDict[DS_name] = round(cind_tr, digits = 3)
        
        DS[DS_name]["params"]["cph_tst_c_ind"] = cind_test
        DS[DS_name]["params"]["cph_train_c_ind"] = cind_tr
        DS[DS_name]["params"]["step"] = verbose

    end
    TESTDict["step"] = verbose
    TRAINDict["step"] = verbose
    
    #OUTS_tr = cphdnn(tcga_datasets[DS_name]["data_prep"]["train_x"])
    
    if verbose % verbose_step ==  0 || verbose == 1
        println(DataFrame(TRAINDict))
        println(DataFrame(TESTDict))
    end 
end 
#### VAE functions 

struct VariationalEncoder
    linear
    mu
    log_sigma
end
    
function VariationalEncoder(input_dim::Int, latent_dim::Int, hidden_dim::Int;device = gpu) 
    return VariationalEncoder(
    device(Dense(input_dim, hidden_dim, leakyrelu)),   # linear
    device(Dense(hidden_dim, latent_dim)),        # mu
    device(Dense(hidden_dim, latent_dim)))        # log sigma
end 

function (encoder::VariationalEncoder)(x)
    h = encoder.linear(x)
    encoder.mu(h), encoder.log_sigma(h)
end
Decoder(input_dim::Int, latent_dim::Int, hidden_dim::Int;device = gpu) = Chain(
    device(Dense(latent_dim, hidden_dim, leakyrelu)),
    device(Dense(hidden_dim, hidden_dim, leakyrelu)),
    device(Dense(hidden_dim, input_dim))
)
function MyReconstruct(encoder, decoder, x;device=gpu)
    mu, log_sigma = encoder(x)
    z = mu + device(randn(Float32, size(log_sigma))) .* exp.(log_sigma)
    mu, log_sigma, decoder(z)
end

# KL-divergence
function VAE_lossf(venc, vdec, X)
    mu, log_sigma, decoder_z = MyReconstruct(venc, vdec, X);
    nb_samples = size(X)[2]
    kl = 0.5f0 * sum(@. exp(log_sigma * 2f0) + mu ^ 2 - 1f0 - 2 * log_sigma) / nb_samples;
    mse = Flux.mse(X, decoder_z,agg=sum) / nb_samples
    return kl + mse 
end 

##### Model specifications
struct dnn
    model::Flux.Chain 
    opt
    lossf
end

function dnn(params::Dict)
    mdl_chain = gpu(Flux.Chain(
    Flux.Dense(params["dim_redux"], params["clf_hl_size"], relu), 
    Flux.Dense(params["clf_hl_size"], params["clf_hl_size"], relu), 
    Flux.Dense(params["clf_hl_size"], params["nclasses"], identity)))
    mdl_opt = Flux.ADAM(params["lr"])
    mdl_lossf = crossentropy_l2
    mdl = dnn(mdl_chain, mdl_opt, mdl_lossf)
    return mdl
end 

struct logistic_regression
    model::Flux.Dense
    opt
    lossf
end 

struct DataFE
    name::String
    data::Array
    factor_1::Array
    factor_2::Array
end

struct FE_model
    net::Flux.Chain
    embed_1::Flux.Embedding
    embed_2::Flux.Embedding
    hl1::Flux.Dense
    hl2::Flux.Dense
    outpl::Flux.Dense
    opt
    lossf
end

function FE_model(params::Dict)
    emb_size_1 = params["emb_size_1"]
    emb_size_2 = params["emb_size_2"]
    a = emb_size_1 + emb_size_2 
    b, c = params["fe_hl1_size"], params["fe_hl2_size"] 
    emb_layer_1 = gpu(Flux.Embedding(params["nsamples"], emb_size_1))
    emb_layer_2 = gpu(Flux.Embedding(params["ngenes"], emb_size_2))
    hl1 = gpu(Flux.Dense(a, b, relu))
    hl2 = gpu(Flux.Dense(b, c, relu))
    outpl = gpu(Flux.Dense(c, 1, identity))
    net = gpu(Flux.Chain(
        Flux.Parallel(vcat, emb_layer_1, emb_layer_2),
        hl1, hl2, outpl,
        vec))
    opt = Flux.ADAM(params["lr"])
    lossf = mse_l2
    FE_model(net, emb_layer_1, emb_layer_2, hl1, hl2, outpl, opt, lossf)
end 

function prep_FE(data; device = gpu)
    ## data preprocessing
    ### remove index columns, log transform
    n = length(data.factor_1)
    m = length(data.factor_2)
    values = Array{Float32,2}(undef, (1, n * m))
    #print(size(values))
    factor_1_index = Array{Int64,1}(undef, max(n * m, 1))
    factor_2_index = Array{Int64,1}(undef, max(n * m, 1))
    # d3_index = Array{Int32,1}(undef, n * m)
    
    for i in 1:n
        for j in 1:m
            index = (i - 1) * m + j 
            values[1, index] = data.data[i, j]
            factor_1_index[index] = i # Int
            factor_2_index[index] = j # Int 
            # d3_index[index] = data.d3_index[i] # Int 
        end
    end
    return (device(factor_1_index), device(factor_2_index)), device(vec(values))
end

struct mtl_FE
    fe::FE_model
    fe_data::DataFE
    clf::dnn 
end 

struct AE_model
    net::Flux.Chain 
    encoder::Flux.Chain
    decoder::Flux.Chain
    outpl::Flux.Dense
    opt
    lossf
end 
struct AE_AE_DNN
    ae::AE_model 
    clf::dnn
    encoder::Chain
    ae2d::AE_model
end 

function AE_model(params::Dict)
    ## 2 x 2 Hidden layers Auto-Encoder model architecture.  
    enc_hl1 = gpu(Flux.Dense(params["ngenes"], params["enc_hl_size"], relu))
    enc_hl2 = gpu(Flux.Dense(params["enc_hl_size"], params["enc_hl_size"], relu))

    redux_layer = gpu(Flux.Dense(params["enc_hl_size"], params["dim_redux"], relu))
    
    dec_hl1 = gpu(Flux.Dense(params["dim_redux"], params["dec_hl_size"], relu))
    dec_hl2 = gpu(Flux.Dense(params["dec_hl_size"], params["dec_hl_size"], relu))

    outpl = gpu(Flux.Dense(params["dec_hl_size"], params["ngenes"], identity))

    net = gpu(Flux.Chain(
        enc_hl1, enc_hl2, redux_layer, dec_hl1, dec_hl2, outpl    
    ))
    encoder = gpu(Flux.Chain(enc_hl1, enc_hl2, redux_layer))
    decoder = gpu(Flux.Chain(dec_hl1, dec_hl2, outpl))

    opt = Flux.ADAM(params["lr_ae"])
    lossf = mse_l2
    AE_model(net, encoder, decoder, outpl, opt, lossf)
end 

struct mtl_AE
    ae::AE_model 
    clf::dnn
    encoder::Chain
end 

struct enccphdnn
    encoder::Chain
    cphdnn::Chain
    opt
    lossf
end 
struct mtcphAE
    ae::AE_model 
    cph::enccphdnn
    encoder::Chain
end 

###### Regularisation functions 
function l2_penalty(model::Flux.Chain)
    l2_sum = 0
    for wm in model
        l2_sum += sum(abs2, wm.weight)
    end 
    return l2_sum
end
function l2_penalty(model::VariationalEncoder)
    l2_sum = 0
    l2_sum += sum(abs2, model.linear.weight)
    l2_sum += sum(abs2, model.mu.weight)
    l2_sum += sum(abs2, model.log_sigma.weight)
    return l2_sum
end
function l2_penalty(model::enccphdnn)
    l2_sum = 0
    for wm in Flux.Chain(model.cphdnn...)
        if !isa(wm, Dropout) 
            l2_sum += sum(abs2, wm.weight)
        end
    end 
    return l2_sum
end
function l2_penalty(model::logistic_regression)
    return sum(abs2, model.model.weight)
end 

function l2_penalty(model::dnn)
    l2_sum = 0
    for wm in model.model
        l2_sum += sum(abs2, wm.weight)
    end 
    return l2_sum
end

function l2_penalty(model::FE_model)
    return sum(abs2, model.embed_1.weight) + sum(abs2, model.embed_2.weight) + sum(abs2, model.hl1.weight) + sum(abs2, model.hl2.weight)
end

function l2_penalty(model::AE_model)
    l2_sum = 0 
    for wm in model.encoder
        if !isa(wm, Dropout) 
            l2_sum += sum(abs2, wm.weight)
        end
    end 
    for wm in model.decoder
        l2_sum += sum(abs2, wm.weight)
    end 
    return l2_sum 
end
# function l2_penalty(model::AE_model)
#     return sum(abs2, model.encoder[end].weight)
# end 
####### Loss functions
function mse_l2(model::AE_model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.net(X), Y) + l2_penalty(model) * weight_decay
end 

function mse_l2(model::FE_model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.net(X), Y) + l2_penalty(model) * weight_decay
end 

function mse_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.mse(model.model(X), Y) + l2_penalty(model) * weight_decay
end 
function crossentropy_l2(model, X, Y;weight_decay = 1e-6)
    return Flux.Losses.logitcrossentropy(model.model(X), Y) + l2_penalty(model) * weight_decay
end 


function cox_nll_vec(mdl::Chain, X_, Y_e_, NE_frac)
    outs = vec(mdl(X_))
    #outs = vec(mdl.cphdnn(mdl.encoder(X_)))
    hazard_ratios = exp.(outs)
    log_risk = log.(cumsum(hazard_ratios))
    uncensored_likelihood = outs .- log_risk
    censored_likelihood = uncensored_likelihood .* Y_e_'
    #neg_likelihood = - sum(censored_likelihood) / sum(e .== 1)
    neg_likelihood = - sum(censored_likelihood) * NE_frac
    return neg_likelihood
end 
####### Model picker
####HELPER functions 
function layer_size(insize, dim_redux;nb_hl=2)
    x1, y1 = 0,dim_redux
    x2, y2 = nb_hl + 1,insize
    f(x::Int) = Int(floor(sqrt(x * (y2 - y1) / (x2 - x1)))) + y1 
    return f 
end 
function compute_c(insize, bn_size, nb_hl)
    return ( insize / bn_size) ^ (1/ (nb_hl + 1) )
end 
####
function build(model_params; adaptative=true)
    # picks right confiration model for given params
    if model_params["model_type"] == "linear"
        chain = gpu(Dense(model_params["insize"] , model_params["outsize"],identity))
        opt = Flux.ADAM(model_params["lr"])
        lossf = crossentropy_l2
        model = logistic_regression(chain, opt, lossf)
    elseif model_params["model_type"] == "clfdnn"
        c = compute_c(model_params["insize"], model_params["outsize"], model_params["nb_hl"] )
        hls = []
        hl_sizes = [Int(floor(model_params["outsize"] * c ^ x)) for x in 1:model_params["nb_hl"]]
        hl_sizes = !adaptative ? Array{Int}(ones(10) .* model_params["hl_size"]) : hl_sizes  
        for i in 1:model_params["nb_hl"]
            in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(hls, gpu(Flux.Dense(in_size, out_size, model_params["n.-lin"])))
        end
        chain = gpu(Chain(hls..., Dense(hl_sizes[end], model_params["outsize"], identity)))
        opt = Flux.ADAM(model_params["lr"])
        lossf = crossentropy_l2
        model = dnn(chain, opt, lossf)
    elseif model_params["model_type"] == "cphclinf"
        chain = gpu(Chain(Dense(model_params["nb_clinf"], 1, sigmoid, bias = false)))
        opt = Flux.ADAM(model_params["cph_lr"])
        model = dnn(chain, opt, cox_l2)

    elseif model_params["model_type"] == "cphdnnclinf"
        chain = gpu(Chain(Dense(model_params["insize"] + model_params["nb_clinf"] , model_params["cph_hl_size"], leakyrelu),
        #Dense(model_params["cph_hl_size"] , model_params["cph_hl_size"], leakyrelu),
        Dense(model_params["cph_hl_size"] , 1, sigmoid, bias = false)))
        opt = Flux.ADAM(model_params["cph_lr"])
        model = dnn(chain, opt, cox_l2)
    elseif model_params["model_type"] == "cphdnnclinf_noexpr"
        chain = gpu(Chain(Dense(model_params["nb_clinf"] , model_params["cph_hl_size"], leakyrelu),
        Dense(model_params["cph_hl_size"] , model_params["cph_hl_size"], leakyrelu),
        Dense(model_params["cph_hl_size"] , 1, identity, bias = false)))
        opt = Flux.ADAM(model_params["cph_lr"])
        model = dnn(chain, opt, cox_l2)


    elseif model_params["model_type"] == "cphdnn"
        chain = gpu(Chain(Dense(model_params["insize"] + model_params["nb_clinf"] , model_params["cph_hl_size"], leakyrelu),
        #Dense(model_params["cph_hl_size"] , model_params["cph_hl_size"], relu),
        Dense(model_params["cph_hl_size"] , 1, sigmoid, bias = false)))
        opt = Flux.ADAM(model_params["cph_lr"])
        model = dnn(chain, opt, cox_l2)
    elseif model_params["model_type"] == "mtl_FE"
        FE = FE_model(model_params)
        data_FE = DataFE("fe_data", model_params["fe_data"], collect(1:model_params["nsamples"]),collect(1:model_params["ngenes"]) )
        clf_chain = gpu(Flux.Chain(FE.embed_1, 
        Flux.Dense(model_params["emb_size_1"], model_params["clf_hl_size"], relu), 
        Flux.Dense(model_params["clf_hl_size"],model_params["clf_hl_size"], relu), 
        Flux.Dense(model_params["clf_hl_size"],model_params["nclasses"], identity)))
        clf_opt = Flux.ADAM(model_params["lr"])
        clf_lossf = crossentropy_l2
        clf = dnn(clf_chain, clf_opt, clf_lossf)
        model = mtl_FE(FE, data_FE , clf)
    elseif model_params["model_type"] == "enccphdnn"
        ls2 = layer_size(model_params["insize"], model_params["dim_redux"])
        #enc_hl1 = gpu(Flux.Dense(model_params["insize"], ls2(2), relu))
        #enc_hl2 = gpu(Flux.Dense(ls2(2),ls2(1), relu))
        #redux_layer = gpu(Flux.Dense(ls2(1), model_params["dim_redux"], relu))
        enc_hl1 = gpu(Flux.Dense(model_params["insize"], model_params["enc_hl_size"], relu))
        enc_hls = []
        for i in 1:model_params["enc_nb_hl"]
            push!(enc_hls, gpu(Flux.Dense(model_params["enc_hl_size"],model_params["enc_hl_size"], relu)))
        end
        redux_layer = gpu(Flux.Dense(model_params["enc_hl_size"], model_params["dim_redux"], relu))
        encoder = gpu(Flux.Chain(enc_hl1,enc_hls..., redux_layer))
        #cphdnn = gpu(Flux.Chain(Dense(model_params["dim_redux"] + model_params["nb_clinf"], 1, identity;bias = false)))
        cphdnn = gpu(Flux.Chain(Dense(model_params["dim_redux"]  + model_params["nb_clinf"] , model_params["cph_hl_size"], tanh),
        Dense(model_params["cph_hl_size"] ,1, identity, bias=false)))#, model_params["cph_hl_size"], relu),
        #Dense(model_params["cph_hl_size"] , 1, sigmoid))) 
        opt = Flux.ADAM(model_params["cph_lr"])
        model = enccphdnn(encoder, cphdnn, opt, cox_l2)
    elseif model_params["model_type"] == "aeclfdnn"
        c = compute_c(model_params["insize"], model_params["dim_redux"], model_params["enc_nb_hl"] )
        enc_hls = []
        hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
        hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end 
        redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], model_params["dim_redux"],identity))
        encoder = Flux.Chain(enc_hls..., redux_layer)
        dec_hls = []
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : hl_sizes[i - 1]
            out_size = hl_sizes[i]
            push!(dec_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end
        output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["insize"], leakyrelu))
        decoder = Flux.Chain(dec_hls..., output_layer)
        ae_chain = Flux.Chain(enc_hls..., redux_layer, dec_hls..., output_layer)
        AE = AE_model(ae_chain, encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2) 
        # classifier DNN
        c = compute_c(model_params["dim_redux"], model_params["outsize"], model_params["clfdnn_nb_hl"] )
        hls = []
        hl_sizes = [Int(floor(model_params["outsize"] * c ^ x)) for x in 1:model_params["clfdnn_nb_hl"]]
        hl_sizes = !adaptative ? Array{Int}(ones(10) .* model_params["clfdnn_hl_size"]) : hl_sizes  
        for i in 1:model_params["clfdnn_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(hls, gpu(Flux.Dense(in_size, out_size, model_params["n.-lin"])))
        end
        clf_chain = gpu(Chain(encoder..., hls..., Dense(hl_sizes[end], model_params["outsize"], identity)))
        
        clf_opt = Flux.ADAM(model_params["clfdnn_lr"])
        clf_lossf = crossentropy_l2
        clf = dnn(clf_chain, clf_opt, clf_lossf)
        model = mtl_AE(AE, clf, AE.encoder)
    elseif model_params["model_type"] == "aeaeclfdnn"
        c = compute_c(model_params["insize"], model_params["dim_redux"], model_params["enc_nb_hl"] )
        enc_hls = []
        hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
        hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end 
        redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], model_params["dim_redux"],leakyrelu))
        encoder = Flux.Chain(enc_hls..., redux_layer)
        dec_hls = []
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : hl_sizes[i - 1]
            out_size = hl_sizes[i]
            push!(dec_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end
        output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["insize"], leakyrelu))
        decoder = Flux.Chain(dec_hls..., output_layer)
        ae_chain = Flux.Chain(enc_hls..., redux_layer, dec_hls..., output_layer)
        AE = AE_model(ae_chain, encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2) 
        # classifier DNN
        c = compute_c(model_params["dim_redux"], model_params["outsize"], model_params["clfdnn_nb_hl"] )
        hls = []
        hl_sizes = [Int(floor(model_params["outsize"] * c ^ x)) for x in 1:model_params["clfdnn_nb_hl"]]
        hl_sizes = !adaptative ? Array{Int}(ones(10) .* model_params["clfdnn_hl_size"]) : hl_sizes  
        for i in 1:model_params["clfdnn_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(hls, gpu(Flux.Dense(in_size, out_size, model_params["n.-lin"])))
        end
        clf_chain = gpu(Chain(encoder..., hls..., Dense(hl_sizes[end], model_params["outsize"], identity)))
        
        clf_opt = Flux.ADAM(model_params["clfdnn_lr"])
        clf_lossf = crossentropy_l2
        clf = dnn(clf_chain, clf_opt, clf_lossf)
        # 2D encoder 
        c = compute_c(model_params["dim_redux"], 2, model_params["enc_nb_hl"] )
        enc_hls = []
        hl_sizes = [Int(floor(2 * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
        hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end 
        redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], 2, identity))
        encoder = Flux.Chain(enc_hls..., redux_layer)
        dec_hls = []
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? 2 : hl_sizes[i - 1]
            out_size = hl_sizes[i]
            push!(dec_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end
        output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["dim_redux"], leakyrelu))
        decoder = Flux.Chain(dec_hls..., output_layer)
        ae_chain = Flux.Chain(enc_hls..., redux_layer, dec_hls..., output_layer)
        AE2D = AE_model(ae_chain, encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2) 
        model = AE_AE_DNN(AE, clf, AE.encoder, AE2D)
    elseif model_params["model_type"] == "aecphdnn"
        c = compute_c(model_params["insize"], model_params["dim_redux"], model_params["enc_nb_hl"] )
        enc_hls = []
        hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
        hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
        
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end 
        redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], model_params["dim_redux"], identity))
        encoder = Flux.Chain(enc_hls..., redux_layer)
        dec_hls = []
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : hl_sizes[i - 1]
            out_size = hl_sizes[i]
            push!(dec_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end
        output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["insize"], leakyrelu))
        decoder = Flux.Chain(dec_hls..., output_layer)
        ae_chain = Flux.Chain(enc_hls..., redux_layer, dec_hls..., output_layer)
        AE = AE_model(ae_chain, encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2) 
        cphdnn = gpu(Flux.Chain(Dense(model_params["dim_redux"]  + model_params["nb_clinf"] , model_params["cph_hl_size"], leakyrelu),
        #Dense(model_params["cph_hl_size"] ,model_params["cph_hl_size"], leakyrelu),
        Dense(model_params["cph_hl_size"] ,1, sigmoid; bias =false)))#, model_params["cph_hl_size"], relu),
        cph_opt = Flux.ADAM(model_params["cph_lr"])
        enccphdnn_model = enccphdnn(encoder, cphdnn, cph_opt, cox_l2)
        model = mtcphAE(AE, enccphdnn_model, AE.encoder)
    elseif model_params["model_type"] == "auto_encoder"
        c = compute_c(model_params["insize"], model_params["dim_redux"], model_params["enc_nb_hl"] )
        enc_hls = []
        hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
        hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
            out_size = reverse(hl_sizes)[i]
            push!(enc_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end 
        redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], model_params["dim_redux"], leakyrelu))
        encoder = Flux.Chain(enc_hls..., redux_layer)
        dec_hls = []
        for i in 1:model_params["enc_nb_hl"]
            in_size = i == 1 ? model_params["dim_redux"] : hl_sizes[i - 1]
            out_size = hl_sizes[i]
            push!(dec_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
        end
        output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["insize"], leakyrelu))
        decoder = Flux.Chain(dec_hls..., output_layer)
        ae_chain = Flux.Chain(enc_hls..., redux_layer, dec_hls..., output_layer)
        model = AE_model(ae_chain, encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2)        
    end 
    return model 
end

function build_internal_cph(model_params)
    cph_chain = gpu(Chain(Dense(model_params["dim_redux"] + model_params["nb_clinf"] , model_params["cph_hl_size"], leakyrelu),
    #Dense(model_params["cph_hl_size"] , model_params["cph_hl_size"], leakyrelu),
    Dense(model_params["cph_hl_size"] , 1, sigmoid, bias = false)))
    opt = Flux.ADAM(model_params["cph_lr"])
    return dnn(cph_chain, opt, cox_l2)
end 

function build_internal_dnn(encoder, model_params)
    hls = Flux.Chain(Flux.Dense(model_params["dim_redux"], model_params["clfdnn_hl_size"], model_params["n.-lin"]))
    clf_chain = gpu(Chain(encoder..., hls..., Dense( model_params["clfdnn_hl_size"], model_params["outsize"], identity)))
    clf_opt = Flux.ADAM(model_params["clfdnn_lr"])
    clf_lossf = crossentropy_l2
    return dnn(clf_chain, clf_opt, clf_lossf)
end 
function build_ae_cph_dnn(model_params)
    encoder = build_encoder(model_params)
    decoder, output_layer = build_decoder(model_params)
    dnn_chain = build_internal_dnn(encoder, model_params)
    cph =  build_internal_cph( model_params)
    AE = AE_model(Flux.Chain(encoder..., decoder...), encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2)   
    aecphdnn = Dict(  "enc"=> encoder, 
                        "cph"=> cph,
                        "dnn"=> dnn_chain,
                        "ae" => AE)  
    return aecphdnn
end 

function build_vaecox(params_dict)
    encoder = VariationalEncoder(params_dict["insize"], params_dict["dim_redux"], params_dict["venc_hl_size"]) # chain 
    decoder = Decoder(params_dict["insize"], params_dict["dim_redux"], params_dict["venc_hl_size"]) # chain 
    cph = gpu(Chain(Dense(params_dict["dim_redux"], params_dict["cph_hl_size"], leakyrelu), Dense(params_dict["cph_hl_size"], params_dict["cph_hl_size"],leakyrelu), Dense(params_dict["cph_hl_size"], 1))) # chain 
    return Dict("venc" => encoder,
    "vdec" => decoder,
    "cph" => cph )
end

function build_aecox(model_params)
    encoder = build_encoder(model_params)
    decoder, output_layer = build_decoder(model_params)
    #dnn_chain = build_internal_dnn(encoder, model_params)
    cph =  build_internal_cph( model_params)
    AE = AE_model(Flux.Chain(encoder..., decoder...), encoder, decoder, output_layer, Flux.ADAM(model_params["ae_lr"]), mse_l2)   
    vaecox = Dict(  "enc"=> encoder, 
                        "cph"=> cph,
     #                   "dnn"=> dnn_chain,
                        "ae" => AE)  
    return vaecox
end 

function build_decoder(model_params)
    enc, hl_sizes = build_internal_layers(model_params)
    dec_hls = []
    for i in 1:model_params["enc_nb_hl"]
        in_size = i == 1 ? model_params["dim_redux"] : hl_sizes[i - 1]
        out_size = hl_sizes[i]
        push!(dec_hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
    end
    output_layer = gpu(Flux.Dense(hl_sizes[end], model_params["insize"], leakyrelu))
    return Flux.Chain(dec_hls..., output_layer), output_layer
end 

function build_encoder(model_params; adaptative=true)
    enc_hls, hl_sizes = build_internal_layers(model_params,adaptative=adaptative)
    redux_layer = gpu(Flux.Dense(reverse(hl_sizes)[end], model_params["dim_redux"],identity))
    return Flux.Chain(enc_hls..., redux_layer)
end 

function build_internal_layers(model_params;adaptative=true)
    c = compute_c(model_params["insize"], model_params["dim_redux"], model_params["enc_nb_hl"] )
    hls = []
    hl_sizes = [Int(floor(model_params["dim_redux"] * c ^ x)) for x in 1:model_params["enc_nb_hl"]]
    hl_sizes = adaptative ?  hl_sizes  : Array{Int}(ones(10) .* model_params["ae_hl_size"])
    for i in 1:model_params["enc_nb_hl"]
        in_size = i == 1 ? model_params["insize"] : reverse(hl_sizes)[i - 1]
        out_size = reverse(hl_sizes)[i]
        push!(hls, gpu(Flux.Dense(in_size, out_size, leakyrelu)))
    end 
    return hls, hl_sizes
end 

###### Train loop functions
function train!(model::AE_model, fold;nepochs = 500, batchsize = 500, wd = 1e-6)
    ## Vanilla Auto-Encoder training function 
    train_x = fold["train_x"]';
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    for iter in 1:nepochs
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_ = gpu(train_x[:,mb_ids])
        lossval = model.lossf(model, X_, X_, weight_decay = 1e-6)
        ps = Flux.params(model.net)
        gs = gradient(ps) do
            model.lossf(model, X_, X_, weight_decay = 1e-6)
        end
        Flux.update!(model.opt, ps, gs)
        # println(my_cor(vec(X_), vec(model.net(X_))))
    end
end 
function train!(model::mtl_AE, fold, dump_cb, params)
    ## mtliative Auto-Encoder + Classifier NN model training function 
    ## Vanilla Auto-Encoder training function 
    batchsize = params["mb_size"]
    nepochs= params["nepochs"]
    wd = params["wd"]
    train_x = fold["train_x"]';
    train_y = fold["train_y"]';
    train_x_gpu = gpu(train_x)
    train_y_gpu = gpu(train_y)
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    # dump init state
    learning_curve = []
    ae_loss = model.ae.lossf(model.ae, gpu(train_x), gpu(train_x), weight_decay = wd)
    #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
    ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
    clf_loss = model.clf.lossf(model.clf, gpu(train_x), gpu(train_y), weight_decay = wd)
    clf_acc = accuracy(gpu(train_y), model.clf.model(gpu(train_x)))
    push!(learning_curve, (ae_loss, ae_cor, clf_loss, clf_acc))
    params["tr_acc"] = accuracy(gpu(train_y), model.clf.model(gpu(train_x)))
    dump_cb(model, learning_curve, params, 0, fold)
    for iter in ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_, Y_ = gpu(train_x[:,mb_ids]), gpu(train_y[:,mb_ids])
        ## gradient Auto-Encoder 
        ps = Flux.params(model.ae.net)
        gs = gradient(ps) do
            model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
        end
        Flux.update!(model.ae.opt, ps, gs)
        ## gradient Classifier
        ps = Flux.params(model.clf.model)
        gs = gradient(ps) do
            model.clf.lossf(model.clf, X_, Y_, weight_decay = wd)
        end
        Flux.update!(model.clf.opt, ps, gs)
        ae_loss = model.ae.lossf(model.ae, X_, X_, weight_decay = wd)
        #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
        ae_cor =  my_cor(vec(X_), vec(model.ae.net(gpu(X_))))
        clf_loss = model.clf.lossf(model.clf, X_, Y_, weight_decay = wd)
        clf_acc = accuracy(Y_, model.clf.model(X_))
        params["tr_acc"] = accuracy(train_y_gpu, model.clf.model(train_x_gpu))
        push!(learning_curve, (ae_loss, ae_cor, clf_loss, clf_acc))
        # save model (bson) every epoch if specified 
        dump_cb(model, learning_curve, params, iter, fold)
        #println("$iter\t AE-loss: $ae_loss\t AE-cor: $ae_cor\t CLF-loss: $clf_loss\t CLF-acc: $clf_acc")
    end
    return params["tr_acc"]
end 

function train!(model::mtl_FE, fold; nepochs = 1000, batchsize=500, wd = 1e-6)
    fe_x, fe_y = prep_FE(model.fe_data);
    nminibatches = Int(floor(length(fe_y) / batchsize))
    nsamples = length(tcga_prediction.rows)
    for iter in ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, length(fe_y)))
        X_i, Y_i = (fe_x[1][mb_ids],fe_x[2][mb_ids]), fe_y[mb_ids];
        model.fe_data
        lossval_fe = model.fe.lossf(model.fe, X_i, Y_i, weight_decay = wd)
        ps = Flux.params(model.fe.net)
        gs = gradient(ps) do 
            model.fe.lossf(model.fe, X_i, Y_i, weight_decay = wd)
        end
        Flux.update!(model.fe.opt,ps,gs)
        corr = my_cor(model.fe.net(X_i), Y_i)

        # training classes
        Yc = gpu(fold["train_y"]')
        Xc = gpu(fold["train_ids"])

        # gradient on classif
        ps = Flux.params(model.clf.model)
        grads = gradient(ps) do 
            model.clf.lossf(model.clf, Xc, Yc)
        end 
        Flux.update!(model.clf.opt, ps, grads)
        lossval_clf = model.clf.lossf(model.clf, Xc, Yc)
        acc = accuracy(Yc, model.clf.model(Xc))

        #println("$iter, FE-loss: $lossval_fe, FE-acc: $corr, CLF-loss: $lossval_clf, CLF-acc: $acc")
    end
end 


function train!(model::dnn, fold; nepochs = 1000, batchsize=500, wd = 1e-6)
    train_x = fold["train_x"]';
    train_y = fold["train_y"]';
    nsamples = size(train_y)[2]
    nminibatches = Int(floor(nsamples/ batchsize))
    lossf = model.lossf
    for iter in ProgressBar(1:nepochs)
        cursor = (iter -1)  % nminibatches + 1 
        mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
        X_, Y_ = gpu(train_x[:,mb_ids]), gpu(train_y[:,mb_ids])
        
        loss_val = lossf(model, X_, Y_, weight_decay = wd)
        ps = Flux.params(model.model)
        gs = gradient(ps) do
            lossf(model,X_, Y_, weight_decay = wd)
        end
        Flux.update!(model.opt, ps, gs)
        # println(accuracy(gpu(train_y), model.model(gpu(train_x))))
    end 
    return accuracy(gpu(train_y), model.model(gpu(train_x)))
end 

function train!(model::logistic_regression, fold; batchsize = 500, nepochs = 1000, wd = 1e-6)
    train_x = gpu(fold["train_x"]');
    train_y = gpu(fold["train_y"]');
    lossf = model.lossf
    for e in ProgressBar(1:nepochs) 
        loss_val = lossf(model, train_x, train_y, weight_decay = wd)
        ps = Flux.params(model.model)
        gs = gradient(ps) do
            lossf(model,train_x, train_y, weight_decay = wd)
        end
        Flux.update!(model.opt, ps, gs)
        #println(accuracy(model.model, train_x, train_y))
    end 
    return accuracy(train_y, model.model(train_x))
end 
####### Inference functions
function test(model::mtl_AE, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.clf.model(test_x)) 
end 

function test(model::logistic_regression, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.model(test_x)) 
end 
function test(model::dnn, fold)
    test_x = gpu(fold["test_x"]');
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.model(test_x)) 
end
function test(model::mtl_FE, fold)
    test_x = gpu(fold["test_ids"]);
    test_y = gpu(fold["test_y"]');
    return cpu(test_y), cpu(model.clf.model(test_x)) 
end


##### Validation functions 

function label_binarizer(labels::Array)
    lbls = unique(labels)
    n = length(labels)
    m = length(lbls)
    binarizer = Array{Bool, 2}(undef, (n, m))
    for s in 1:n
        binarizer[s,:] = lbls .== labels[s]
    end 
    return binarizer
end 


function accuracy(model, X, Y)
    n = size(X)[2]
    preds = model(X) .== maximum(model(X), dims = 1)
    acc = Y .& preds
    pct = sum(acc) / n
    return pct
end 

# function accuracy(X_true, X_pred)
#     preds = (X_pred .== maximum(X_pred, dims=1)) 
#     TP = sum(sum((preds .== X_true) .&& (preds .== 1),dims = 1) )
#     return 100 * TP / size(X_true)[2] 
# end 



function accuracy(true_labs, pred_labs)
    n = size(true_labs)[2]
    preds = pred_labs .== maximum(pred_labs, dims = 1)
    acc = true_labs .& preds
    pct = sum(acc) / n
    return pct
end 

function my_cor(X::AbstractVector, Y::AbstractVector)
    sigma_X = std(X)
    sigma_Y = std(Y)
    mean_X = mean(X)
    mean_Y = mean(Y)
    cov = sum((X .- mean_X) .* (Y .- mean_Y)) / length(X)
    return cov / sigma_X / sigma_Y
end 

function split_train_test(X::Matrix, targets; nfolds = 5)
    folds = Array{Dict, 1}(undef, nfolds)
    nsamples = size(X)[1]
    fold_size  = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples)
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds 
        tst_ids = shuffled_ids[collect((i-1) * fold_size +1: min(nsamples, i * fold_size))]
        tr_ids = setdiff(ids, tst_ids)
        train_x = X[tr_ids,:]
        train_y = targets[tr_ids, :]
        test_x = X[tst_ids, :]
        test_y = targets[tst_ids, :]
        folds[i] = Dict("foldn" => i, "train_x"=> train_x, "train_ids"=>tr_ids, "train_y" =>train_y,"test_x"=> test_x, "test_ids" =>tst_ids,"test_y" => test_y )
    end
    return folds  
end

function bootstrap(acc_function, tlabs, plabs; bootstrapn = 1000)
    nsamples = sum([size(tbl)[2] for tbl in tlabs])
    tlabsm = hcat(tlabs...);
    plabsm = hcat(plabs...);
    accs = []
    for i in 1:bootstrapn
        sample = rand(1:nsamples, nsamples);
        push!(accs, acc_function(tlabsm[:,sample], plabsm[:,sample]))
    end 
    sorted_accs = sort(accs)

    low_ci, med, upp_ci = sorted_accs[Int(round(bootstrapn * 0.025))], median(sorted_accs), sorted_accs[Int(round(bootstrapn * 0.975))]
    return low_ci, med, upp_ci
end 
####### CAllback functions
function to_cpu(model::AE_AE_DNN)
    return AE_AE_DNN(cpu(model.ae), cpu(model.clf), cpu(model.encoder),cpu(model.ae2d))
end 
function to_cpu(model::mtl_AE)
    return mtl_AE(cpu(model.ae), cpu(model.clf), cpu(model.encoder))
end 
function to_cpu(model::dnn)
    return dnn(cpu(model.model), model.opt, model.lossf)
end

function to_cpu(model::mtcphAE)
    return mtcphAE(cpu(model.ae), cpu(model.cph), cpu(model.encoder))
end 
function to_cpu(model::enccphdnn)
    return enccphdnn(cpu(model.encoder),cpu(model.cphdnn), model.opt, model.lossf)
end 
function to_cpu(model::AE_model)
    return AE_model(cpu(model.net),cpu(model.encoder), cpu(model.decoder), cpu(model.outpl), model.opt, model.lossf)
end 


# define dump call back 
function dump_model_cb(dump_freq, labels; export_type = "png")
    return (model, tr_metrics, params_dict, iter::Int, fold) -> begin 
        # check if end of epoch / start / end 
        if iter % dump_freq == 0 || iter == 0 || iter == params_dict["nepochs"]
            model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
            # saves model BUGGED 
            # bson("RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad =3))/model_$(zpad(iter)).bson", Dict("model"=>to_cpu(model)))
            # plot learning curve
            lr_fig_outpath = "RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"
            plot_learning_curves_aeclf(tr_metrics, params_dict, lr_fig_outpath)
            # plot embedding
            X_tr = cpu(model.encoder(gpu(fold["train_x"]')))
            X_tst = cpu(model.encoder(gpu(fold["test_x"]')))
            
            tr_lbls = labels[fold["train_ids"]]
            tst_lbls = labels[fold["test_ids"]]
            emb_fig_outpath = "RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/model_$(zpad(iter)).$export_type"
            plot_embed(X_tr, X_tst, tr_lbls, tst_lbls,  params_dict, emb_fig_outpath;acc="clf_tr_acc")
            #fig = Figure(resolution = (1024,1024));
            #ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", title = "Predicted vs True of $(brca_ae_params["ngenes"]) Genes Expression Profile TCGA BRCA with AE \n$(round(ae_cor_test;digits =3))", aspect = DataAspect())
            #hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
            #CairoMakie.save("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/1B_AE_BRCA_AE_SCATTER_DIM_REDUX.pdf", fig)

        end 
    end 
end 

function dump_aecphclf_model_cb(dump_freq, labels; export_type = "png")
    return (model, tr_metrics, params_dict, iter::Int, fold) -> begin 
        # check if end of epoch / start / end 
        if iter % dump_freq == 0 || iter == 0 || iter == params_dict["nepochs"]
            model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
            # saves model BUGGED
            # bson("RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad =3))/model_$(zpad(iter)).bson", Dict("model"=>to_cpu(model)))
            # plot learning curve
            lr_fig_outpath = "RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"
            plot_learning_curves_aecphclf(tr_metrics, params_dict, lr_fig_outpath)
            # plot embedding
            #X_proj = Matrix(cpu(model.ae2d.encoder(model.encoder(gpu(fold["train_x"]')))'))
            #tr_labels = labels[fold["train_ids"]]
            #tr_embed = DataFrame(:emb1=>X_proj[:,1], :emb2=>X_proj[:,2], :cancer_type => tr_labels)
            #train = AlgebraOfGraphics.data(tr_embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type) * visual(markersize =20)
            #tr_acc,tst_acc = tr_metrics[end][2], tr_metrics[end][8]
            #fig = draw(train, axis = (;aspect = AxisAspect(1), autolimitaspect = 1, width = 1024, height =1024,
            #title="$(params_dict["model_type"]) on $(params_dict["dataset"]) data\naccuracy by DNN TRAIN: $(round(tr_acc* 100, digits=2))% TEST: $(round(tst_acc*100, digits=2))%"))

            #emb_fig_outpath = "RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad=3))/model_$(zpad(iter)).$export_type"
            #CairoMakie.save(emb_fig_outpath, fig)
            #plot_embed(X_tr, X_tst, tr_lbls, tst_lbls,  params_dict, emb_fig_outpath;acc="clf_tr_acc")
            #fig = Figure(resolution = (1024,1024));
            #ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", title = "Predicted vs True of $(brca_ae_params["ngenes"]) Genes Expression Profile TCGA BRCA with AE \n$(round(ae_cor_test;digits =3))", aspect = DataAspect())
            #hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
            #CairoMakie.save("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/1B_AE_BRCA_AE_SCATTER_DIM_REDUX.pdf", fig)

        end 
    end 
end 

function dump_aeaeclfdnn_model_cb(dump_freq, labels; export_type = "png")
    return (model, tr_metrics, params_dict, iter::Int, fold) -> begin 
        # check if end of epoch / start / end 
        if iter % dump_freq == 0 || iter == 0 || iter == params_dict["nepochs"]
            model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
            # saves model BUGGED
            # bson("RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad =3))/model_$(zpad(iter)).bson", Dict("model"=>to_cpu(model)))
            # plot learning curve
            lr_fig_outpath = "RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"
            plot_learning_curves_aeaeclf(tr_metrics, params_dict, lr_fig_outpath)
            # plot embedding
            X_proj = Matrix(cpu(model.ae2d.encoder(model.encoder(gpu(fold["train_x"]')))'))
            tr_labels = labels[fold["train_ids"]]
            tr_embed = DataFrame(:emb1=>X_proj[:,1], :emb2=>X_proj[:,2], :cancer_type => tr_labels)
            train = AlgebraOfGraphics.data(tr_embed) * mapping(:emb1,:emb2,color = :cancer_type,marker = :cancer_type) * visual(markersize =20)
            tr_acc,tst_acc = tr_metrics[end][2], tr_metrics[end][8]
            fig = draw(train, axis = (;aspect = AxisAspect(1), autolimitaspect = 1, width = 1024, height =1024,
            title="$(params_dict["model_type"]) on $(params_dict["dataset"]) data\naccuracy by DNN TRAIN: $(round(tr_acc* 100, digits=2))% TEST: $(round(tst_acc*100, digits=2))%"))

            emb_fig_outpath = "RES/$model_params_path/FOLD$(zpad(fold["foldn"],pad=3))/model_$(zpad(iter)).$export_type"
            CairoMakie.save(emb_fig_outpath, fig)
            #plot_embed(X_tr, X_tst, tr_lbls, tst_lbls,  params_dict, emb_fig_outpath;acc="clf_tr_acc")
            #fig = Figure(resolution = (1024,1024));
            #ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", title = "Predicted vs True of $(brca_ae_params["ngenes"]) Genes Expression Profile TCGA BRCA with AE \n$(round(ae_cor_test;digits =3))", aspect = DataAspect())
            #hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
            #CairoMakie.save("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/1B_AE_BRCA_AE_SCATTER_DIM_REDUX.pdf", fig)

        end 
    end 
end 
# define dump call back 
function dump_aecphdnn_model_cb(dump_freq, labels; export_type = "png")
    return (model, tr_metrics, params_dict, iter::Int, fold) -> begin 
        # check if end of epoch / start / end 
        if iter % dump_freq == 0 || iter == 0 || iter == params_dict["nepochs"]
            # saves model
            bson("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad =3))/model_$(zpad(iter)).bson", Dict("model"=>to_cpu(model)))
            # plot learning curve
            lr_fig_outpath = "RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))_lr.pdf"
            plot_learning_curves(tr_metrics, params_dict, lr_fig_outpath)
            # plot embedding
            X_tr = cpu(model.encoder(gpu(fold["train_x"]')))
            X_tst = cpu(model.encoder(gpu(fold["test_x"]')))
            
            tr_lbls = labels[fold["train_ids"]]
            tst_lbls = labels[fold["test_ids"]]
            emb_fig_outpath = "RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/model_$(zpad(iter)).$export_type"
            #plot_embed(X_tr, X_tst, tr_lbls, tst_lbls,  params_dict, emb_fig_outpath;acc="clf_tr_acc")
            #fig = Figure(resolution = (1024,1024));
            #ax = Axis(fig[1,1];xlabel="Predicted", ylabel = "True Expr.", title = "Predicted vs True of $(brca_ae_params["ngenes"]) Genes Expression Profile TCGA BRCA with AE \n$(round(ae_cor_test;digits =3))", aspect = DataAspect())
            #hexbin!(fig[1,1], outs, test_xs, cellsize=(0.02, 0.02), colormap=cgrad([:grey,:yellow], [0.00000001, 0.1]))
            #CairoMakie.save("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(fold["foldn"],pad=3))/1B_AE_BRCA_AE_SCATTER_DIM_REDUX.pdf", fig)

        end 
    end 
end 
function dummy_dump_cb(model, tr_metrics, params, iter::Int, fold) end 

####### cross validation loops 

function validate!(params, Data, dump_cb)
    # init 
    mkdir("RES/$(params["session_id"])/$(params["modelid"])")
    # init results lists 
    true_labs_list, pred_labs_list = [],[]
    # create fold directories
    [mkdir("RES/$(params["session_id"])/$(params["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params["nfolds"]]
    # splitting, dumped 
    folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
    dump_folds(folds, params, Data.rows)
    # dump params
    bson("RES/$(params["session_id"])/$(params["modelid"])/params.bson", params)

    # start crossval
    for (foldn, fold) in enumerate(folds)
        model = build(params)
        train_metrics = train!(model, fold, dump_cb, params)
        true_labs, pred_labs = test(model, fold)
        push!(true_labs_list, true_labs)
        push!(pred_labs_list, pred_labs)
        println("train: ", train_metrics)
        println("test: ", accuracy(true_labs, pred_labs))
        params["tst_acc"] = accuracy(true_labs, pred_labs)
        plot_embed(cpu(model.ae.encoder(gpu(fold["test_x"]'))), 
            Data.targets[fold["test_ids"]], 
            params,
            "RES/$(params["session_id"])/$(params["modelid"])/fold_$(foldn)_tst.pdf",
            acc = "tst_acc")
        # post run 
        # save model
        # save 2d embed svg
        # training curves svg, csv 
    end
    ### bootstrap results get 95% conf. interval 
    low_ci, med, upp_ci = bootstrap(accuracy, true_labs_list, pred_labs_list) 
    ### returns a dict 
    ret_dict = Dict("cv_acc_low_ci" => low_ci,
    "cv_acc_upp_ci" => upp_ci,
    "cv_acc_median" => med
    )
    params["cv_acc_low_ci"] = low_ci
    params["cv_acc_median"] = med
    params["cv_acc_upp_ci"] = upp_ci

    # param dict 
    return ret_dict


end 
######### Fit transform functions
function fit_transform!(model::mtl_AE, fold, dump_cb, params::Dict)
    #### trains a model following params dict on given Data 
    #### returns data transformation using AE
    #### rerurns accuracy metrics  
    X = Data.data
    targets = label_binarizer(Data.targets)
    learning_curves = train!(model, fold, dump_cb, params)
    return model.clf.model(gpu(X')), learning_curves
end  