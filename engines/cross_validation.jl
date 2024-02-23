


function validate_clfdnn!(params_dict, x_data, y_data, samples;build_adaptative=false,nfolds=5,device =gpu)
    folds = split_train_test(x_data, y_data, samples;nfolds = nfolds)
    mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])")
    x_pred_by_fold, test_xs = [],[]
    [mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/params.bson",params_dict)
    for fold in folds
        model = build(params_dict;adaptative=build_adaptative)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        wd = params_dict["wd"]
        train_x = device(Matrix(fold["train_x"]'));
        train_y = device(Matrix(fold["train_y"]'));
        test_x = device(Matrix(fold["test_x"]'));
        test_y = device(Matrix(fold["test_y"]'));

        nsamples = size(train_y)[2]    
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            ## gradient Auto-Encoder 
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                model.lossf(model, train_x, train_y, weight_decay = wd)
            end
            Flux.update!(model.opt, ps, gs)
            train_loss = round(model.lossf(model, train_x, train_y, weight_decay = wd),digits=3)
            train_acc =  round(accuracy(train_y, model.model(train_x)),digits = 3)
            params_dict["tr_acc"] = train_acc
            test_loss = round(model.lossf(model, test_x, test_y, weight_decay = wd), digits = 3)
            test_acc = round(accuracy(test_y, model.model(test_x)),digits = 3)
            params_dict["tst_cor"] = test_acc
            push!(learning_curve, (train_loss, test_acc, test_loss, test_acc))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN loss $(round(train_loss,digits =3)) \t acc.%: $(round(train_acc, digits = 3))\t TEST loss $(round(test_loss,digits =3)) \t acc.%: $(round(test_acc, digits = 3))")
             #dump_cb_brca(model, learning_curve, params_dict, iter, fold)
        end
        push!(x_pred_by_fold, Matrix(cpu(model.model(test_x))'))
        push!(test_xs, Matrix(cpu(test_y)'))
    end
    concat_OUTs = Matrix(vcat(x_pred_by_fold...)')
    concat_tests = Matrix(vcat(test_xs...)')

    return concat_OUTs, concat_tests#Dict(:tr_acc=>accuracy(concat_tests,concat_OUTs))
end 


function validate_aeclfdnn!(params_dict, x_data, y_data, samples, dump_cb_brca;build_adaptative=false,nfolds=5,device =gpu)
    folds = split_train_test(x_data, y_data, samples;nfolds = nfolds)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    train_y_pred_by_fold, y_pred_by_fold, test_ys, train_ys = [],[], [], []
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build(params_dict;adaptative=build_adaptative)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        wd = params_dict["wd"]
        train_x = device(Matrix(fold["train_x"]'));
        train_y = device(Matrix(fold["train_y"]'));
        test_x = device(Matrix(fold["test_x"]'));
        test_y = device(Matrix(fold["test_y"]'));

        nsamples = size(train_y)[2]    
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            ## gradient Auto-Encoder 
            ps = Flux.params(model.ae.net)
            gs = gradient(ps) do
                model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd)
            end
            Flux.update!(model.ae.opt, ps, gs)
            ## gradient Classfier DNN
            ps = Flux.params(model.clf.model)
            gs = gradient(ps) do 
                model.clf.lossf(model.clf, train_x, train_y, weight_decay = wd)
            end
            Flux.update!(model.clf.opt, ps, gs)

            train_clf_loss = round(model.clf.lossf(model.clf, train_x, train_y, weight_decay = wd),digits=3)
            train_clf_acc =  round(accuracy(train_y, model.clf.model(train_x)),digits = 3)
            train_ae_loss = round(model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd),digits=3)
            train_ae_cor = round(my_cor(vec(train_x), vec(model.ae.net(train_x))),digits=3)
            
            test_clf_loss = round(model.clf.lossf(model.clf, test_x, test_y, weight_decay = wd), digits = 3)
            test_clf_acc = round(accuracy(test_y, model.clf.model(test_x)),digits = 3)
            test_ae_loss =  round(model.ae.lossf(model.ae, test_x, test_x, weight_decay = wd),digits=3)
            test_ae_cor = round(my_cor(vec(test_x), vec(model.ae.net(test_x))),digits=3)
            
            push!(learning_curve, (train_clf_loss, train_clf_acc, train_ae_loss, train_ae_cor, test_clf_loss, test_clf_acc, test_ae_loss, test_ae_cor))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN CLF loss $(round(train_clf_loss,digits =3)) \t acc.%: $(round(train_clf_acc, digits = 3))\tAE loss: $train_ae_loss \tcor: $train_ae_cor\t TEST CLF loss $(round(test_clf_loss,digits =3)) \t acc.%: $(round(test_clf_acc, digits = 3)) AE loss: $test_ae_loss \t cor $test_ae_cor")
            
            dump_cb_brca(model, learning_curve, params_dict, iter, fold)
            
        end
        if learning_curve[end][6] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][6]
            best_model_fold = fold
            params_dict["bm_tst_acc"] = round(best_accuracy, digits =3 )
        end 
        ## tst accuracies 
        push!(y_pred_by_fold, Matrix(cpu(model.clf.model(test_x))'))
        push!(test_ys, Matrix(cpu(test_y)'))
        ## train accuracies
        push!(train_y_pred_by_fold, Matrix(cpu(model.clf.model(train_x))'))
        push!(train_ys, Matrix(cpu(train_y)'))
    end
    outs_test = Matrix(vcat(y_pred_by_fold...)')
    y_test = Matrix(vcat(test_ys...)')
    outs_train = Matrix(vcat(train_y_pred_by_fold...)')
    y_train = Matrix(vcat(train_ys...)')
    model_tr_acc = accuracy(y_train, outs_train)
    model_tst_acc = accuracy(y_test, outs_test)
    params_dict["clf_tr_acc"] = model_tr_acc
    params_dict["clf_tst_acc"] = model_tst_acc
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    if params_dict["dim_redux"] == 2
        ## 2d scatter 
        plot_embed_train_test_2d(best_model,best_model_fold, params_dict)
    end 
    if params_dict["dim_redux"] == 3
        ## 3d scatter 
        plot_embed_train_test_3d(best_model,best_model_fold, params_dict)
    end 
    ## hexbin               
    plot_hexbin_pred_true_aeclfdnn(best_model, best_model_fold, params_dict)
    return best_model, best_model_fold, outs_test, y_test, outs_train, y_train   
end 

function validate_aeaeclfdnn!(params_dict, x_data, y_data, samples, dump_cb_brca;build_adaptative=false,nfolds=5,device =gpu)
    folds = split_train_test(x_data, y_data, samples;nfolds = nfolds)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    train_y_pred_by_fold, y_pred_by_fold, test_ys, train_ys, tr_cor, tst_cor, ae2_tr_cor, ae2_tst_cor= [],[], [], [],[],[],[],[]
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build(params_dict;adaptative=build_adaptative)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        wd = params_dict["wd"]
        train_x = device(Matrix(fold["train_x"]'));
        train_y = device(Matrix(fold["train_y"]'));
        test_x = device(Matrix(fold["test_x"]'));
        test_y = device(Matrix(fold["test_y"]'));

        nsamples = size(train_y)[2]    
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            ## gradient Auto-Encoder 
            ps1 = Flux.params(model.ae.net)
            gs1 = gradient(ps1) do
                model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd)
            end
            
            ## gradient Classfier DNN
            ps2 = Flux.params(model.clf.model)
            gs2 = gradient(ps2) do 
                model.clf.lossf(model.clf, train_x, train_y, weight_decay = wd)
            end
            
            ## gradient 2D Auto-Encoder for tracking.
            ps3 = Flux.params(model.ae2d.net)
            gs3 = gradient(ps3) do 
                model.ae2d.lossf(model.ae2d, model.ae.encoder(train_x), model.ae.encoder(train_x), weight_decay = wd)
            end
            ## updates
            Flux.update!(model.ae.opt, ps1, gs1)
            Flux.update!(model.clf.opt, ps2, gs2)
            Flux.update!(model.ae2d.opt, ps3, gs3)

            train_clf_loss = round(model.clf.lossf(model.clf, train_x, train_y, weight_decay = wd),digits=3)
            train_clf_acc =  round(accuracy(train_y, model.clf.model(train_x)),digits = 3)
            train_ae_loss = round(model.ae.lossf(model.ae, train_x, train_x, weight_decay = wd),digits=3)
            train_ae_cor = round(my_cor(vec(train_x), vec(model.ae.net(train_x))),digits=3)
            train_ae2_loss = round(model.ae2d.lossf(model.ae2d, model.ae.encoder(train_x), model.ae.encoder(train_x), weight_decay = wd),digits=3)
            train_ae2_cor = round(my_cor(vec(model.ae.encoder(train_x)), vec(model.ae2d.net(model.ae.encoder(train_x)))),digits=3)

            test_clf_loss = round(model.clf.lossf(model.clf, test_x, test_y, weight_decay = wd), digits = 3)
            test_clf_acc = round(accuracy(test_y, model.clf.model(test_x)),digits = 3)
            test_ae_loss =  round(model.ae.lossf(model.ae, test_x, test_x, weight_decay = wd),digits=3)
            test_ae_cor = round(my_cor(vec(test_x), vec(model.ae.net(test_x))),digits=3)
            test_ae2_loss = round(model.ae2d.lossf(model.ae2d, model.ae.encoder(test_x), model.ae.encoder(test_x), weight_decay = wd),digits=3)
            test_ae2_cor = round(my_cor(vec(model.ae.encoder(test_x)), vec(model.ae2d.net(model.ae.encoder(test_x)))),digits=3)

            push!(learning_curve, (train_clf_loss, train_clf_acc, train_ae_loss, train_ae_cor, train_ae2_loss, train_ae2_cor, test_clf_loss, test_clf_acc, test_ae_loss, test_ae_cor, test_ae2_loss,  test_ae2_cor))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN CLF loss $(round(train_clf_loss,digits =3)) \t acc.%: $(round(train_clf_acc, digits = 3))\tAE loss: $train_ae_loss \tcor: $train_ae_cor\t AE2D loss: $train_ae2_loss \t cor: $train_ae2_cor ")
            println("\t\tTEST CLF loss $(round(test_clf_loss,digits =3)) \t acc.%: $(round(test_clf_acc, digits = 3)) AE loss: $test_ae_loss \t cor $test_ae_cor \t AE2D loss: $test_ae2_loss \t cor: $test_ae2_cor")
            
            dump_cb_brca(model, learning_curve, params_dict, iter, fold)
            
        end
        if learning_curve[end][8] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][8]
            best_model_fold = fold
            params_dict["bm_tst_acc"] = round(best_accuracy, digits =3 )
        end 
        ## tst accuracies 
        push!(y_pred_by_fold, Matrix(cpu(model.clf.model(test_x))'))
        push!(test_ys, Matrix(cpu(test_y)'))
        ## train accuracies
        push!(train_y_pred_by_fold, Matrix(cpu(model.clf.model(train_x))'))
        push!(train_ys, Matrix(cpu(train_y)'))
        ## tr corr 
        push!(tr_cor, learning_curve[end][4])
        ## tst corr
        push!(tst_cor, learning_curve[end][10])
        ## AE2 tr corr 
        push!(ae2_tr_cor, learning_curve[end][6])
        ## AE2 tst corr
        push!(ae2_tst_cor, learning_curve[end][12])
        
    end
    outs_test = Matrix(vcat(y_pred_by_fold...)')
    y_test = Matrix(vcat(test_ys...)')
    outs_train = Matrix(vcat(train_y_pred_by_fold...)')
    y_train = Matrix(vcat(train_ys...)')
    model_tr_acc = accuracy(y_train, outs_train)
    model_tst_acc = accuracy(y_test, outs_test)
    params_dict["clf_tr_acc"] = model_tr_acc
    params_dict["clf_tst_acc"] = model_tst_acc
    params_dict["ae_tr_cor"] = mean(tr_cor)
    params_dict["ae_tst_cor"] = mean(tst_cor)
    params_dict["ae2_tr_cor"] = mean(ae2_tr_cor)
    params_dict["ae2_tst_cor"] = mean(ae2_tst_cor)
    
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    ## hexbin               
    plot_hexbin_pred_true_aeclfdnn(best_model, best_model_fold, params_dict)
    return best_model, best_model_fold, outs_test, y_test, outs_train, y_train   
end 



function validate_auto_encoder!(params_dict, brca_prediction, dump_cb_brca, clinf;build_adaptative = false, device=gpu)
    
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    train_x_pred_by_fold, x_pred_by_fold, test_xs, train_xs = [],[], [], []
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds

        model = build(params_dict;adaptative=build_adaptative)
        ## STATIC VARS    
        batchsize = params_dict["mb_size"]
        nepochs= params_dict["nepochs"]
        wd = params_dict["wd"]
        train_x = device(Matrix(fold["train_x"]'));
        test_x = device(Matrix(fold["test_x"]'));

        nsamples = size(train_x)[2]
        nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []
        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            #X_ = train_x[:,mb_ids]
            ## gradient Auto-Encoder 
            ps = Flux.params(model.net)
            gs = gradient(ps) do
                model.lossf(model, train_x, train_x, weight_decay = wd)
            end
            Flux.update!(model.opt, ps, gs)
            ae_loss = model.lossf(model, train_x, train_x, weight_decay = wd)
            #ae_cor = my_cor(vec(train_x), cpu(vec(model.ae.net(gpu(train_x)))))
            ae_cor =  round(my_cor(vec(train_x), vec(model.net(train_x))),digits = 3)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            ae_loss_test = round(model.lossf(model, test_x, test_x, weight_decay = wd), digits = 3)
            ae_cor_test = round(my_cor(vec(test_x), vec(model.net(test_x))), digits= 3)
            
            push!(learning_curve, (ae_loss, ae_cor, ae_loss_test, ae_cor_test))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t TEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))")
            dump_cb_brca(model, learning_curve, params_dict, iter, fold)
        end
        if learning_curve[end][4] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][4]
            best_model_fold = fold
            params_dict["bm_tst_cor"] = round(best_accuracy, digits =3 )
        end 
        ## test set correlations
        push!(x_pred_by_fold, cpu(vec(model.net(test_x))))
        push!(test_xs, cpu(vec(test_x)))
        ## train set correlations
        push!(train_x_pred_by_fold, cpu(vec(model.net(train_x))))
        push!(train_xs, cpu(vec(train_x)))
    end
    outs_test = vcat(x_pred_by_fold...)
    x_test = vcat(test_xs...)
    outs_train = vcat(train_x_pred_by_fold...)
    x_train = vcat(train_xs...)
    model_tr_corr = my_cor(outs_train, x_train)
    model_tst_corr = my_cor(outs_test, x_test)
    params_dict["ae_tr_corr"] = model_tr_corr
    params_dict["ae_tst_corr"] = model_tst_corr
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    ## hexbin               
    plot_hexbin_pred_true_ae(best_model, best_model_fold, params_dict)
    return best_model, best_model_fold, outs_test, x_test, outs_train, x_train 
end 

function validate_cphclinf!(params_dict, dataset, dump_cb, clinf; device = gpu)
    folds = split_train_test(Matrix(dataset[:tpm_data]), Matrix(clinf), dataset[:survt], dataset[:surve], dataset[:samples];nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold, train_scores_by_fold, test_scores_by_fold  = [],[], [], [],[],[]
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    
    for fold in folds
        model = build(params_dict)

        ## STATIC VARS    
        #batchsize = params_dict["mb_size"]
        nepochs= params_dict["nepochs"]
        #wd = params_dict["wd"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        #nsamples = size(train_y_t)[2]
        #nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            #cursor = (iter -1)  % nminibatches + 1
            #mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            #X_ = gpu(train_x[:,mb_ids])
            #X_c_ = gpu(train_x_c[:,mb_ids])
            ## gradient CPH
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                
                model.lossf(model,train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS_tr = vec(model.model(train_x_c))
            cph_loss =model.lossf(model, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x_c,  test_y_e, NE_frac_tst, params_dict["wd"]), digits= 3)
            OUTS_tst =  vec(model.model(test_x_c))
            cind_test, cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, OUTS_tst)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\tTEST cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
        end
        OUTS = cpu(vec(model.model(test_x_c)))
        if learning_curve[end][4] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][4]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ## train c-index
        push!(train_scores_by_fold, cpu(vec(model.model( train_x_c))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ## test c-index
        push!(test_scores_by_fold, cpu(vec(model.model( test_x_c))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))
        
    end 
    #### TESTS
    outs_test = vcat(test_scores_by_fold...)
    test_yt = vcat(test_y_t_by_fold...)
    test_ye = vcat(test_y_e_by_fold...)
    #### TRAIN
    outs_train = vcat(train_scores_by_fold...)
    train_yt = vcat(train_y_t_by_fold...)
    train_ye = vcat(train_y_e_by_fold...)
    # Cis = []
    # for i in 1:1000
    #     sampling = rand(1:length(concat_OUTS),length(concat_OUTS))
    #     push!(Cis, concordance_index(concat_yt[sampling], concat_ye[sampling], -1 .* concat_OUTS[sampling])[1])
    # end
    params_dict["cphdnn_tst_c_ind"] = concordance_index(test_yt, test_ye, -1 .* outs_test)[1]
    params_dict["cphdnn_train_c_ind"] = concordance_index(train_yt, train_ye, -1 .* outs_train)[1]
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    return best_model, best_model_fold, outs_test, test_yt, test_ye, outs_train, train_yt, train_ye 
    # groups = ["low_risk" for i in 1:length(concat_OUTS)]    
    # high_risk = concat_OUTS .> median(concat_OUTS)
    # low_risk = concat_OUTS .< median(concat_OUTS)
    # end_of_study = 365 * 10
    # groups[high_risk] .= "high_risk"
    # p_high, x_high, sc1_high, sc2_high = surv_curve(concat_yt[high_risk], concat_ye[high_risk]; color = "red")
    # p_low, x_low, sc1_low, sc2_low = surv_curve(concat_yt[low_risk], concat_ye[low_risk]; color = "blue")

    # lrt_pval = round(log_rank_test(concat_yt, concat_ye, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)
    # f = draw(p_high + x_high + p_low + x_low, axis = (;title = "CPHDNN Single Task - Predicted Low (blue) vs High (red) risk\nc-index: $(round(median(Cis), digits = 3))\nlog-rank-test pval: $lrt_pval"))
    # CairoMakie.save("$outpath/$(params_dict["modelid"])/low_vs_high_surv_curves.pdf",f)
    # c_tst = round(median(Cis), digits = 3)
    
end 

function validate_cphclinf_noexpr!(params_dict, dataset, dump_cb, clinf; device = gpu)
    folds = split_train_test(Matrix(dataset[:tpm_data]), Matrix(clinf), dataset[:survt], dataset[:surve], dataset[:samples];nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold, train_scores_by_fold, test_scores_by_fold  = [],[], [], [],[],[]
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    
    for fold in folds
        model = build(params_dict)

        ## STATIC VARS    
        #batchsize = params_dict["mb_size"]
        nepochs= params_dict["nepochs"]
        #wd = params_dict["wd"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        #nsamples = size(train_y_t)[2]
        #nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            #cursor = (iter -1)  % nminibatches + 1
            #mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            #X_ = gpu(train_x[:,mb_ids])
            #X_c_ = gpu(train_x_c[:,mb_ids])
            ## gradient CPH
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                
                model.lossf(model,train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS_tr = vec(model.model(train_x_c))
            cph_loss =model.lossf(model, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x_c,  test_y_e, NE_frac_tst, params_dict["wd"]), digits= 3)
            OUTS_tst =  vec(model.model(test_x_c))
            cind_test, cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, OUTS_tst)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\tTEST cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
        end
        OUTS = cpu(vec(model.model(test_x_c)))
        if learning_curve[end][4] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][4]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ## train c-index
        push!(train_scores_by_fold, cpu(vec(model.model( train_x_c))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ## test c-index
        push!(test_scores_by_fold, cpu(vec(model.model( test_x_c))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))
        
    end 
    #### TESTS
    outs_test = vcat(test_scores_by_fold...)
    test_yt = vcat(test_y_t_by_fold...)
    test_ye = vcat(test_y_e_by_fold...)
    #### TRAIN
    outs_train = vcat(train_scores_by_fold...)
    train_yt = vcat(train_y_t_by_fold...)
    train_ye = vcat(train_y_e_by_fold...)
    # Cis = []
    # for i in 1:1000
    #     sampling = rand(1:length(concat_OUTS),length(concat_OUTS))
    #     push!(Cis, concordance_index(concat_yt[sampling], concat_ye[sampling], -1 .* concat_OUTS[sampling])[1])
    # end
    params_dict["cphdnn_tst_c_ind"] = concordance_index(test_yt, test_ye, -1 .* outs_test)[1]
    params_dict["cphdnn_train_c_ind"] = concordance_index(train_yt, train_ye, -1 .* outs_train)[1]
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    return best_model, best_model_fold, outs_test, test_yt, test_ye, outs_train, train_yt, train_ye 
    # groups = ["low_risk" for i in 1:length(concat_OUTS)]    
    # high_risk = concat_OUTS .> median(concat_OUTS)
    # low_risk = concat_OUTS .< median(concat_OUTS)
    # end_of_study = 365 * 10
    # groups[high_risk] .= "high_risk"
    # p_high, x_high, sc1_high, sc2_high = surv_curve(concat_yt[high_risk], concat_ye[high_risk]; color = "red")
    # p_low, x_low, sc1_low, sc2_low = surv_curve(concat_yt[low_risk], concat_ye[low_risk]; color = "blue")

    # lrt_pval = round(log_rank_test(concat_yt, concat_ye, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)
    # f = draw(p_high + x_high + p_low + x_low, axis = (;title = "CPHDNN Single Task - Predicted Low (blue) vs High (red) risk\nc-index: $(round(median(Cis), digits = 3))\nlog-rank-test pval: $lrt_pval"))
    # CairoMakie.save("$outpath/$(params_dict["modelid"])/low_vs_high_surv_curves.pdf",f)
    # c_tst = round(median(Cis), digits = 3)
    
end 

function validate_aecphdnn!(params_dict, brca_prediction, dump_cb_brca, clinf;device=gpu)
    
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:6]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold = [],[], [], []
    test_scores_by_fold, train_scores_by_fold = [],[]
    train_x_pred_by_fold, test_x_pred_by_fold, test_xs, train_xs = [],[], [], []
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build(params_dict)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));

        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        nsamples = size(train_y_t)[2]
             
        learning_curve = []
        for iter in 1:nepochs
            ## gradient Auto-Encoder 
            ps = Flux.params(model.ae.net)
            gs = gradient(ps) do
                model.ae.lossf(model.ae, train_x, train_x, weight_decay = params_dict["ae_wd"])
            end
            Flux.update!(model.ae.opt, ps, gs)
            ## gradient CPH
            
            ps = Flux.params(model.cph.encoder, model.cph.cphdnn)
            gs = gradient(ps) do
                model.cph.lossf(model.cph, train_x, train_x_c, train_y_e, NE_frac_tr, params_dict["cph_wd"])
            end
            Flux.update!(model.cph.opt, ps, gs)
            OUTS_tr = vec(model.cph.cphdnn(vcat(model.cph.encoder(train_x), train_x_c)))
            ae_loss = model.ae.lossf(model.ae, train_x, train_x, weight_decay = params_dict["ae_wd"])
            ae_cor =  round(my_cor(vec(train_x), vec(model.ae.net(train_x))),digits = 3)
            cph_loss = model.cph.lossf(model.cph,train_x, train_x_c, train_y_e, NE_frac_tr, params_dict["cph_wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            ae_loss_test = round(model.ae.lossf(model.ae, test_x, test_x, weight_decay = params_dict["ae_wd"]), digits = 3)
            ae_cor_test = round(my_cor(vec(test_x), vec(model.ae.net(test_x))), digits= 3)
            cph_loss_test = round(model.cph.lossf(model.cph,test_x, test_x_c, test_y_e, NE_frac_tst, params_dict["cph_wd"]), digits= 3)
            OUTS_tst =  vec(model.cph.cphdnn(vcat(model.encoder(test_x), test_x_c)))
            cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e,OUTS_tst)
            push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr, ae_loss_test, ae_cor_test, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\t TEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))\t cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, learning_curve, params_dict, iter, fold)

        end
        if learning_curve[end][8] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][8]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ### AUTO-ENCODER 
        ## test set correlations
        push!(test_x_pred_by_fold, cpu(vec(model.ae.net(test_x))))
        push!(test_xs, cpu(vec(test_x)))
        ## train set correlations
        push!(train_x_pred_by_fold, cpu(vec(model.ae.net(train_x))))
        push!(train_xs, cpu(vec(train_x)))
        #### CPH 
        ### TRAIN 
        push!(train_scores_by_fold, cpu(vec(model.cph.cphdnn(vcat(model.encoder(train_x), train_x_c)))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ### TEST 
        push!(test_scores_by_fold, cpu(vec(model.cph.cphdnn(vcat(model.encoder(test_x), test_x_c)))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))
        
    end
    #### AUTO-ENCODER
    x_train= vcat(train_xs...)
    x_test = vcat(test_xs...)
    ae_outs_train = vcat(train_x_pred_by_fold...)
    ae_outs_test = vcat(test_x_pred_by_fold...)
    #### CPH 
    ### TESTS
    cph_outs_test = vcat(test_scores_by_fold...)
    yt_test = vcat(test_y_t_by_fold...)
    ye_test = vcat(test_y_e_by_fold...)
    ### TRAIN
    cph_outs_train = vcat(train_scores_by_fold...)
    yt_train = vcat(train_y_t_by_fold...)
    ye_train = vcat(train_y_e_by_fold...)
    
    params_dict["aecphdnn_tst_c_ind"] = concordance_index(yt_test, ye_test, -1 .* cph_outs_test)[1]
    params_dict["aecphdnn_train_c_ind"] = concordance_index(yt_train, ye_train, -1 .* cph_outs_train)[1]
    params_dict["aecphdnn_tst_corr"] = my_cor(ae_outs_test, x_test)
    params_dict["aecphdnn_train_corr"] = my_cor(ae_outs_train, x_train)
    
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    return Dict(
    ## best model for inspection
    "best_model"=>best_model, "best_model_fold" => best_model_fold, 
    ### CPH train set and test sets 
    "cph_outs_test" => cph_outs_test, "yt_test" => yt_test, "ye_test" => ye_test, 
    "cph_outs_train" => cph_outs_train, "yt_train" => yt_train, "ye_train" => ye_train,
    ### AE train set and test sets 
    "ae_outs_test" => ae_outs_test, "x_test" =>x_test, 
    "ae_outs_train" => ae_outs_train, "x_train" => x_train
    )
end 

function validate_enccphdnn!(params_dict, brca_prediction, dump_cb_brca, clinf;device = gpu)
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:end]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
    #device()
    #device!()
    mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])")
    # init results lists 
    scores_by_fold, yt_by_fold, ye_by_fold = [],[], []
    # create fold directories
    [mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    # splitting, dumped 
    #folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
    # dump_folds(folds, params_dict, brca_prediction.rows)
    # dump params
    bson("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/params.bson",params_dict)

    for fold in folds
        model = build(params_dict)

        ## STATIC VARS    
        batchsize = params_dict["mb_size"]
        nepochs= params_dict["nepochs"]
        wd = params_dict["wd"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        nsamples = size(train_y_t)[2]
        nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            cursor = (iter -1)  % nminibatches + 1
            mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            X_ = gpu(train_x[:,mb_ids])
            X_c_ = gpu(train_x_c[:,mb_ids])
            Y_e_ = gpu(train_y_e[mb_ids])
            ## gradient CPH
            ps = Flux.params(model.encoder, model.cphdnn)
            gs = gradient(ps) do
                
                model.lossf(model,X_, X_c_, Y_e_, NE_frac_tr, params_dict["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS_tr = vec(model.cphdnn(vcat(model.encoder(train_x), train_x_c)))
            cph_loss =model.lossf(model,train_x, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            params_dict["tr_acc"] = round(cind_tr, digits = 3)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x, test_x_c,  test_y_e, NE_frac_tst, params_dict["wd"]), digits= 3)
            OUTS_tst =  vec(model.cphdnn(vcat(model.encoder(test_x), test_x_c)))
            cind_test, cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, OUTS_tst)
            params_dict["tst_acc"] = round(cind_test, digits = 3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\tTEST cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
            dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
        end
        OUTS = cpu(vec(model.cphdnn(vcat(model.encoder(test_x), test_x_c))))
        push!(scores_by_fold, OUTS)
        push!(yt_by_fold, cpu(vec(test_y_t)))
        push!(ye_by_fold, cpu(vec(test_y_e)))
    end 
    #### TESTS
    concat_OUTS = vcat(scores_by_fold...)
    concat_yt = vcat(yt_by_fold...)
    concat_ye = vcat(ye_by_fold...)
    Cis = []
    for i in 1:1000
        sampling = rand(1:length(concat_OUTS),length(concat_OUTS))
        push!(Cis, concordance_index(concat_yt[sampling], concat_ye[sampling], -1 .* concat_OUTS[sampling])[1])
    end
   
    groups = ["low_risk" for i in 1:length(concat_OUTS)]    
    high_risk = concat_OUTS .> median(concat_OUTS)
    low_risk = concat_OUTS .< median(concat_OUTS)
    end_of_study = 365 * 10
    groups[high_risk] .= "high_risk"
    p_high, x_high, sc1_high, sc2_high = surv_curve(concat_yt[high_risk], concat_ye[high_risk]; color = "red")
    p_low, x_low, sc1_low, sc2_low = surv_curve(concat_yt[low_risk], concat_ye[low_risk]; color = "blue")

    lrt_pval = round(log_rank_test(concat_yt, concat_ye, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)
    f = draw(p_high + x_high + p_low + x_low, axis = (;title = "CPHDNN Single Task - Predicted Low (blue) vs High (red) risk\nc-index: $(round(median(Cis), digits = 3))\nlog-rank-test pval: $lrt_pval"))
    CairoMakie.save("$outpath/$(params_dict["modelid"])/low_vs_high_surv_curves.pdf",f)
end 


function validate_cphdnn_clinf!(params_dict, brca_prediction, dump_cb_brca, clinf;device = gpu)
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:6]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold = [],[], [], []
    test_scores_by_fold, train_scores_by_fold = [],[]
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build(params_dict)

        ## STATIC VARS    
        #batchsize = params_dict["mb_size"]
        nepochs= params_dict["nepochs"]
        #wd = params_dict["wd"]
        ordering = sortperm(-fold["Y_t_train"])
        #train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        #test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        nsamples = size(train_y_t)[2]    
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            ## gradient CPH
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                model.lossf(model, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            OUTS_tr = vec(model.model(train_x_c))
            cph_loss =model.lossf(model, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            #params_dict["tr_acc"] = round(cind_tr, digits = 3)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,  test_x_c,  test_y_e, NE_frac_tst, params_dict["wd"]), digits= 3)
            OUTS_tst =  vec(model.model(test_x_c))
            cind_test, cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, OUTS_tst)
            #params_dict["tst_acc"] = round(cind_test, digits = 3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\tTEST cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
        end
        if learning_curve[end][4] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][4]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ## train c-index
        push!(train_scores_by_fold, cpu(vec(model.model( train_x_c))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ## test c-index
        push!(test_scores_by_fold, cpu(vec(model.model( test_x_c))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))

    end 
    #### TESTS
    outs_test = vcat(test_scores_by_fold...)
    test_yt = vcat(test_y_t_by_fold...)
    test_ye = vcat(test_y_e_by_fold...)
    #### TRAIN
    outs_train = vcat(train_scores_by_fold...)
    train_yt = vcat(train_y_t_by_fold...)
    train_ye = vcat(train_y_e_by_fold...)
    
    # test_cis = []
    # for i in 1:1000
    #     sampling = rand(1:length(outs_test),length(outs_test))
    #     push!(test_cis, concordance_index(test_yt[sampling], test_ye[sampling], -1 .* outs_test[sampling])[1])
    # end
    # train_cis = []
    # for i in 1:1000
    #     sampling = rand(1:length(outs_train),length(outs_train))
    #     push!(train_cis, concordance_index(train_yt[sampling], train_ye[sampling], -1 .* outs_train[sampling])[1])
    # end
    
    params_dict["cphdnn_tst_c_ind"] = concordance_index(test_yt, test_ye, -1 .* outs_test)[1]
    params_dict["cphdnn_train_c_ind"] = concordance_index(train_yt, train_ye, -1 .* outs_train)[1]
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    
    return best_model, best_model_fold, outs_test, test_yt, test_ye, outs_train, train_yt, train_ye  
end 



function validate_cphdnn_clinf_noise!(params_dict, brca_prediction, dump_cb_brca, clinf;device = gpu)
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:6]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold = [],[], [], []
    test_scores_by_fold, train_scores_by_fold = [],[]
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build(params_dict)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        nsamples = size(train_y_t)[2]    
        learning_curve = []
        for iter in 1:nepochs#ProgressBar(1:nepochs)
            ## gradient CPH
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                model.lossf(model,train_x, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            OUTS_tr = vec(model.model(vcat(train_x, train_x_c)))
            cph_loss =model.lossf(model,train_x, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            #params_dict["tr_acc"] = round(cind_tr, digits = 3)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x, test_x_c,  test_y_e, NE_frac_tst, params_dict["wd"]), digits= 3)
            OUTS_tst =  vec(model.model(vcat(test_x, test_x_c)))
            cind_test, cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, OUTS_tst)
            #params_dict["tst_acc"] = round(cind_test, digits = 3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\tTEST cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
        end
        if learning_curve[end][4] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][4]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ## train c-index
        push!(train_scores_by_fold, cpu(vec(model.model( vcat(train_x, train_x_c)))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ## test c-index
        push!(test_scores_by_fold, cpu(vec(model.model( vcat(test_x, test_x_c)))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))

    end 
    #### TESTS
    outs_test = vcat(test_scores_by_fold...)
    test_yt = vcat(test_y_t_by_fold...)
    test_ye = vcat(test_y_e_by_fold...)
    #### TRAIN
    outs_train = vcat(train_scores_by_fold...)
    train_yt = vcat(train_y_t_by_fold...)
    train_ye = vcat(train_y_e_by_fold...)
    
    # test_cis = []
    # for i in 1:1000
    #     sampling = rand(1:length(outs_test),length(outs_test))
    #     push!(test_cis, concordance_index(test_yt[sampling], test_ye[sampling], -1 .* outs_test[sampling])[1])
    # end
    # train_cis = []
    # for i in 1:1000
    #     sampling = rand(1:length(outs_train),length(outs_train))
    #     push!(train_cis, concordance_index(train_yt[sampling], train_ye[sampling], -1 .* outs_train[sampling])[1])
    # end
    
    params_dict["cphdnn_tst_c_ind"] = concordance_index(test_yt, test_ye, -1 .* outs_test)[1]
    params_dict["cphdnn_train_c_ind"] = concordance_index(train_yt, train_ye, -1 .* outs_train)[1]
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    
    return best_model, best_model_fold, outs_test, test_yt, test_ye, outs_train, train_yt, train_ye  
end 


function validate_cphclinf_dev!(params_dict, brca_prediction, dump_cb_brca, clinf;device = gpu)
    folds = split_train_test(Matrix(brca_prediction.data), Matrix(clinf[:,2:6]), brca_prediction.survt, brca_prediction.surve, brca_prediction.samples;nfolds =5)
    #device()
    #device!()
    mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])")
    # init results lists 
    scores_by_fold, yt_by_fold, ye_by_fold = [],[], []
    # create fold directories
    [mkdir("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    # splitting, dumped 
    #folds = split_train_test(Data.data, label_binarizer(Data.targets), nfolds = params["nfolds"])
    # dump_folds(folds, params_dict, brca_prediction.rows)
    # dump params
    bson("RES/$(params_dict["session_id"])/$(params_dict["modelid"])/params.bson",params_dict)

    for fold in folds
        model = build(params_dict)

        ## STATIC VARS    
        #batchsize = params_dict["mb_size"]
        nepochs= params_dict["nepochs"]
        #wd = params_dict["wd"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,:]'));
            
        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,:]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
         
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0

        nsamples = size(train_y_t)[2]
        #nminibatches = Int(floor(nsamples/ batchsize))
            
        learning_curve = []

        for iter in 1:nepochs#ProgressBar(1:nepochs)
            #cursor = (iter -1)  % nminibatches + 1
            #mb_ids = collect((cursor -1) * batchsize + 1: min(cursor * batchsize, nsamples))
            #X_ = gpu(train_x[:,mb_ids])
            #X_c_ = gpu(train_x_c[:,mb_ids])
            ## gradient CPH
            ps = Flux.params(model.model)
            gs = gradient(ps) do
                
                model.lossf(model,train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            end
            Flux.update!(model.opt, ps, gs)
            
            OUTS_tr = vec(model.model(train_x_c))
            cph_loss =model.lossf(model, train_x_c, train_y_e, NE_frac_tr, params_dict["wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            params_dict["tr_acc"] = round(cind_tr, digits = 3)
            #push!(learning_curve, (ae_loss, ae_cor, cph_loss, cind_tr))
            # save model (bson) every epoch if specified 
            cph_loss_test = round(model.lossf(model,test_x_c,  test_y_e, NE_frac_tst, params_dict["wd"]), digits= 3)
            OUTS_tst =  vec(model.model(test_x_c))
            cind_test, cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e, OUTS_tst)
            params_dict["tst_acc"] = round(cind_test, digits = 3)
            push!(learning_curve, ( cph_loss, cind_tr, cph_loss_test, cind_test))
            println("FOLD $(fold["foldn"]) - $iter\t TRAIN cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\tTEST cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]")
            #dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
            dump_cb_brca(model, EncCPHDNN_LearningCurve(learning_curve), params_dict, iter, fold)
        end
        OUTS = cpu(vec(model.model(test_x_c)))
        push!(scores_by_fold, OUTS)
        push!(yt_by_fold, cpu(vec(test_y_t)))
        push!(ye_by_fold, cpu(vec(test_y_e)))
    end 
    #### TESTS
    concat_OUTS = vcat(scores_by_fold...)
    concat_yt = vcat(yt_by_fold...)
    concat_ye = vcat(ye_by_fold...)
    Cis = []
    for i in 1:1000
        sampling = rand(1:length(concat_OUTS),length(concat_OUTS))
        push!(Cis, concordance_index(concat_yt[sampling], concat_ye[sampling], -1 .* concat_OUTS[sampling])[1])
    end
   
    groups = ["low_risk" for i in 1:length(concat_OUTS)]    
    high_risk = concat_OUTS .> median(concat_OUTS)
    low_risk = concat_OUTS .< median(concat_OUTS)
    end_of_study = 365 * 10
    groups[high_risk] .= "high_risk"
    p_high, x_high, sc1_high, sc2_high = surv_curve(concat_yt[high_risk], concat_ye[high_risk]; color = "red")
    p_low, x_low, sc1_low, sc2_low = surv_curve(concat_yt[low_risk], concat_ye[low_risk]; color = "blue")

    lrt_pval = round(log_rank_test(concat_yt, concat_ye, groups, ["low_risk", "high_risk"]; end_of_study = end_of_study); digits = 5)
    f = draw(p_high + x_high + p_low + x_low, axis = (;title = "CPHDNN Single Task - Predicted Low (blue) vs High (red) risk\nc-index: $(round(median(Cis), digits = 3))\nlog-rank-test pval: $lrt_pval"))
    CairoMakie.save("$outpath/$(params_dict["modelid"])/low_vs_high_surv_curves.pdf",f)
    c_tst = round(median(Cis), digits = 3)
    return c_tst
end 



function validate_aecphdnn_dev!(params_dict, dataset, dump_cb, clinf;device=gpu)
    folds = split_train_test(Matrix(dataset[:tpm_data]), Matrix(clinf), dataset[:survt], dataset[:surve], dataset[:samples];nfolds =5)
    model_params_path = "$(params_dict["session_id"])/$(params_dict["model_type"])_$(params_dict["modelid"])"
    mkdir("RES/$model_params_path")
    test_y_t_by_fold, test_y_e_by_fold, train_y_t_by_fold, train_y_e_by_fold = [],[], [], []
    test_scores_by_fold, train_scores_by_fold = [],[]
    train_x_pred_by_fold, test_x_pred_by_fold, test_xs, train_xs = [],[], [], []
    [mkdir("RES/$model_params_path/FOLD$(zpad(foldn,pad =3))") for foldn in 1:params_dict["nfolds"]]
    bson("RES/$model_params_path/params.bson",params_dict)
    best_model = nothing
    best_accuracy = 0
    best_model_fold = nothing 
    for fold in folds
        model = build_ae_cph_dnn(params_dict)
        ## STATIC VARS    
        nepochs= params_dict["nepochs"]
        ordering = sortperm(-fold["Y_t_train"])
        train_x = device(Matrix(fold["train_x"][ordering,:]'));
        train_x_c = device(Matrix(fold["train_x_c"][ordering,1:8]'));

        train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
        train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
        NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

        ordering = sortperm(-fold["Y_t_test"])
        test_x = device(Matrix(fold["test_x"][ordering,:]'));
        test_x_c = device(Matrix(fold["test_x_c"][ordering,1:8]'));

        test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
        test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
        NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0
        
        train_y =gpu(Matrix(fold["train_x_c"][:,9:end]'))
        test_y = gpu(Matrix(fold["test_x_c"][:,9:end]'))
        
        #nsamples = size(train_y_t)[2]
             
        learning_curve = []
        for iter in 1:nepochs
            ## gradient CPH            
            ps1 = Flux.params(model["cph"].model, model["enc"])
            gs1 = gradient(ps1) do
                model["cph"].lossf(model["cph"],model["enc"], train_x,  train_x_c, train_y_e, NE_frac_tr, params_dict["cph_wd"])
            end

            ## gradient Auto-Encoder 
            ps2 = Flux.params(model["ae"].net)
            gs2 = gradient(ps2) do
                model["ae"].lossf(model["ae"], train_x, train_x, weight_decay = params_dict["ae_wd"])
            end
            
            ## gradient Classfier DNN
            ps3 = Flux.params(model["dnn"].model)
            gs3 = gradient(ps3) do 
                model["dnn"].lossf(model["dnn"], train_x, train_y, weight_decay = params_dict["clfdnn_wd"])
            end

            Flux.update!(model["cph"].opt, ps1, gs1)
            Flux.update!(model["ae"].opt, ps2, gs2)
            #Flux.update!(model["dnn"].opt, ps3, gs3)

            OUTS_tr = vec(model["cph"].model(vcat(model["enc"](train_x), train_x_c)))
            ae_loss = model["ae"].lossf(model["ae"], train_x, train_x, weight_decay = params_dict["ae_wd"])
            ae_cor =  round(my_cor(vec(train_x), vec(model["ae"].net(train_x))),digits = 3)
            cph_loss = model["cph"].lossf(model["cph"],model["enc"](train_x),  train_x_c, train_y_e, NE_frac_tr, params_dict["cph_wd"])
            cind_tr, cdnt_tr, ddnt_tr, tied_tr  = concordance_index(train_y_t, train_y_e, OUTS_tr)
            ae_loss_test = round(model["ae"].lossf(model["ae"], test_x, test_x, weight_decay = params_dict["ae_wd"]), digits = 3)
            ae_cor_test = round(my_cor(vec(test_x), vec(model["ae"].net(test_x))), digits= 3)
            cph_loss_test = round(model["cph"].lossf(model["cph"],model["enc"](test_x),  test_x_c, test_y_e, NE_frac_tst, params_dict["cph_wd"]), digits= 3)
            OUTS_tst =  vec(model["cph"].model(vcat(model["enc"](test_x), test_x_c)))
            cind_test,cdnt_tst, ddnt_tst, tied_tst = concordance_index(test_y_t, test_y_e,OUTS_tst)
            
            train_clf_loss = round(model["dnn"].lossf(model["dnn"], train_x, train_y, weight_decay = params_dict["clfdnn_wd"]),digits=3)
            train_clf_acc =  round(accuracy(Matrix{Int}(train_y), cpu(model["dnn"].model(train_x))),digits = 3)
            
            test_clf_loss = round(model["dnn"].lossf(model["dnn"], test_x, test_y, weight_decay = params_dict["clfdnn_wd"]),digits=3)
            test_clf_acc =  round(accuracy(Matrix{Int}(test_y), cpu(model["dnn"].model(test_x))),digits = 3)
            
            push!(learning_curve, (ae_loss, ae_cor, ae_loss_test, ae_cor_test,  cph_loss, cind_tr, cph_loss_test, cind_test, train_clf_loss, train_clf_acc, test_clf_loss, test_clf_acc))
            println("FOLD $(fold["foldn"]) $iter\t TRAIN AE-loss $(round(ae_loss,digits =3)) \t AE-cor: $(round(ae_cor, digits = 3))\t cph-loss-avg: $(round(cph_loss / params_dict["nsamples_train"],digits =6)) \t cph-cind: $(round(cind_tr,digits =3))\t CLF loss $(round(train_clf_loss,digits =3)) \t acc : $(round(train_clf_acc, digits = 3))")
            println("\t\tTEST AE-loss $(round(ae_loss_test,digits =3)) \t AE-cor: $(round(ae_cor_test, digits = 3))\t cph-loss-avg: $(round(cph_loss_test / params_dict["nsamples_test"],digits =6)) \t cph-cind: $(round(cind_test,digits =3)) [$(Int(cdnt_tst)), $(Int(ddnt_tst)), $(Int(tied_tst))]  CLF loss $(round(test_clf_loss,digits =3)) \t acc : $(round(test_clf_acc, digits = 3))")
            dump_cb(model, learning_curve, params_dict, iter, fold)

        end
        if learning_curve[end][8] > best_accuracy
            best_model = model
            best_accuracy = learning_curve[end][8]
            best_model_fold = fold
            params_dict["bm_tst_c_ind"] = round(best_accuracy, digits =3 )
        end 
        ### AUTO-ENCODER 
        ## test set correlations
        push!(test_x_pred_by_fold, cpu(vec(model["ae"].net(test_x))))
        push!(test_xs, cpu(vec(test_x)))
        ## train set correlations
        push!(train_x_pred_by_fold, cpu(vec(model["ae"].net(train_x))))
        push!(train_xs, cpu(vec(train_x)))
        #### CPH 
        ### TRAIN 
        push!(train_scores_by_fold, cpu(vec(model["cph"].model(vcat(model["enc"](train_x), train_x_c)))))
        push!(train_y_t_by_fold, cpu(vec(train_y_t)))
        push!(train_y_e_by_fold, cpu(vec(train_y_e)))
        ### TEST 
        push!(test_scores_by_fold, cpu(vec(model["cph"].model(vcat(model["enc"](test_x), test_x_c)))))
        push!(test_y_t_by_fold, cpu(vec(test_y_t)))
        push!(test_y_e_by_fold, cpu(vec(test_y_e)))
        
    end
    #### AUTO-ENCODER
    x_train= vcat(train_xs...)
    x_test = vcat(test_xs...)
    ae_outs_train = vcat(train_x_pred_by_fold...)
    ae_outs_test = vcat(test_x_pred_by_fold...)
    #### CPH 
    ### TESTS
    cph_outs_test = vcat(test_scores_by_fold...)
    yt_test = vcat(test_y_t_by_fold...)
    ye_test = vcat(test_y_e_by_fold...)
    ### TRAIN
    cph_outs_train = vcat(train_scores_by_fold...)
    yt_train = vcat(train_y_t_by_fold...)
    ye_train = vcat(train_y_e_by_fold...)
    
    params_dict["cph_tst_c_ind"] = concordance_index(yt_test, ye_test, -1 .* cph_outs_test)[1]
    params_dict["cph_train_c_ind"] = concordance_index(yt_train, ye_train, -1 .* cph_outs_train)[1]
    params_dict["ae_tst_corr"] = my_cor(ae_outs_test, x_test)
    params_dict["ae_train_corr"] = my_cor(ae_outs_train, x_train)
    
    params_dict["model_cv_complete"] = true
    bson("RES/$model_params_path/params.bson",params_dict)
    # return Dict(
    # ## best model for inspection
    # "best_model"=>best_model, "best_model_fold" => best_model_fold, 
    # ### CPH train set and test sets 
    # "cph_outs_test" => cph_outs_test, "yt_test" => yt_test, "ye_test" => ye_test, 
    # "cph_outs_train" => cph_outs_train, "yt_train" => yt_train, "ye_train" => ye_train,
    # ### AE train set and test sets 
    # "ae_outs_test" => ae_outs_test, "x_test" =>x_test, 
    # "ae_outs_train" => ae_outs_train, "x_train" => x_train
    # )
end 

function split_train_test(X::Matrix, Y_t::Vector,Y_e::Vector, case_ids::Vector; nfolds = 10)
    folds = Array{Dict, 1}(undef, nfolds)
    nsamples = size(X)[1]
    fold_size = Int(floor(nsamples / nfolds))
    ids = collect(1:nsamples)
    shuffled_ids = shuffle(ids)
    for i in 1:nfolds
        tst_ids = shuffled_ids[collect((i-1) * fold_size +1: min(nsamples, i * fold_size))]
        tr_ids = setdiff(ids, tst_ids)
        X_train = X[tr_ids,:]
        Y_t_train = Y_t[tr_ids]
        Y_e_train = Y_e[tr_ids]
        X_test = X[tst_ids,:]
        Y_t_test = Y_t[tst_ids]
        Y_e_test = Y_e[tst_ids]

        folds[i] = Dict("foldn"=> i, "train_ids"=>tr_ids, "test_ids"=>tst_ids,
                        "train_case_ids"=>case_ids[tr_ids], "train_x"=>X_train,"Y_t_train"=>Y_t_train, "Y_e_train"=>Y_e_train,
                        "tst_case_ids"=>case_ids[tst_ids], "test_x"=>X_test, "Y_t_test"=>Y_t_test, "Y_e_test"=>Y_e_test)
    end
    return folds 
end 
function format_train_test(fold; device = gpu)

    ordering = sortperm(-fold["Y_t_train"])
    train_x = device(Matrix(fold["train_x"][ordering,:]'));
    train_y_t = device(Matrix(fold["Y_t_train"][ordering,:]'));
    train_y_e = device(Matrix(fold["Y_e_train"][ordering,:]'));
    NE_frac_tr = sum(train_y_e .== 1) != 0 ? 1 / sum(train_y_e .== 1) : 0

    ordering = sortperm(-fold["Y_t_test"])
    test_x = device(Matrix(fold["test_x"][ordering,:]'));
    test_y_t = device(Matrix(fold["Y_t_test"][ordering,:]'));
    test_y_e = device(Matrix(fold["Y_e_test"][ordering,:]'));
    NE_frac_tst = sum(test_y_e .== 1) != 0 ? 1 / sum(test_y_e .== 1) : 0
    return train_x, train_y_t, train_y_e, NE_frac_tr, test_x, test_y_t, test_y_e, NE_frac_tst
end 
function concordance_index(T, E, S)
    concordant_pairs = S .< S'
    admissable_pairs = T .< T'
    discordant_pairs = S .> S'
    tied_pairs = sum(S .== S') - length(T)
    concordant = sum(E .* (admissable_pairs .* concordant_pairs))
    discordant = sum(E .* (admissable_pairs .* discordant_pairs) )
    C_index = concordant / (concordant + discordant)  
    return C_index, concordant, discordant, tied_pairs
end


function bootstrap_c_ind(OUTS_TST, Y_T_TST, Y_E_TST)
    # merge results, obtain risk scores, test data.
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
    return med_c_ind, lo_ci, up_ci, sorted_cinds
end 
