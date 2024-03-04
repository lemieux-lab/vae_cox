include("engines/init.jl")
include("engines/data_processing.jl")
include("engines/deep_learning.jl")
include("engines/cross_validation.jl")
include("engines/model_evaluation.jl")
include("engines/figures.jl")
outpath, session_id = set_dirs() ;

res = gather_params("RES/")
fig = Figure(size = (1524, 1524));
for (dname, pos) in zip(["LgnAML", "BRCA", "LGG", "OV"],[(1,1),(1,2), (2,1),(2,2)])
    data_df = res[res[:,"dataset"] .== dname,:]
    data_df = data_df[data_df[:,"insize"] .!= 0,:]
    data_df = data_df[isnan.(data_df[:,"cph_test_c_ind"]) .== 0,:]
    RDM_df = data_df[data_df[:,"dim_redux_type"] .== "RDM",:]    
    PCA_df = data_df[data_df[:,"dim_redux_type"] .== "PCA",:]    
    
    ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
    row, col = pos
    ax = Axis(fig[row,col],
        xticks = (log10.(ticks),string.(ticks)),
        #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
        title = "Survival prediction in $dname with CPH-DNN & COX-ridge \n by input size with random gene selections",
        xlabel = "Input size",
        ylabel = "concordance index")#,
        # limits = (nothing, nothing, 0.4, 0.9))
    lines!(ax,[0,log10(19701)],[0.5,0.5], linestyle = :dash)
    draw_scatter_sma!(ax,   RDM_df[RDM_df[:,"model_type"] .== "cphdnn", "insize"], 
                            RDM_df[RDM_df[:,"model_type"] .== "cphdnn", "cph_test_c_ind"], 
                            "blue", "RDM-CPHDNN", 1, :solid, 10, 10)
    
    draw_scatter_sma!(ax,   RDM_df[RDM_df[:,"model_type"] .== "coxridge", "insize"], 
                            RDM_df[RDM_df[:,"model_type"] .== "coxridge", "cph_test_c_ind"], 
                            "orange", "RDM-Cox-ridge", 1, :solid, 10, 10)
    
    draw_scatter_sma!(ax,   PCA_df[PCA_df[:,"model_type"] .== "cphdnn", "insize"], 
                            PCA_df[PCA_df[:,"model_type"] .== "cphdnn", "cph_test_c_ind"], 
                            "black", "PCA-CPHDNN", 1, :solid, 10, 10)
    
    draw_scatter_sma!(ax,   PCA_df[PCA_df[:,"model_type"] .== "coxridge", "insize"], 
                            PCA_df[PCA_df[:,"model_type"] .== "coxridge", "cph_test_c_ind"], 
                            "grey", "PCA-Cox-ridge", 1, :solid, 10, 10)
        
    axislegend(ax, position = :rb);
end 
fig
CairoMakie.save("figures/lgg_lgnaml_brca_ov_coxridge_cphdnn_rdm_pca.svg",fig)
CairoMakie.save("figures/lgg_lgnaml_brca_ov_coxridge_cphdnn_rdm_pca.png",fig)
CairoMakie.save("figures/lgg_lgnaml_brca_ov_coxridge_cphdnn_rdm_pca.pdf",fig)

fig = Figure(size = (1524, 1524));
for (dname, pos) in zip(["LgnAML", "BRCA", "LGG", "OV"],[(1,1),(1,2), (2,1),(2,2)])
    data_df = res[res[:,"dataset"] .== dname,:]
    data_df = data_df[data_df[:,"insize"] .!= 0,:]
    data_df = data_df[isnan.(data_df[:,"cph_tst_c_ind_med"]) .== 0,:]
    RDM_df = data_df[data_df[:,"dim_redux_type"] .== "RDM",:]    
    PCA_df = data_df[data_df[:,"dim_redux_type"] .== "PCA",:]    
    
    ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
    row, col = pos
    ax = Axis(fig[row,col],
        xticks = (log10.(ticks),string.(ticks)),
        #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
        title = "Survival prediction in $dname with CPH-DNN & COX-ridge \n by input size with random gene selections",
        xlabel = "Input size",
        ylabel = "concordance index")#,
        #limits = (nothing, nothing, 0.4, 0.9))
    lines!(ax,[0,log10(19701)],[0.5,0.5], linestyle = :dash)
    draw_scatter_sma!(ax,   RDM_df[RDM_df[:,"model_type"] .== "cphdnn", "insize"], 
                            RDM_df[RDM_df[:,"model_type"] .== "cphdnn", "cph_tst_c_ind_med"], 
                            "blue", "RDM-CPHDNN", 1, :solid, 10, 7)
    
    draw_scatter_sma!(ax,   RDM_df[RDM_df[:,"model_type"] .== "coxridge", "insize"], 
                            RDM_df[RDM_df[:,"model_type"] .== "coxridge", "cph_tst_c_ind_med"], 
                            "orange", "RDM-Cox-ridge", 1, :solid, 10, 7)
    
    draw_scatter_sma!(ax,   PCA_df[PCA_df[:,"model_type"] .== "cphdnn", "insize"], 
                            PCA_df[PCA_df[:,"model_type"] .== "cphdnn", "cph_tst_c_ind_med"], 
                            "black", "PCA-CPHDNN", 1, :solid, 7, 7)
    
    draw_scatter_sma!(ax,   PCA_df[PCA_df[:,"model_type"] .== "coxridge", "insize"], 
                            PCA_df[PCA_df[:,"model_type"] .== "coxridge", "cph_tst_c_ind_med"], 
                            "grey", "PCA-Cox-ridge", 1, :solid, 7, 7)
        
    axislegend(ax, position = :rb);
end 
fig
CairoMakie.save("figures/lgg_lgnaml_brca_ov_coxridge_cphdnn_rdm_pca_bootstrap.svg",fig)
CairoMakie.save("figures/lgg_lgnaml_brca_ov_coxridge_cphdnn_rdm_pca_bootstrap.png",fig)
CairoMakie.save("figures/lgg_lgnaml_brca_ov_coxridge_cphdnn_rdm_pca_bootstrap.pdf",fig)


PARAMS = gather_params("RES2")
function make_boxplots(PARAMS)
    fig = Figure(size=(600,512));
    offshift = 0.05
    up_y = 0.80
    DATA_df = PARAMS[PARAMS[:,"dataset"] .== "LgnAML",:]
    DATA_df = innerjoin(DATA_df,DataFrame("dim_redux_type"=>unique(DATA_df[:,"dim_redux_type"]), "ID" => collect(1:size(unique(DATA_df[:,"dim_redux_type"]))[1])), on =:dim_redux_type)
    dtype_insize = combine(groupby(DATA_df, ["dim_redux_type"]), :insize=>maximum)
    ticks = ["$dtype\n($ins_max)" for (dtype,ins_max) in zip(dtype_insize[:,1], dtype_insize[:,2])]
    ax1 = Axis(fig[1,1];
                title = "Leucegene",
                limits = (0.5,4.5, nothing, up_y),
                xlabel = "Dimensionality reduction",
                ylabel = "Concordance index",
                xticks = (collect(1:size(ticks)[1]), ticks))
    lines!(ax1,[0,5],[0.5,0.5], linestyle = :dot)
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", "cph_test_c_ind"], color = "blue", label = "CPHDNN")
    boxplot!(ax1, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", "cph_test_c_ind"], color = "orange", label = "Cox-ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", "cph_test_c_ind"]], ["ID", "model_type"]), :cph_test_c_ind=>median) 
    text!(ax1, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median, digits = 3)))
    text!(ax1, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median, digits = 3)))
    axislegend(ax1, position = :rb)
    fig
    DATA_df = PARAMS[PARAMS[:,"dataset"] .== "BRCA",:]
    DATA_df = innerjoin(DATA_df,DataFrame("dim_redux_type"=>unique(DATA_df[:,"dim_redux_type"]), "ID" => collect(1:size(unique(DATA_df[:,"dim_redux_type"]))[1])), on =:dim_redux_type)
    dtype_insize = combine(groupby(DATA_df, ["dim_redux_type"]), :insize=>maximum)
    ticks = ["$dtype\n($ins_max)" for (dtype,ins_max) in zip(dtype_insize[:,1], dtype_insize[:,2])]
    ax2 = Axis(fig[2,1];
                title = "BRCA",
                limits = (0.5,4.5, nothing,up_y),
                xlabel = "Dimensionality reduction",
                ylabel = "Concordance index",
                xticks = (collect(1:size(ticks)[1]), ticks))
    lines!(ax2,[0,5],[0.5,0.5], linestyle = :dot)
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="cphdnn"] .- 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="cphdnn", "cph_test_c_ind"], color = "blue", label = "CPHDNN")
    boxplot!(ax2, DATA_df.ID[DATA_df[:,"model_type"].=="coxridge"] .+ 0.2, width = 0.5,  DATA_df[DATA_df[:,"model_type"].=="coxridge", "cph_test_c_ind"], color = "orange", label = "Cox-Ridge")
    medians = combine(groupby(DATA_df[:,["ID", "model_type", "cph_test_c_ind"]], ["ID", "model_type"]), :cph_test_c_ind=>median) 
    text!(ax2, medians.ID[medians.model_type .== "cphdnn"].-0.35, medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "cphdnn",:].cph_test_c_ind_median, digits = 3)))
    text!(ax2, medians.ID[medians.model_type .== "coxridge"] .+ 0.04, medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median .+ offshift, text= string.(round.(medians[medians.model_type .== "coxridge",:].cph_test_c_ind_median, digits = 3)))
    CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.svg",fig)
    CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.png",fig)
    CairoMakie.save("figures/lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign.pdf",fig)
    return fig
end
make_boxplots(PARAMS)



PARAMS = gather_params("RES")
LC = gather_learning_curves("RES")
LOSSES_DF = innerjoin(LC, PARAMS, on = "modelid");
fig = Figure(size = (1024,512));
TRUNC_DF = LOSSES_DF[(LOSSES_DF.steps .% 100 .== 0) .| (LOSSES_DF.steps .== 1),:]
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = TRUNC_DF[TRUNC_DF[:,"dataset"] .== dataset,:]
    for (col_id, dim_redux_type) in enumerate(unique(DATA_df.dim_redux_type))
        DRD_df = DATA_df[DATA_df[:,"dim_redux_type"] .== dim_redux_type,:]
        DRD_df = DRD_df[DRD_df[:,"model_type"] .== "cphdnn",:]
        ax = Axis(fig[row_id,col_id]; 
            xlabel = "steps", ylabel = "Loss value", 
            title = "$dataset - $dim_redux_type");
        println("processing $dataset - $dim_redux_type ...")
        for modelid in unique(DRD_df[:, "modelid"])
            MOD_df = DRD_df[DRD_df[:,"modelid"] .== modelid, :]
            for foldn in 1:5
                FOLD_data = sort(MOD_df[MOD_df[:,"foldns"] .== foldn,:], "steps")
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train","loss_vals"], color = "blue") 
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test","loss_vals"], color = "orange")
            end    
        end 
    end 
end 
#axislegend(ax1)
CairoMakie.save("figures/cphdnn_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign_overfit.svg",fig)
CairoMakie.save("figures/cphdnn_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign_overfit.png",fig)
CairoMakie.save("figures/cphdnn_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign_overfit.pdf",fig)

fig = Figure(size = (1024,512));
TRUNC_DF = LOSSES_DF[(LOSSES_DF.steps .% 100 .== 0) .| (LOSSES_DF.steps .== 1),:]
for (row_id, dataset) in enumerate(["LgnAML", "BRCA"])
    DATA_df = TRUNC_DF[TRUNC_DF[:,"dataset"] .== dataset,:]
    for (col_id, dim_redux_type) in enumerate(unique(DATA_df.dim_redux_type))
        DRD_df = DATA_df[DATA_df[:,"dim_redux_type"] .== dim_redux_type,:]
        DRD_df = DRD_df[DRD_df[:,"model_type"] .== "coxridge",:]
        ax = Axis(fig[row_id,col_id]; 
            xlabel = "steps", ylabel = "Loss value", 
            title = "$dataset - $dim_redux_type");
        println("processing $dataset - $dim_redux_type ...")
        for modelid in unique(DRD_df[:, "modelid"])
            MOD_df = DRD_df[DRD_df[:,"modelid"] .== modelid, :]
            for foldn in 1:5
                FOLD_data = sort(MOD_df[MOD_df[:,"foldns"] .== foldn,:], "steps")
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "train","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "train","loss_vals"], color = "blue") 
                lines!(ax, FOLD_data[FOLD_data[:,"tst_train"] .== "test","steps"], FOLD_data[FOLD_data[:,"tst_train"] .== "test","loss_vals"], color = "orange")
            end    
        end 
    end 
end 
#axislegend(ax1)
CairoMakie.save("figures/coxridge_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign_overfit.svg",fig)
CairoMakie.save("figures/coxridge_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign_overfit.png",fig)
CairoMakie.save("figures/coxridge_lgnaml_brca_coxridge_cphdnn_rdm_pca_clinf_sign_overfit.pdf",fig)

fig