function SMA(DATA_X, DATA_Y; n=10, k=10)
    means_y = []
    means_x = []
    Y_infos =  DATA_Y[sortperm(DATA_X)]
    X_infos = sort(DATA_X)
    for X_id in vcat(collect(1:Int(floor(length(X_infos) / n )):length(X_infos)), length(X_infos))
        x_id_min = max(X_id - k, 1)
        x_id_max = min(X_id + k, length(DATA_X))
        sma = mean(Y_infos[x_id_min:x_id_max])
        push!(means_y, sma)
        push!(means_x, X_infos[X_id])
    end
    return float.(means_x),float.(means_y)
end 


function draw_scatter_sma!(ax, X, Y, col,label,alpha,ls, SMA_K, SMA_N)
    scatter!(ax, log10.(X), Y, color= col, alpha = alpha)
    sma_x, sma_y = SMA(log10.(X), Y, k=SMA_K,n=SMA_N)
    lines!(ax, sma_x,sma_y, color = col, label = label, linewidth=3, linestyle =ls)
    text!(ax, sma_x .- 0.25,sma_y,text = string.(round.(sma_y, digits=2)), color= col)

end 

function add_multi_scatter!(fig, row, col, data_df; SMA_K=10, SMA_N=10)
    ticks = [minimum(data_df[:,"insize"]),100,1000,maximum(data_df[:,"insize"])]
    dname = unique(data_df[:,"dataset"])[1]
    ax = Axis(fig[row,col],
        xticks = (log10.(ticks),string.(ticks)),
        #yticks = (collect(1:20)/20, ["$x" for x in collect(1:20)/20]),
        title = "Survival prediction in $dname with CPH-DNN & COX-ridge \n by input size with random and directed dimensionality reductions",
        xlabel = "Input size",
        ylabel = "concordance index",
        limits = (nothing, nothing, 0.4, 0.75))
    lines!(ax,log10.(ticks[[1,end]]),[0.5,0.5],linetype = "dashed")

    cphdnn = data_df[data_df[:,"model_type"] .== "cphdnn",:]
    coxridge = data_df[data_df[:,"model_type"] .== "cox_ridge",:]

    draw_scatter_sma!(ax,   cphdnn[cphdnn[:,"nb_clinf"] .!= 0 ,"insize"], 
                            cphdnn[cphdnn[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], 
                            "blue", "CPHDNN with clinical factors (16)", 0.5, :dash, SMA_K, SMA_N)
    draw_scatter_sma!(ax,   coxridge[coxridge[:,"nb_clinf"] .!= 0 ,"insize"], 
                            coxridge[coxridge[:,"nb_clinf"] .!= 0,"cph_test_c_ind"], 
                            "orange", "Cox-ridge with clinical factors (16)", 0.5, :dash,SMA_K, SMA_N)
    draw_scatter_sma!(ax,   cphdnn[cphdnn[:,"nb_clinf"] .== 0 ,"insize"], 
                            cphdnn[cphdnn[:,"nb_clinf"] .== 0,"cph_test_c_ind"], 
                            "blue", "CPHDNN no clin. f", 1, :solid,SMA_K, SMA_N)
    draw_scatter_sma!(ax,   coxridge[coxridge[:,"nb_clinf"] .== 0 ,"insize"], 
                            coxridge[coxridge[:,"nb_clinf"] .== 0,"cph_test_c_ind"], 
                            "orange", "Cox-ridge no clin. f", 1, :solid, SMA_K, SMA_N)
    return fig, ax
end 