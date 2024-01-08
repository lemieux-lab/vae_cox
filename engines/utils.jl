function create_counter(iterable)
    counter = Dict{Any, Int}()
    for item in iterable
        counter[item] = get(counter, item, 0) + 1
    end
    return counter
end

function generate_labels(Data)
    # translate label 
    cnames = CSV.read("Data/GDC_processed/TCGA_abbrev.txt", DataFrame)
    labels = ["$(cnames[cnames[:,"abbrv"] .== split(l,"-")[2],"def"][1])" for l in TCGA.labels]
    # add number
    counts = create_counter(labels)
    labs = []
    for l in labels 
        nstr = "($(counts[l])) $l"
        newlab = nstr[1:min(length(nstr),30)]
        push!(labs, newlab)  
    end  
    return labs
end 