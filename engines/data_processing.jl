struct MLDataset
    data::Matrix
    samples::Array 
    genes::Array
    biotypes::Array
    labels::Array
end 

struct MLSurvDataset
    data::Matrix
    samples::Array 
    genes::Array
    biotypes::Array
    labels::Array
    survt::Array
    surve::Array
end 

function MLSurvDataset(infilename)
    infile = h5open(infilename, "r")
    data = infile["data"][:,:] 
    samples = infile["samples"][:]  
    labels = infile["labels"][:]  
    genes = infile["genes"][:]  
    biotypes = infile["biotypes"][:]
    survt = infile["survt"][:]  
    surve = infile["surve"][:]  
    close(infile)
    return MLSurvDataset(data, samples, genes, biotypes, labels, survt, surve)
end 

function MLDataset(infilename)
    infile = h5open(infilename, "r")
    data = infile["data"][:,:] 
    samples = infile["samples"][:]  
    labels = infile["labels"][:]  
    genes = infile["genes"][:]  
    biotypes = infile["biotypes"][:]
    close(infile)
    return MLDataset(data, samples, genes, biotypes, labels)
end 


function load_datasets()
    ## load GE data (TPM norm)
    ## samples in USI 
    ## genes in symbols
    ## biotypes (protein coding, non-coding)
    ## labels (molecular classification)
    return Dict("TCGA"=>MLDataset("Data/TCGA_BRCA_tpm_n1050_btypes_labels.h5"),
    "TALL"=>MLDataset("Data/TARGET_ALL_tpm_n264_btypes_labels.h5"),
    "BRCA"=>MLDataset("Data/TCGA_BRCA_tpm_n1050_btypes_labels.h5"),
    "LAML"=>load_laml())
end 
function create_fetch_data_file(;prefix = "TARGET_ALL")
    baseurl = "https://api.gdc.cancer.gov/data"
    basepath = "Data/GDC_raw"
    outputfile = "$basepath/$(prefix)_fetch_data.sh"
    FILES = "$basepath/$(prefix)_files.json"
    MANIFEST = CSV.read("$basepath/$(prefix)_manifest.txt", DataFrame)
    J = JSON.parsefile(FILES)
    f = open(outputfile, "w")
    for file_id::Int in 1:length(J)
        case_id = J[file_id]["cases"][1]["case_id"]
        filename = J[file_id]["file_name"]
        
        UUID = MANIFEST[MANIFEST[:,"filename"] .== filename,"id"][1]
        cmd = "curl --remote-header-name $baseurl/$UUID -o $basepath/$(prefix)/$case_id\n"
        write(f, cmd)
    end
    close(f)
    return outputfile
end 
function merge_GDC_data(basepath)
    files = readdir(basepath)
    sample_data = CSV.read("$basepath/$(files[1])", DataFrame, delim = "\t", header = 2)
    biotypes = sample_data[5:end,"gene_type"]
    sample_data = sample_data[5:end, ["gene_name", "stranded_second"]]
    genes = sample_data.gene_name
    nsamples = length(files)
    ngenes = size(sample_data)[1]
    m=Array{Float32, 2}(undef, (nsamples, ngenes))
    
    for fid::Int in 1:length(files)
        file = files[fid]
        dat = CSV.read("$basepath/$(file)", DataFrame, delim = "\t", header = 2)
        dat = dat[5:end, ["gene_name", "stranded_second"]]
        m[fid, :] = dat.stranded_second
    end
    return m, files, genes, biotypes
end 

function process_tall(outfilename = "Data/TARGET_ALL_tpm_n264_btypes_labels.h5")
    m, samples, genes, biotypes = merge_GDC_data("Data/GDC_raw/TARGET_ALL")
    basepath = "Data/GDC_raw"
    FILES = "$basepath/TARGET_ALL_clinical.json"
    J = JSON.parsefile(FILES)
    submitids, case_ids = [], []
    for F in J
        case_id = F["case_id"]
        # surv_t 
        # surv_e 
        # primary diagnosis 
        subtype = F["diagnoses"][1]["primary_diagnosis"] 
        ethnicity = F["demographic"]["race"]
        crea_dtime = F["demographic"]["created_datetime"]
        upd_dtime =  F["demographic"]["updated_datetime"]
        submitter_ID = split(F["demographic"]["submitter_id"],"-")[3][1:6]
        push!(case_ids, case_id)
        push!(submitids, submitter_ID)
        elapsed_time = Day(DateTime(split(upd_dtime, "T")[1], "yyyy-mm-dd") - DateTime(split(crea_dtime, "T")[1], "yyyy-mm-dd")).value
        surv_status = F["demographic"]["vital_status"] == "Dead" ? 1 : 0
        surv_t = surv_status == 1 ? F["demographic"]["days_to_death"] : elapsed_time
        #println("$(case_id) $ethnicity $submitter_ID $surv_t $surv_status")
    end 
    CLIN_df = DataFrame(:USI=>submitids, :case_id=>case_ids)

    fpath = "Data/GDC_raw/TARGET_phase2_SampleID_tSNE-perplexity20"
    ALL_subtypes = XLSX.readxlsx("$fpath.xlsx")["Sheet1"][:]
    ALL_df = DataFrame(:X1=>ALL_subtypes[2:end,2],:X2=>ALL_subtypes[2:end,3],:USI =>  ALL_subtypes[2:end,6], :sampleID =>   ALL_subtypes[2:end,7], 
        :subtype =>   ALL_subtypes[2:end,8], :ETP_classification => ALL_subtypes[2:end,4])
    # join ALL_df to USI_rnaseq_filename
    samples_df = DataFrame(:case_id=>samples, :II=>collect(1:length(samples)))
    samples_df = sort(innerjoin(samples_df, CLIN_df, on = :case_id), :II)
    FULL_CLIN_DF = innerjoin(ALL_df, samples_df, on = :USI)
    labels = FULL_CLIN_DF.subtype # already sorted
    outfile = h5open(outfilename, "w")
    outfile["data"] = log10.(m[FULL_CLIN_DF.II,:] .+ 1) 
    outfile["samples"] = Array{String}(samples[FULL_CLIN_DF.II]) 
    outfile["labels"] = Array{String}(labels) 
    outfile["genes"] = Array{String}(genes) 
    outfile["biotypes"] = Array{String}(biotypes)
    close(outfile)
end 


function process_brca(outfilename = "Data/TCGA_BRCA_tpm_n1050_btypes_labels.h5")
    m, samples, genes, biotypes = merge_GDC_data("Data/GDC_raw/TCGA_BRCA")
    basepath = "Data/GDC_raw"
    FILES = "$basepath/TCGA_BRCA_clinical.json"
    J = JSON.parsefile(FILES)
    submitids, case_ids, surves, survts = [], [],[],[]
    for (i,F) in enumerate(J)
        if "demographic" in keys(F) 
        case_id = F["case_id"]
        submitter_ID = split(F["demographic"]["submitter_id"],"_")[1]
        surve = F["demographic"]["vital_status"] == "Dead" ? 1 : 0 # status 
        if surve == 1 && "days_to_death" in keys(F["demographic"])
            survt = Int(F["demographic"]["days_to_death"]) 
            push!(submitids, submitter_ID)
            push!(case_ids, case_id)
            push!(surves, surve)
            push!(survts, survt)
            #println("$i $(case_id) $submitter_ID $surve $survt")
        elseif surve == 0
            survt = Int(F["diagnoses"][1]["days_to_last_follow_up"])
            push!(submitids, submitter_ID)
            push!(case_ids, case_id)
            push!(surves, surve)
            push!(survts, survt)
            #println("$i $(case_id) $submitter_ID $surve $survt")
        end 
        end 
    end 
    CLIN_df = DataFrame(:case_id=>case_ids, :submitid=>submitids,:survt=>Array{Int}(survts),:surve=>Array{Int}(surves))
    
    subtypes = CSV.read("Data/GDC_raw/TCGA_BRCA_survival_pam50.csv", DataFrame)
    DF = DataFrame(:USI=> [tag[1:end-4] for tag in subtypes[:,"samples"]],:subtype => subtypes[:,"clinical_data_PAM50MRNA"])
    # join DF to USI_rnaseq_filename
    samples_df = DataFrame(:case_id=>samples, :II=>collect(1:length(samples)))

    samples_df = sort(innerjoin(samples_df, CLIN_df, on = :case_id), :II)

    FULL_CLIN_DF = innerjoin(DF, samples_df, on = :USI)
    outfile = h5open(outfilename, "w")
    outfile["data"] = log10.(m[FULL_CLIN_DF.II,:] .+ 1) 
    outfile["samples"] = Array{String}(samples[FULL_CLIN_DF.II]) 
    outfile["labels"] = Array{String}(FULL_CLIN_DF.subtype ) 
    outfile["genes"] = Array{String}(genes) 
    outfile["biotypes"] = Array{String}(biotypes)
    outfile["survt"] = FULL_CLIN_DF.survt
    outfile["surve"] = FULL_CLIN_DF.surve
    close(outfile)
end 

function merge_LGN_data(sampleIDs)
    nsamples = size(sampleIDs)[1]
    sample_id = "01H001"
    sample_data = CSV.read("/u/leucegene/data/sample/$sample_id/transcriptome/rsem/unstranded/$(sample_id).genes.results.annotated", DataFrame, delim = "\t")
    sample_data = sample_data[:,["Gene","Category","TPM"]]
    genes = sample_data[:,"Gene"]
    biotypes = sample_data[:,"Category"]
    ngenes = size(sample_data)[1]
    m = Array{Float32, 2}(undef, (nsamples, ngenes))
    for fid in 1:nsamples
        sample_id = sampleIDs[fid]
        dat = CSV.read("/u/leucegene/data/sample/$sample_id/transcriptome/rsem/unstranded/$(sample_id).genes.results.annotated", DataFrame, delim = "\t")
        dat = dat[:,["Gene","Category","TPM"]]
        m[fid,:] = dat.TPM
    end 
    return m, sampleIDs, genes, biotypes
end

function process_lgn()
    lgn_pronostic_CF = CSV.read("Data/LEUCEGENE/lgn_pronostic_CF", DataFrame)
    sampleIDs = Array{String}(lgn_pronostic_CF[:,"sampleID"])
    m, sampleIDs, genes, biotypes = merge_LGN_data(sampleIDs)
    outf = h5open("Data/LGN_AML_tpm_n300_btypes_labels.h5","w")
    outf["data"] = m 
    outf["samples"] = sampleIDs
    outf["genes"] = genes 
    outf["biotypes"] = Array{String}(biotypes)
    outf["labels"] = Array{String}(lgn_pronostic_CF[:, "WHO classification"])
    close(outf)
end 

function process_TCGA(outfilename = "Data/TCGA_n10384_btypes_labels.h5")

    #outputfile = create_fetch_data_file(prefix = "TCGA")
    m, samples, genes, biotypes = merge_GDC_data("Data/GDC_raw/TCGA")
    basepath = "Data/GDC_raw"
    FILES = "$basepath/TCGA_clinical.json"
    J = JSON.parsefile(FILES)
    submitids, case_ids = [], []
    for (i,F) in enumerate(J)
        if "demographic" in keys(F)
        case_id = F["case_id"]
        # surv_t 
        # surv_e 
        # primary diagnosis 
        #subtype = F["diagnoses"][1]["primary_diagnosis"] 
        #ethnicity = F["demographic"]["race"]
        #crea_dtime = F["demographic"]["created_datetime"]
        #upd_dtime =  F["demographic"]["updated_datetime"]
        submitter_ID = split(F["demographic"]["submitter_id"],"_")[1]
        push!(case_ids, case_id)
        push!(submitids, submitter_ID)
        #elapsed_time = Day(DateTime(split(upd_dtime, "T")[1], "yyyy-mm-dd") - DateTime(split(crea_dtime, "T")[1], "yyyy-mm-dd")).value
        #surv_status = F["demographic"]["vital_status"] == "Dead" ? 1 : 0
        #surv_t = surv_status == 1 ? F["demographic"]["days_to_death"] : elapsed_time
        #println("$i $(case_id) $submitter_ID")
        end 
    end 
    CLIN_df = DataFrame(:USI=>submitids, :case_id=>case_ids)
    FILES = "Data/GDC_raw/TCGA_USI_CASE_ID.json"
    J = JSON.parsefile(FILES)
    USI = []
    Project_ID = []
    for file_id::Int in 1:length(J)
        case_id = J[file_id]["cases"][1]["case_id"]
        project_id = J[file_id]["cases"][1]["project"]["project_id"]
        push!(USI, case_id)
        push!(Project_ID, project_id)
    end
    Projects = DataFrame("case_id" => USI, "project"=> Project_ID);

    # join DF to USI_rnaseq_filename
    samples_df = DataFrame(:case_id=>samples, :II=>collect(1:length(samples)))
    samples_df = sort(innerjoin(samples_df, Projects, on = :case_id), :II)
    outfile = h5open(outfilename, "w")
    outfile["data"] = log10.(m[samples_df.II,:] .+ 1) 
    outfile["samples"] = Array{String}(samples[samples_df.II]) 
    outfile["labels"] = Array{String}(samples_df.project) 
    outfile["genes"] = Array{String}(genes) 
    outfile["biotypes"] = Array{String}(biotypes)
    close(outfile)
end