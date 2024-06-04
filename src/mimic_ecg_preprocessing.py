import os
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def multihot_encode(x, num_classes):
    res = np.zeros(num_classes,dtype=np.float32)
    for y in x:
        res[y]=1
    return res
############################################################################################################


def prepare_consistency_mapping(codes_unique, codes_unique_all, propagate_all=False):
    res={}
    for c in codes_unique:
        if(propagate_all):
            res[c]=[c[:i] for i in range(3,len(c)+1)]
        else:#only propagate if categories are already present
            res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
    return res



def prepare_mimic_ecg(finetune_dataset, target_folder, df_mapped=None, df_diags=None):

    '''finetune_dataset e.g. mimic_all_all_all_all_2000_5A 
    -mimic_{subsettrain}_{labelsettrain}_{subsettest}_{labelsettest}_{mincnt}_{digits} where _{digits} is optional
    -subsettrain: all/ed/hosp/allnonzero/ednonzero/hospnonzero/allnonzerofirst/ednonzerofirst/hospnonzerofirst/allfirst/edfirst/hospfirst default: allnonzero
    -labelsettrain: {all/hosp/ed}{/af/I} first part selects the label set all: both ed diagnosis and hosp diagnosis hosp: just hosp diagnosis ed: just ed diagnosis; second part: can be omitted or af for af labels or collection of uppercase letters such as I to select specific label sets
    -similar for subsettest/labelsettest but labelsettest can only be {all/hosp/ed}
    -digits: 3/4/5/3A/4A/5A or just empty corresponding to I48, I48.1 or I48.19; append an A to include all ancestors'''

    def flatten(l):
        return [item for sublist in l for item in sublist]  
    
    
    df_diags_initial = df_diags  # keep track what to return based on preprocessing level

    subsettrain = finetune_dataset.split("_")[1]
    labelsettrain = finetune_dataset.split("_")[2]
    subsettest = finetune_dataset.split("_")[3]
    labelsettest = finetune_dataset.split("_")[4]
    min_cnt = int(finetune_dataset.split("_")[5])
    if(len(finetune_dataset.split("_"))<7):
        digits = None
        propagate_all = False
    else:
        digits = finetune_dataset.split("_")[6]
        if(digits[-1]=="A"):
            propagate_all = True
            digits= int(digits[:-1])
        else:
            propagate_all = False
            digits = int(digits)
    
    
    # load label dataframe
    if df_diags is not None:
        df_diags = df_diags
    else:
        df_diags = pd.read_pickle(target_folder/"records_w_diag_icd10.pkl")
        
    
    #select the desired label set (train)
    if(labelsettrain.startswith("hosp")):#just hospital discharge diagnosis
        df_diags["label_train"]=df_diags["all_diag_hosp"]
        labelsettrain=labelsettrain[len("hosp"):]
    elif(labelsettrain.startswith("ed")):#just ED discharge diagnosis
        df_diags["label_train"]=df_diags["ed_diag_ed"]
        labelsettrain=labelsettrain[len("ed"):]
    elif(labelsettrain.startswith("all")):#both ed and hospital discharge diagnosis (the latter if available)
        df_diags["label_train"]=df_diags["all_diag_all"]
        labelsettrain=labelsettrain[len("all"):]
    else:
        assert(False)

    if(labelsettest.startswith("hosp")):#just hospital discharge diagnosis
        df_diags["label_test"]=df_diags["all_diag_hosp"]
    elif(labelsettest.startswith("ed")):#just ED discharge diagnosis
        df_diags["label_test"]=df_diags["ed_diag_ed"]
    elif(labelsettest.startswith("all")):#both ed and hospital discharge diagnosis (the latter if available)
        df_diags["label_test"]=df_diags["all_diag_all"]
    else:
        assert(False)

    df_diags["has_statements_train"]=df_diags["label_train"].apply(lambda x: len(x)>0)#keep track if there were any ICD statements for this sample
    df_diags["has_statements_test"]=df_diags["label_test"].apply(lambda x: len(x)>0)#keep track if there were any ICD statements for this sample
    
    #first truncate to desired number of digits
    if(digits is not None):
        df_diags["label_train"]=df_diags["label_train"].apply(lambda x: list(set([y.strip()[:digits] for y in x])))
        df_diags["label_test"]=df_diags["label_test"].apply(lambda x: list(set([y.strip()[:digits] for y in x])))
    
    #remove trailing placeholder Xs    
    df_diags["label_train"]=df_diags["label_train"].apply(lambda x: list(set([y.rstrip("X") for y in x])))
    df_diags["label_test"]=df_diags["label_test"].apply(lambda x: list(set([y.rstrip("X") for y in x])))
    
    # apply labelset selection af or I for example
    if(labelsettrain=="af"):#special case for atrial fibrillation
        df_diags["label_train"]=df_diags["label_train"].apply(lambda x: [c for c in x if c.startswith("I48")])
    elif(labelsettrain!=""):#special case for specific categories
        df_diags["label_train"]=df_diags["label_train"].apply(lambda x: [c for c in x if c[0] in labelsettrain])
    
    #apply consistency mapping
    col_flattrain = flatten(np.array(df_diags["label_train"]))
    cons_maptrain = prepare_consistency_mapping(np.unique(col_flattrain),np.unique(col_flattrain),propagate_all)
    df_diags["label_train"]=df_diags["label_train"].apply(lambda x: list(set(flatten([cons_maptrain[y] for y in x]))))
    col_flattest = flatten(np.array(df_diags["label_test"]))
    cons_maptest = prepare_consistency_mapping(np.unique(col_flattest),np.unique(col_flattrain),propagate_all)
    df_diags["label_test"]=df_diags["label_test"].apply(lambda x: list(set(flatten([cons_maptest[y] for y in x]))))

    #identify statements/lbl_itos only based on labelsettrain!
    col_flat = flatten(np.array(df_diags["label_train"]))
    codes,counts = np.unique(col_flat,return_counts=True)
    idxs = np.argsort(counts)[::-1]
    codes = codes[idxs]
    counts = counts[idxs]
    codes=codes[np.where(counts>=min_cnt)[0]]

    lbl_itos=codes
    
    if df_diags_initial is not None:
        #select only the selected statements (return made for strat folds in full_preprocessing.py
        df_diags["label_train"]=df_diags["label_train"].apply(lambda x: [v for v in x if v in lbl_itos])
        return df_diags, lbl_itos
    
    
    if(df_mapped is not None):
        print("Label set:",len(lbl_itos),"labels.")#,lbl_itos)
        
        #join the two dataframes
        df_diags = df_diags.set_index("study_id")
        df_diags.drop(["subject_id","ecg_time"],axis=1,inplace=True)
        df_mapped = df_mapped.join(df_diags,on="study_id")
        max_fold = df_mapped.fold.max()
        
        #TRAIN select the desired subset (all/ed/hosp/allnonzero/ednonzero/hospnonzero)
        df_mappedtrain = df_mapped[df_mapped.fold<max_fold-1].copy()#pick the first n-2 folds
        df_mappedtrain["label"]=df_mappedtrain["label_train"]
        if(subsettrain.startswith("all")):#ed and hosp
            df_mappedtrain=df_mappedtrain[df_mappedtrain["ecg_taken_in_ed_or_hosp"]].copy()
        elif(subsettrain.startswith("ed")):#ed only
            df_mappedtrain=df_mappedtrain[df_mappedtrain["ecg_taken_in_ed"]].copy()
        elif(subsettrain.startswit("hosp")):#hosp only
            df_mappedtrain=df_mappedtrain[df_mappedtrain["ecg_taken_in_hosp"]].copy()
        #include samples with zero ICD codes (e.g. not admitted to hospital if predicting hosp diagnosis, or missing outputs from the report)
        
        df_mappedtrain = df_mappedtrain[df_mappedtrain.has_statements_train==True].copy()
        if(subsettrain.endswith("first")):#only select first ecg
            df_mappedtrain = df_mappedtrain[df_mappedtrain.ecg_no_within_stay==0].copy()
        
        #TEST select the desired subset (all/ed/hosp/allnonzero/ednonzero/hospnonzero)
        df_mappedtest = df_mapped[df_mapped.fold>=(max_fold-1)].copy()#pick the final two folds
        df_mappedtest["label"]=df_mappedtest["label_test"]
        if(subsettest.startswith("all")):#ed and hosp
            df_mappedtest=df_mappedtest[df_mappedtest["ecg_taken_in_ed_or_hosp"]].copy()
        elif(subsettest.startswith("ed")):#ed only
            df_mappedtest=df_mappedtest[df_mappedtest["ecg_taken_in_ed"]].copy()
        elif(subsettrain.startswit("hosp")):#hosp only
            df_mappedtest=df_mappedtest[df_mappedtest["ecg_taken_in_hosp"]].copy()
        #include samples with zero ICD codes (e.g. not admitted to hospital if predicting hosp diagnosis, or missing outputs from the report)

        df_mappedtest = df_mappedtest[df_mappedtest.has_statements_test==True].copy()
                
        if(subsettest.endswith("first")):#only select first ecg
            df_mappedtest = df_mappedtest[df_mappedtest.ecg_no_within_stay==0].copy()
        
        #combine train and test
        df_mapped = pd.concat([df_mappedtrain,df_mappedtest])
        #select only the selected statements
        lbl_stoi={s:i for i,s in enumerate(lbl_itos)}
        df_mapped["label"]=df_mapped["label"].apply(lambda x: multihot_encode([lbl_stoi[y] for y in x if y in codes],len(lbl_itos)))
        
    return df_mapped, lbl_itos
