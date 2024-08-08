#!/usr/bin/env python
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from extract_headers import extract_and_open_files_in_zip

#requires icd-mappings
from icdmappings import Mapper
from ecg_utils import prepare_mimicecg
from timeseries_utils import reformat_as_memmap
from utils.stratify import stratified_subsets
from mimic_ecg_preprocessing import prepare_mimic_ecg


def main():
    parser = argparse.ArgumentParser(description='A script to extract two paths from the command line.')
    
    # Add arguments for the two paths
    parser.add_argument('--mimic-path', help='path to mimic iv folder with subfolders hosp and ed',default="./mimic")
    parser.add_argument('--zip-path', help='path to mimic ecg zip',default="mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip")
    parser.add_argument('--target-path', help='desired output path',default="./")
    
    # you have to explicitly pass this argument to convert to numpy and memmapp
    parset.add_argument('--numpy-memmap', help='convert to numpy and memmap for fast access', action='store_true')
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    # Access the paths
    mimic_path = Path(args.mimic_path)
    zip_file_path = Path(args.zip_path)
    target_path = Path(args.target_path)
    
    
    numpy_memmap = args.numpy_memmap
    
    print("mimic_path",mimic_path)
    print("zip_file_path",zip_file_path)
    print("target_path",target_path)
    ##################################################################################################
    print("Step 1: Extract available records from mimic-ecg-zip-path to create records.pkl")
    if((target_path/"records.pkl").exists()):
        print("Skipping: using existing records.pkl")
        df = pd.read_pickle(target_path/"records.pkl")
    else:
        print("Creating records.pkl")
        if(not Path("records.pkl").exists()):
            df=extract_and_open_files_in_zip(zip_file_path, ".hea")
            df.to_pickle("records.pkl")
    
    #################################################################################################
    print("Step 2: Extract diagnoses for records in raw format to create records_w_diag.pkl")
    if((target_path/"records_w_diag.pkl").exists()):
        print("Skipping: using existing records_w_diag.pkl")
        df_full = pd.read_pickle(target_path/"records_w_diag.pkl")
    else:
        mapper = Mapper()
        
        df_hosp_icd_description = pd.read_csv(mimic_path/"hosp/d_icd_diagnoses.csv.gz")
        df_hosp_icd_diagnoses = pd.read_csv(mimic_path/"hosp/diagnoses_icd.csv.gz")
        df_hosp_admissions = pd.read_csv(mimic_path/"hosp/admissions.csv.gz")
        df_hosp_admissions["admittime"]=pd.to_datetime(df_hosp_admissions["admittime"])
        df_hosp_admissions["dischtime"]=pd.to_datetime(df_hosp_admissions["dischtime"])
        df_hosp_admissions["deathtime"]=pd.to_datetime(df_hosp_admissions["deathtime"])
        
        df_hosp_icd_description["icd10_code"]=df_hosp_icd_description.apply(lambda row:row["icd_code"] if row["icd_version"]==10 else mapper.map(row["icd_code"], source="icd9", target="icd10"),axis=1)#mapper="icd9toicd10"
        icd_mapping = {ic:ic10 for ic,ic10 in zip(df_hosp_icd_description["icd_code"],df_hosp_icd_description["icd10_code"])}
        df_ed_stays = pd.read_csv(mimic_path/"ed/edstays.csv.gz")
        df_ed_stays["intime"]=pd.to_datetime(df_ed_stays["intime"])
        df_ed_stays["outtime"]=pd.to_datetime(df_ed_stays["outtime"])
        df_ed_diagnosis = pd.read_csv(mimic_path/"ed/diagnosis.csv.gz")

        def get_diagnosis_hosp(subject_id, ecg_time):
            df_ecg_during_hosp= df_hosp_admissions[(df_hosp_admissions.subject_id==subject_id) & (df_hosp_admissions.admittime<ecg_time) & ((df_hosp_admissions.dischtime>ecg_time)|(df_hosp_admissions.deathtime>ecg_time))]
            if(len(df_ecg_during_hosp)==0):
                return [],np.nan
            else:
                if(len(df_ecg_during_hosp)>1):
                    print("Error in get_diagnosis_hosp: multiple entries for",subject_id,ecg_time,". Considering only the first one.")
                hadm_id=df_ecg_during_hosp.hadm_id.iloc[0]
                return list(df_hosp_icd_diagnoses[(df_hosp_icd_diagnoses.subject_id==subject_id)&(df_hosp_icd_diagnoses.hadm_id==hadm_id)].sort_values(by=['seq_num']).icd_code), hadm_id #diags_hosp, hadm_id

        def get_diagnosis_ed(subject_id, ecg_time,also_hosp_diag=True):
            df_ecg_during_ed = df_ed_stays[(df_ed_stays.subject_id==subject_id) & (df_ed_stays.intime<ecg_time) & (df_ed_stays.outtime>ecg_time)]
            if(len(df_ecg_during_ed)==0):
                return ([],[],np.nan,np.nan) if also_hosp_diag else ([],np.nan)
            else:
                if(len(df_ecg_during_ed)>1):
                    print("Error in get_diagnosis_ed: multiple entries for",subject_id,ecg_time,". Considering only the first one.")
                stay_id=df_ecg_during_ed.stay_id.iloc[0]
                hadm_id=df_ecg_during_ed.hadm_id.iloc[0]#potentially none
                res=list(df_ed_diagnosis[(df_ed_diagnosis.subject_id==subject_id)&(df_ed_diagnosis.stay_id==stay_id)].sort_values(by=['seq_num']).icd_code)
                if(also_hosp_diag):
                    res2=list(df_hosp_icd_diagnoses[(df_hosp_icd_diagnoses.subject_id==subject_id)&(df_hosp_icd_diagnoses.hadm_id==hadm_id)].sort_values(by=['seq_num']).icd_code)
                    return res, res2, stay_id, (np.nan if hadm_id is None else hadm_id) #diags_ed, diags_hosp, stay_id, hadm_id
                else:
                    return res, stay_id #diags_ed, stay_id


        result=[]

        for id,row in tqdm(df.iterrows(),total=len(df)):
            tmp={}
            tmp["file_name"]=row["file_name"]
            tmp["study_id"]=row["study_id"]
            tmp["subject_id"]=row["subject_id"]
            tmp["ecg_time"]=row["ecg_time"]
            hosp_diag_hosp, hosp_hadm_id =get_diagnosis_hosp(row["subject_id"], row["ecg_time"])
            tmp["hosp_diag_hosp"] = hosp_diag_hosp
            tmp["hosp_hadm_id"] =hosp_hadm_id
            ed_diag_ed,ed_diag_hosp,ed_stay_id,ed_hadm_id = get_diagnosis_ed(row["subject_id"], row["ecg_time"])
            tmp["ed_diag_ed"]=ed_diag_ed
            tmp["ed_diag_hosp"]=ed_diag_hosp
            tmp["ed_stay_id"]=ed_stay_id
            tmp["ed_hadm_id"]=ed_hadm_id
            result.append(tmp)
        df_full = pd.DataFrame(result)
        df_full["hosp_diag_hosp"]=df_full["hosp_diag_hosp"].apply(lambda x: [] if x is None else x)
        df_full.to_pickle(target_path/"records_w_diag.pkl")
        
    #################################################################################################
    print("Step 3: Map everything to ICD10 and enrich with more metadata to create output records_w_diag_icd10.pkl")
    if((target_path/"records_w_diag_icd10.pkl").exists()):
        print("Skipping: using existing records_w_diag_icd10.pkl")
        df_full = pd.read_pickle(target_path/"records_w_diag_icd10.pkl")
    else:
        df_full["hosp_diag_hosp"]=df_full["hosp_diag_hosp"].apply(lambda x: [icd_mapping[y] for y in x])
        df_full["hosp_diag_hosp"]=df_full["hosp_diag_hosp"].apply(lambda x: list(set([y for y in x if (y!="NoDx" and y!=None)])))
        df_full["ed_diag_hosp"]=df_full["ed_diag_hosp"].apply(lambda x: [icd_mapping[y] for y in x])
        df_full["ed_diag_hosp"]=df_full["ed_diag_hosp"].apply(lambda x: list(set([y for y in x if (y!="NoDx" and y!=None)])))
        df_full["ed_diag_ed"]=df_full["ed_diag_ed"].apply(lambda x: [icd_mapping[y] for y in x if y!="NoDx"])
        df_full["ed_diag_ed"]=df_full["ed_diag_ed"].apply(lambda x: list(set([y for y in x if (y!="NoDx" and y!=None)])))
        #ed or hosp ecgs with discharge diagnosis
        df_full["all_diag_hosp"]=df_full.apply(lambda row: list(set(row["hosp_diag_hosp"]+row["ed_diag_hosp"])),axis=1)
        # 'all_diag_all': 'all_diag_hosp' if available otherwise 'ed_diag_ed'
        df_full['all_diag_all'] = df_full.apply(lambda row: row['all_diag_hosp'] if row['all_diag_hosp'] else row['ed_diag_ed'],axis=1)
        

        #add demographics
        df_hosp_patients = pd.read_csv(mimic_path/"hosp/patients.csv.gz")
        df_full=df_full.join(df_hosp_patients.set_index("subject_id"),on="subject_id")
        df_full["age"]=df_full.ecg_time.apply(lambda x: x.year)-df_full.anchor_year+df_full.anchor_age

        #add ecg number within stay
        df_full["ecg_no_within_stay"]=-1
        df_full=df_full.sort_values(["subject_id","ecg_time"],ascending=True)

        df_full.loc[~df_full.ed_stay_id.isna(),"ecg_no_within_stay"]=df_full[~df_full.ed_stay_id.isna()].groupby("ed_stay_id",as_index=False).cumcount()
        df_full.loc[~df_full.hosp_hadm_id.isna(),"ecg_no_within_stay"]=df_full[~df_full.hosp_hadm_id.isna()].groupby("hosp_hadm_id",as_index=False).cumcount()

        df_full["ecg_taken_in_ed"]=df_full["ed_stay_id"].notnull()
        df_full["ecg_taken_in_hosp"]=df_full["hosp_hadm_id"].notnull()
        df_full["ecg_taken_in_ed_or_hosp"]=(df_full["ecg_taken_in_ed"]|df_full["ecg_taken_in_hosp"])
        
        
        # Fols used in the manuscript experiments, use them for reproducibility.
        df_full["fold"] = np.load('utils/folds.npy')
        
        # STRATIFIED FOLDS based on'all_diag'. folds not used in experiments, but provided for convenience
        df_full, _ = prepare_mimic_ecg('mimic_all_all_allfirst_all_2000_5A',target_folder,df_mapped=None,df_diags=df_full)
        df_full['label_train'] = df_full['label_train'].apply(lambda x: x if x else ['outpatient'])
        df_full.rename(columns={'label_train':'label_strat_all2all'}, inplace=True)
        age_bins = pd.qcut(df_full['age'], q=4)
        unique_intervals = age_bins.cat.categories
        bin_labels = {interval: f'{interval.left}-{interval.right}' for interval in unique_intervals}
        df_full['age_bin'] = age_bins.map(bin_labels)
        df_full['age_bin'] = df_full['age_bin'].cat.add_categories(['missing']).fillna('missing')
        df_full['gender'] = df_full['gender'].fillna('missing')
        
        df_full['merged_strat'] = df_full.apply(lambda row: row['label_strat_all2all'] + [row['age_bin'], row['gender']], axis=1)
        
        col_label = 'merged_strat'
        col_group = 'subject_id'
        
        res = stratified_subsets(df_full,
                       col_label,
                       [0.05]*20,
                       col_group=col_group,
                       label_multi_hot=False,
                       random_seed=42)
        
        df_full['strat_fold'] = res
        df=df[["file_name",
               "study_id",
               "subject_id",
               "ecg_time",
               "ed_stay_id",
               "ed_hadm_id",
               "hosp_hadm_id",
               "ed_diag_ed",
               "ed_diag_hosp",
               "hosp_diag_hosp",
               "all_diag_hosp",
               "all_diag_all",
               "gender","age",
               "anchor_year",
               "anchor_age",
               "dod",
               "ecg_no_within_stay",
               "ecg_taken_in_ed",
               "ecg_taken_in_hosp",
               "ecg_taken_in_ed_or_hosp",
               "fold",
               "strat_fold"]]
        df_full.to_csv(target_path/"records_w_diag_icd10.csv", index=False)
        
        
    if numpy_memmap:
        
    
        print("Step 4: Convert signals into numpy in  target-path/processed")
        (target_path/"processed").mkdir(parents=True, exist_ok=True)
        df,_,_,_=prepare_mimicecg(zip_file_path, target_folder=target_path/"processed")

        print("Step 5: Reformat as memmap for fast access")
        (target_path/"memmap").mkdir(parents=True, exist_ok=True)
        reformat_as_memmap(df, target_path/"memmap/memmap.npy", data_folder=target_path/"processed", annotation=False, max_len=0, delete_npys=True,col_data="data",col_lbl=None, batch_length=0, skip_export_signals=False)
    
    
    print("Done.")
        
    

if __name__ == '__main__':
    main()
