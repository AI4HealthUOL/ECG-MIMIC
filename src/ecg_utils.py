import wfdb

import os
import shutil
import zipfile

import numpy as np
import pandas as pd

import resampy
from tqdm.auto import tqdm
from pathlib import Path


import datetime

from clinical_ts.timeseries_utils import *

from sklearn.model_selection import StratifiedKFold

channel_stoi_default = {"i": 0, "ii": 1, "v1":2, "v2":3, "v3":4, "v4":5, "v5":6, "v6":7, "iii":8, "avr":9, "avl":10, "avf":11, "vx":12, "vy":13, "vz":14}

def get_stratified_kfolds(labels,n_splits,random_state):
    skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=random_state)
    return skf.split(np.zeros(len(labels)),labels)

def resample_data(sigbufs, channel_labels, fs, target_fs, channels=12, channel_stoi=None):#,skimage_transform=True,interpolation_order=3):
    channel_labels = [c.lower() for c in channel_labels]
    #https://github.com/scipy/scipy/issues/7324 zoom issues
    factor = target_fs/fs
    timesteps_new = int(len(sigbufs)*factor)
    if(channel_stoi is not None):
        data = np.zeros((timesteps_new, channels), dtype=np.float32)
        for i,cl in enumerate(channel_labels):
            if(cl in channel_stoi.keys() and channel_stoi[cl]<channels):
                data[:,channel_stoi[cl]] = resampy.resample(sigbufs[:,i], fs, target_fs).astype(np.float32)
    else:
        data = resampy.resample(sigbufs, fs, target_fs, axis=0).astype(np.float32)
    return data

def prepare_mimicecg(data_path="", clip_amp=3, target_fs=100, channels=12, strat_folds=20, channel_stoi=channel_stoi_default, target_folder=None, recreate_data=True):

    def fix_nans_and_clip(signal,clip_amp=3):
        for i in range(signal.shape[1]):
            tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
            signal[:,i]= np.clip(tmp,a_max=clip_amp, a_min=-clip_amp) if clip_amp>0 else tmp
    
    if(recreate_data):
        target_folder = Path(target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(Path(data_path)/"mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip", 'r') as archive:
            lst = archive.namelist()
            lst = [x for x in lst if x.endswith(".hea")]

            meta = []
            for l in tqdm(lst):
                archive.extract(l, path="tmp_dir/")
                archive.extract(l[:-3]+"dat", path="tmp_dir/")
                filename = Path("tmp_dir")/l
                sigbufs, header = wfdb.rdsamp(str(filename)[:-4])
            
                tmp={}
                tmp["data"]=filename.parent.parent.stem+"_"+filename.parent.stem+".npy" #patientid_study.npy
                tmp["study_id"]=int(filename.stem)
                tmp["subject_id"]=int(filename.parent.parent.stem[1:])
                tmp['ecg_time']= datetime.datetime.combine(header["base_date"],header["base_time"])
                tmp["nans"]= list(np.sum(np.isnan(sigbufs),axis=0))#save nans channel-dependent
                if(np.sum(tmp["nans"])>0):#fix nans
                    fix_nans_and_clip(sigbufs,clip_amp=clip_amp)
                elif(clip_amp>0):
                    sigbufs = np.clip(sigbufs,a_max=clip_amp,a_min=-clip_amp)

                data = resample_data(sigbufs=sigbufs,channel_stoi=channel_stoi,channel_labels=header['sig_name'],fs=header['fs'],target_fs=target_fs,channels=channels)
                
                assert(target_fs<=header['fs'])
                np.save(target_folder/tmp["data"],data)
                meta.append(tmp)
                
                os.unlink("tmp_dir/"+l)
                os.unlink("tmp_dir/"+l[:-3]+"dat")
                shutil.rmtree("tmp_dir")

        df = pd.DataFrame(meta)

        #random split by patients
        #unique_patients = np.unique(df.subject_id)
        #splits_patients = get_stratified_kfolds(np.zeros(len(unique_patients)),n_splits=strat_folds,random_state=42)
        #df["fold"]=-1
        #for i,split in enumerate(splits_patients):
        #    df.loc[df.subject_id.isin(unique_patients[split[-1]]),"fold"]=i
        
        #add means and std
        dataset_add_mean_col(df,data_folder=target_folder)
        dataset_add_std_col(df,data_folder=target_folder)
        dataset_add_length_col(df,data_folder=target_folder)

        #save means and stds
        mean, std = dataset_get_stats(df)

        #save
        lbl_itos=[]
        save_dataset(df,lbl_itos,mean,std,target_folder)
    else:
        df, lbl_itos, mean, std = load_dataset(target_folder,df_mapped=False)
    return df, lbl_itos, mean, std

