import torch
from torch import nn
import lightning.pytorch as lp
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

import os
import subprocess
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar
from clinical_ts.xresnet1d import xresnet1d50,xresnet1d101
from clinical_ts.inception1d import inception1d
from clinical_ts.s4_model import S4Model
from clinical_ts.misc_utils import add_default_args, LRMonitorCallback

#################
#specific
from clinical_ts.timeseries_utils import *
from clinical_ts.schedulers import *
from clinical_ts.eval_utils_cafa import multiclass_roc_curve
from clinical_ts.bootstrap_utils import *

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from pathlib import Path
import numpy as np
import pandas as pd

# mimic_{subsettrain}{labelsettrain}{subsettest}{labelsettest}{mincnt}_{digits}
# {subsettrain}:all/ed/hosp/allnonzero/ednonzero/hospnonzero/allnonzerofirst/ednonzerofirst/hospnonzerofirst/allfirst/edfirst/hospfirst default: allnonzero
# {labelsettrain}: {all/hosp/ed} first part selects the label set all: both ed diagnosis and hosp diagnosis hosp: just hosp diagnosis ed: just ed diagnosis
# {subsettest}/{labelsettest}: similar than {subsettrain}/{labelsettrain}
# {mincnt}: minimum number of samples per label
# {digits}: 3/4/5/3A/4A/5A; append an A to include all ancestors

MLFLOW_AVAILABLE=True
try:
    import mlflow
    import mlflow.pytorch
    import argparse

    def namespace_to_dict(namespace):
        return {
            k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
            for k, v in vars(namespace).items()
        }
except ImportError:
    MLFLOW_AVAILABLE=False

def get_git_revision_short_hash():
    return ""#subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def multihot_encode(x, num_classes):
    res = np.zeros(num_classes,dtype=np.float32)
    for y in x:
        res[y]=1
    return res
############################################################################################################
#at this scope to avoid pickle issues
def mcrc_flat(targs,preds,classes):
    _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
    return np.array(list(res.values()))

def prepare_consistency_mapping(codes_unique, codes_unique_all, propagate_all=False):
    res={}
    for c in codes_unique:
        if(propagate_all):
            res[c]=[c[:i] for i in range(3,len(c)+1)]
        else:#only propagate if categories are already present
            res[c]=np.intersect1d([c[:i] for i in range(3,len(c)+1)],codes_unique_all)
    return res

def prepare_mimic_ecg(finetune_dataset, target_folder, df_mapped=None):
    # e.g. mimic_all_all_all_all_2000_5A #ED mimic_{subsettrain}_{labelsettrain}_{subsettest}_{labelsettest}_{mincnt}_{digits} where _{digits} is optional
    #subsettrain: all/ed/hosp/allnonzero/ednonzero/hospnonzero/allnonzerofirst/ednonzerofirst/hospnonzerofirst/allfirst/edfirst/hospfirst default: allnonzero
    #labelsettrain: {all/hosp/ed}{/af/I} first part selects the label set all: both ed diagnosis and hosp diagnosis hosp: just hosp diagnosis ed: just ed diagnosis; second part: can be omitted or af for af labels or collection of uppercase letters such as I to select specific label sets
    #similar for subsettest/labelsettest but labelsettest can only be {all/hosp/ed}
    #digits: 3/4/5/3A/4A/5A or just empty corresponding to I48, I48.1 or I48.19; append an A to include all ancestors
    def flatten(l):
        return [item for sublist in l for item in sublist]  

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

    #load label dataframe
    df_diags = pd.read_pickle(target_folder/"records_w_diag_icd10.pkl")

    #select the desired label set (train)
    if(labelsettrain.startswith("hosp")):#just hospital discharge diagnosis
        df_diags["label_train"]=df_diags["ed_or_hosp_diag"]
        labelsettrain=labelsettrain[len("hosp"):]
    elif(labelsettrain.startswith("ed")):#just ED discharge diagnosis
        df_diags["label_train"]=df_diags["ed_diag_ed"]
        labelsettrain=labelsettrain[len("ed"):]
    elif(labelsettrain.startswith("all")):#both ed and hospital discharge diagnosis (the latter if available)
        df_diags["label_train"]=df_diags["ed_or_hosp_diag"]
        selection = df_diags["ed_or_hosp_diag"].apply(lambda x: len(x)==0)
        df_diags.loc[selection,"label_train"]=df_diags.loc[selection,"ed_diag_ed"]# if no discharge diagnosis is available, use the ED diagnosis
        labelsettrain=labelsettrain[len("all"):]
    else:
        assert(False)

    if(labelsettest.startswith("hosp")):#just hospital discharge diagnosis
        df_diags["label_test"]=df_diags["ed_or_hosp_diag"]
    elif(labelsettest.startswith("ed")):#just ED discharge diagnosis
        df_diags["label_test"]=df_diags["ed_diag_ed"]
    elif(labelsettest.startswith("all")):#both ed and hospital discharge diagnosis (the latter if available)
        df_diags["label_test"]=df_diags["ed_or_hosp_diag"]
        selection = df_diags["ed_or_hosp_diag"].apply(lambda x: len(x)==0)
        df_diags.loc[selection,"label_test"]=df_diags.loc[selection,"ed_diag_ed"]# if no discharge diagnosis is available, use the ED diagnosis
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
    
    if(df_mapped is not None):
        print("Label set:",len(lbl_itos),"labels.")#,lbl_itos)
        
        #join the two dataframes
        df_diags = df_diags.set_index("study")
        df_diags.drop(["patient_id","ecg_time"],axis=1,inplace=True)
        df_mapped = df_mapped.join(df_diags,on="study")
        max_fold = df_mapped.strat_fold.max()
        
        #TRAIN select the desired subset (all/ed/hosp/allnonzero/ednonzero/hospnonzero)
        df_mappedtrain = df_mapped[df_mapped.strat_fold<max_fold-1].copy()#pick the first n-2 folds
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
        df_mappedtest = df_mapped[df_mapped.strat_fold>=(max_fold-1)].copy()#pick the final two folds
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
    
class Main_ECG(lp.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr

        print(hparams)
        if(hparams.finetune_dataset == "thew"):
            num_classes = 5
        elif(hparams.finetune_dataset == "ribeiro_train"):
            num_classes = 6
        elif(hparams.finetune_dataset == "ptbxl_super"):
            num_classes = 5
        elif(hparams.finetune_dataset == "ptbxl_sub"):
            num_classes = 24
        elif(hparams.finetune_dataset == "ptbxl_all"):
            num_classes = 71
        elif(hparams.finetune_dataset.startswith("segrhythm")):
            num_classes = int(hparams.finetune_dataset[9:])
        elif(hparams.finetune_dataset.startswith("rhythm")):
            num_classes = int(hparams.finetune_dataset[6:])
        elif(hparams.finetune_dataset.startswith("mimic")):
            _, lbl_itos = prepare_mimic_ecg(self.hparams.finetune_dataset,Path(self.hparams.data.split(",")[0]))
            num_classes = len(lbl_itos)

        # also works in the segmentation case
        self.criterion = F.cross_entropy if (hparams.finetune_dataset == "thew" or hparams.finetune_dataset.startswith("segrhythm"))  else F.binary_cross_entropy_with_logits
    
        if(hparams.architecture=="xresnet1d50"):
            self.model = xresnet1d50(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="xresnet1d101"):
            self.model = xresnet1d101(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="inception1d"):
            self.model = inception1d(input_channels=hparams.input_channels, num_classes=num_classes)
        elif(hparams.architecture=="s4"):
            self.model = S4Model(d_input=hparams.input_channels, d_output=num_classes, l_max=self.hparams.input_size, d_state=self.hparams.s4_n, d_model=self.hparams.s4_h, n_layers = self.hparams.s4_layers,bidirectional=True)#,backbone="s4new")
        else:
            assert(False)
        

    def forward(self, x, **kwargs):
        # QUICK FIX FOR REMAINING NANS IN INPUT
        x[torch.isnan(x)]=0
        return self.model(x, **kwargs)
    
    def on_validation_epoch_end(self):
        for i in range(len(self.val_preds)):
            self.on_valtest_epoch_eval({"preds":self.val_preds[i], "targs":self.val_targs[i]}, dataloader_idx=i, test=False)
            self.val_preds[i].clear()
            self.val_targs[i].clear()
    
    def on_test_epoch_end(self):
        for i in range(len(self.test_preds)):
            self.on_valtest_epoch_eval({"preds":self.test_preds[i], "targs":self.test_targs[i]}, dataloader_idx=i, test=True)
            self.test_preds[i].clear()
            self.test_targs[i].clear()

    def eval_scores(self, targs,preds,classes=None,bootstrap=False):
        _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
        if(bootstrap):
            point,low,high,nans = empirical_bootstrap((targs,preds), mcrc_flat, n_iterations=self.hparams.bootstrap_iterations ,score_fn_kwargs={"classes":classes},ignore_nans=True)
            res2={}
            for i,k in enumerate(res.keys()):
                res2[k]=point[i]
                res2[k+"_low"]=low[i]
                res2[k+"_high"]=high[i]
                res2[k+"_nans"]=nans[i]
            return res2
        return res 

    def on_valtest_epoch_eval(self, outputs_all, dataloader_idx, test=False):
        #for dataloader_idx,outputs in enumerate(outputs_all): #multiple val dataloaders
            preds_all = torch.cat(outputs_all["preds"]).cpu()
            targs_all = torch.cat(outputs_all["targs"]).cpu()
            # apply softmax/sigmoid to ensure that aggregated scores are calculated based on them
            if(self.hparams.finetune_dataset == "thew" or self.hparams.finetune_dataset.startswith("segrhythm")):
                preds_all = F.softmax(preds_all.float(),dim=-1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
            else:
                preds_all = torch.sigmoid(preds_all.float())
            
            preds_all = preds_all.numpy()
            targs_all = targs_all.numpy()
            #instance level score
            res = self.eval_scores(targs_all,preds_all,classes=self.lbl_itos,bootstrap=test)
            res = {k+"_auc_noagg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res.items()}
            res = {k.replace("(","_").replace(")","_"):v for k,v in res.items()}#avoid () for mlflow
            self.log_dict(res)
            print("epoch",self.current_epoch,"test" if test else "val","noagg:",res["macro_auc_noagg_"+("test" if test else "val")+str(dataloader_idx)])#,"agg:",res_agg)
            
            preds_all_agg,targs_all_agg = aggregate_predictions(preds_all,targs_all,self.test_idmaps[dataloader_idx] if test else self.val_idmaps[dataloader_idx],aggregate_fn=np.mean)
            res_agg = self.eval_scores(targs_all_agg,preds_all_agg,classes=self.lbl_itos,bootstrap=test)
            res_agg = {k+"_auc_agg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res_agg.items()}
            res_agg = {k.replace("(","_").replace(")","_"):v for k,v in res_agg.items()}
            self.log_dict(res_agg)

            #export predictions
            if(test and self.hparams.export_predictions_path!=""):
                df_test = pd.read_pickle(Path(self.hparams.export_predictions_path)/("df_test"+str(dataloader_idx)+".pkl"))
                df_test["preds"]=list(preds_all_agg)
                df_test["targs"]=list(targs_all_agg)             
                df_test.to_pickle(Path(self.hparams.export_predictions_path)/("df_test"+str(dataloader_idx)+".pkl"))

            print("epoch",self.current_epoch,"test" if test else "val","agg:",res_agg["macro_auc_agg_"+("test" if test else "val")+str(dataloader_idx)])#,"agg:",res_agg)
            
    def setup(self, stage):
        rhythm = self.hparams.finetune_dataset.startswith("rhythm")
        if(rhythm):
            num_classes_rhythm = int(hparams.finetune_dataset[6:])    

        # configure dataset params
        chunkify_train = self.hparams.chunkify_train
        chunk_length_train = int(self.hparams.chunk_length_train*self.hparams.input_size) if chunkify_train else 0
        stride_train = int(self.hparams.stride_fraction_train*self.hparams.input_size)
        
        chunkify_valtest = True
        chunk_length_valtest = self.hparams.input_size if chunkify_valtest else 0
        stride_valtest = int(self.hparams.stride_fraction_valtest*self.hparams.input_size)

        train_datasets = []
        val_datasets = []
        test_datasets = []

        self.ds_mean = None
        self.ds_std = None
        self.lbl_itos = None

        for i,target_folder in enumerate(list(self.hparams.data.split(","))):
            
            target_folder = Path(target_folder)           
            
            df_mapped, lbl_itos,  mean, std = load_dataset(target_folder)
            
            print("Folder:",target_folder,"Samples:",len(df_mapped))

            if(self.ds_mean is None):
                if(self.hparams.finetune_dataset.startswith("rhythm") or self.hparams.finetune_dataset.startswith("segrhythm")):
                    self.ds_mean = np.array([0.,0.])
                    self.ds_std = np.array([1.,1.])
                else:
                    # always use PTB-XL stats
                    self.ds_mean = np.array([-0.00184586, -0.00130277,  0.00017031, -0.00091313, -0.00148835,  -0.00174687, -0.00077071, -0.00207407,  0.00054329,  0.00155546,  -0.00114379, -0.00035649])
                    self.ds_std = np.array([0.16401004, 0.1647168 , 0.23374124, 0.33767231, 0.33362807,  0.30583013, 0.2731171 , 0.27554379, 0.17128962, 0.14030828,   0.14606956, 0.14656108])

            #specific for PTB-XL
            if(self.hparams.finetune_dataset.startswith("ptbxl")):
                if(self.hparams.finetune_dataset=="ptbxl_super"):
                    ptb_xl_label = "label_diag_superclass"
                elif(self.hparams.finetune_dataset=="ptbxl_sub"):
                    ptb_xl_label = "label_diag_subclass"
                elif(self.hparams.finetune_dataset=="ptbxl_all"):
                    ptb_xl_label = "label_all"
                    
                lbl_itos= np.array(lbl_itos[ptb_xl_label])
                df_mapped["label"]= df_mapped[ptb_xl_label+"_filtered_numeric"].apply(lambda x: multihot_encode(x,len(lbl_itos)))
            elif(self.hparams.finetune_dataset == "ribeiro_train"):
                df_mapped = df_mapped[df_mapped.strat_fold>=0].copy()#select on labeled subset (-1 is unlabeled)
                df_mapped["label"]= df_mapped["label"].apply(lambda x: multihot_encode(x,len(lbl_itos))) #multi-hot encode
            elif(self.hparams.finetune_dataset.startswith("segrhythm")):
                num_classes_segrhythm = int(hparams.finetune_dataset[9:])  
                df_mapped = df_mapped[df_mapped.label.apply(lambda x: x<num_classes_segrhythm)]
                lbl_itos = lbl_itos[:num_classes_segrhythm]
            elif(self.hparams.finetune_dataset.startswith("mimic")):
                df_mapped, lbl_itos = prepare_mimic_ecg(self.hparams.finetune_dataset, target_folder, df_mapped=df_mapped)

            
            
            
            if(self.lbl_itos is None):
                self.lbl_itos = lbl_itos[:num_classes_rhythm] if rhythm else lbl_itos
            
            if(rhythm):
                if(self.hparams.segmentation):
                    tfms_ptb_xl_cpc = ToTensor(transpose_label=True)
                else:#map to global label for given window
                    def annotation_to_multilabel(lbl):
                        lbl_unique = np.unique(lbl)
                        lbl_unique = [x for x in lbl_unique if x<num_classes_rhythm]
                        return multihot_encode(lbl_unique,num_classes_rhythm)
                    tfms_ptb_xl_cpc = transforms.Compose([Transform(annotation_to_multilabel),ToTensor()])
            else:
                assert(self.hparams.segmentation is False)
                tfms_ptb_xl_cpc = ToTensor() if self.hparams.normalize is False else transforms.Compose([Normalize(self.ds_mean,self.ds_std),ToTensor()])
            
            max_fold_id = df_mapped.strat_fold.max() #unfortunately 1-based for PTB-XL; sometimes 100 (Ribeiro)
            df_train = df_mapped[df_mapped.strat_fold<max_fold_id-1]
            df_val = df_mapped[df_mapped.strat_fold==max_fold_id-1]
            df_test = df_mapped[df_mapped.strat_fold==max_fold_id]
            
            
            train_datasets.append(TimeseriesDatasetCrops(df_train,self.hparams.input_size,data_folder=target_folder,chunk_length=chunk_length_train,min_chunk_length=self.hparams.input_size, stride=stride_train,transforms=tfms_ptb_xl_cpc,col_lbl ="label" ,memmap_filename=target_folder/("memmap.npy")))
            val_datasets.append(TimeseriesDatasetCrops(df_val,self.hparams.input_size,data_folder=target_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms_ptb_xl_cpc,col_lbl ="label",memmap_filename=target_folder/("memmap.npy")))
            test_datasets.append(TimeseriesDatasetCrops(df_test,self.hparams.input_size,data_folder=target_folder,chunk_length=chunk_length_valtest,min_chunk_length=self.hparams.input_size, stride=stride_valtest,transforms=tfms_ptb_xl_cpc,col_lbl ="label",memmap_filename=target_folder/("memmap.npy")))
            
            if(self.hparams.export_predictions_path!=""):# save lbl_itos and test dataframe for later
                np.save(Path(self.hparams.export_predictions_path)/"lbl_itos.npy",self.lbl_itos)
                df_test.to_pickle(Path(self.hparams.export_predictions_path)/("df_test"+str(len(test_datasets)-1)+".pkl"))

            print("\n",target_folder)
            if(i<len(self.hparams.data)):
                print("train dataset:",len(train_datasets[-1]),"samples")
            print("val dataset:",len(val_datasets[-1]),"samples")
            print("test dataset:",len(test_datasets[-1]),"samples")
        

            
        if(len(train_datasets)>1): #multiple data folders
            print("\nCombined:")
            self.train_dataset = ConcatDatasetTimeseriesDatasetCrops(train_datasets)
            self.val_datasets = [ConcatDatasetTimeseriesDatasetCrops(val_datasets)]+val_datasets
            print("train dataset:",len(self.train_dataset),"samples")
            print("val datasets (total):",len(self.val_datasets[0]),"samples")
            self.test_datasets = [ConcatDatasetTimeseriesDatasetCrops(test_datasets)]+test_datasets
            print("test datasets (total):",len(self.test_datasets[0]),"samples")
        else: #just a single data folder
            self.train_dataset = train_datasets[0]
            self.val_datasets = val_datasets
            self.test_datasets = test_datasets

        #create empty lists for results
        self.val_preds=[[] for _ in range(len(self.val_datasets))]
        self.val_targs=[[] for _ in range(len(self.val_datasets))]
        self.test_preds=[[] for _ in range(len(self.test_datasets))]
        self.test_targs=[[] for _ in range(len(self.test_datasets))]
        
        # store idmaps for aggregation
        self.val_idmaps = [ds.get_id_mapping() for ds in self.val_datasets]
        self.test_idmaps = [ds.get_id_mapping() for ds in self.test_datasets]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=8, shuffle=True, drop_last = True)
        
    def val_dataloader(self):
        return [DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=8) for ds in self.val_datasets]
    
    def test_dataloader(self):
        return [DataLoader(ds, batch_size=self.hparams.batch_size, num_workers=8) for ds in self.test_datasets]
        
    def _step(self,data_batch, batch_idx, train, test=False, dataloader_idx=0):
        #if(torch.sum(torch.isnan(data_batch[0])).item()>0):#debugging
        #    print("nans",torch.sum(torch.isnan(data_batch[0])).item())
        preds_all = self.forward(data_batch[0])

        loss = self.criterion(preds_all,data_batch[1])
        self.log("train_loss" if train else ("test_loss" if test else "val_loss"), loss)
        
        if(not train and not test):
            self.val_preds[dataloader_idx].append(preds_all.detach())
            self.val_targs[dataloader_idx].append(data_batch[1])
        elif(not train and test):
            self.test_preds[dataloader_idx].append(preds_all.detach())
            self.test_targs[dataloader_idx].append(data_batch[1])
        
        return loss
    
    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch,batch_idx,train=True)
        
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=False, dataloader_idx=dataloader_idx)
    
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        return self._step(test_batch,batch_idx,train=False,test=True, dataloader_idx=dataloader_idx)
    
    def configure_optimizers(self):
        
        if(self.hparams.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
            
        params = self.parameters()

        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)

        if(self.hparams.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif(self.hparams.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        elif(self.hparams.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="linear"): #linear decay to be combined with warmup-invsqrt c.f. https://arxiv.org/abs/2106.04560
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.epochs*len(self.train_dataloader()))
        else:
            assert(False)
        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        ])
        
    def load_weights_from_checkpoint(self, checkpoint):
        """ Function that loads the weights from a given checkpoint file. 
        based on https://github.com/PyTorchLightning/pytorch-lightning/issues/525
        """
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def load_state_dict(self, state_dict):
        #S4-compatible load_state_dict
        for name, param in self.named_parameters():
            param.data = state_dict[name].data.to(param.device)
        for name, param in self.named_buffers():
            param.data = state_dict[name].data.to(param.device)

######################################################################################################
# MISC
######################################################################################################
def load_from_checkpoint(pl_model, checkpoint_path):
    """ load from checkpoint function that is compatible with S4
    """
    lightning_state_dict = torch.load(checkpoint_path)
    state_dict = lightning_state_dict["state_dict"]
    
    for name, param in pl_model.named_parameters():
        param.data = state_dict[name].data
    for name, param in pl_model.named_buffers():
        param.data = state_dict[name].data


    
#####################################################################################################
#ARGPARSER
#####################################################################################################
def add_model_specific_args(parser):
    parser.add_argument("--input-channels", type=int, default=12)
    parser.add_argument("--architecture", type=str, help="xresnet1d50/xresnet1d101/inception1d/s4", default="xresnet1d50")
    
    parser.add_argument("--s4-n", type=int, default=8, help='S4: N (Sashimi default:64)')
    parser.add_argument("--s4-h", type=int, default=512, help='S4: H (Sashimi default:64)')
    parser.add_argument("--s4-layers", type=int, default=4, help='S4: number of layers (Sashimi default:8)')
    parser.add_argument("--s4-batchnorm", action='store_true', help='S4: use BN instead of LN')
    parser.add_argument("--s4-prenorm", action='store_true', help='S4: use prenorm')
     
    return parser

def add_application_specific_args(parser):
    parser.add_argument("--normalize", action='store_true', help='Normalize input using dataset stats')
    parser.add_argument("--finetune-dataset", type=str, help="...", default="ptbxl_all")
    parser.add_argument("--chunk-length-train", type=float, default=1.,help="training chunk length in multiples of input size")
    parser.add_argument("--stride-fraction-train", type=float, default=1.,help="training stride in multiples of input size")
    parser.add_argument("--stride-fraction-valtest", type=float, default=1.,help="val/test stride in multiples of input size")
    parser.add_argument("--chunkify-train", action='store_true')
    
    parser.add_argument("--segmentation", action='store_true')
    
    parser.add_argument("--eval-only", type=str, help="path to model checkpoint for evaluation", default="")
    parser.add_argument("--bootstrap-iterations", type=int, help="number of bootstrap iterations for score estimation", default=1000)

    parser.add_argument("--export-predictions-path", type=str, default="", help="path to directory to export predictions")
    return parser
            
###################################################################################################
#MAIN
###################################################################################################
if __name__ == '__main__':
    parser = add_default_args()
    parser = add_model_specific_args(parser)
    parser = add_application_specific_args(parser)

    hparams = parser.parse_args()
    hparams.executable = "main_ecg"
    hparams.revision = get_git_revision_short_hash()
    if(hparams.eval_only!=""):
        hparams.epochs=0

    if not os.path.exists(hparams.output_path):
        os.makedirs(hparams.output_path)
        
    model = Main_ECG(hparams)

    logger = TensorBoardLogger(
        save_dir=hparams.output_path,
        #version="",#hparams.metadata.split(":")[0],
        name="")
    print("Output directory:",logger.log_dir)

    if(MLFLOW_AVAILABLE):
        mlflow.set_experiment(hparams.executable)
        mlflow.pytorch.autolog(log_models=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=logger.log_dir,
        filename="best_model",
        save_top_k=1,
		save_last=True,
        verbose=True,
        monitor= "macro_auc_agg_val0" ,#val_loss/dataloader_idx_0
        mode='max')

    lr_monitor = LearningRateMonitor(logging_interval="step")
    #lr_monitor2 = LRMonitorCallback(start=False,end=True)#interval="step")

    callbacks = [checkpoint_callback,lr_monitor]#,lr_monitor2]

    if(hparams.refresh_rate>0):
        callbacks.append(TQDMProgressBar(refresh_rate=hparams.refresh_rate))

    trainer = lp.Trainer(
        num_sanity_val_steps=0,#no debugging
        #overfit_batches=50,#debugging

        accumulate_grad_batches=hparams.accumulate,
        max_epochs=hparams.epochs,
        min_epochs=hparams.epochs,
        
        default_root_dir=hparams.output_path,
        
        logger=logger,
        callbacks = callbacks,
        benchmark=True,
    
        accelerator="gpu" if hparams.gpus>0 else "cpu",
        devices=hparams.gpus if hparams.gpus>0 else 1,
        num_nodes=hparams.num_nodes,
        precision=hparams.precision,
        #distributed_backend=hparams.distributed_backend,
        
        enable_progress_bar=hparams.refresh_rate>0)
        
    if(hparams.auto_batch_size):#auto tune batch size batch size
        tuner=Tuner(trainer)
        tuner.scale_batch_size(model, mode="binsearch")

    if(hparams.lr_find):# lr find
        tuner=Tuner(trainer)
        lr_finder = tuner.lr_find(model)

    if(hparams.epochs>0 and hparams.eval_only==""):
        if(MLFLOW_AVAILABLE):
            with mlflow.start_run(run_name=hparams.metadata) as run:
                for k,v  in dict(hparams._get_kwargs()).items():
                    mlflow.log_param(k," " if v=="" else v)#mlflow as issues with empty strings
                trainer.fit(model,ckpt_path= None if hparams.resume=="" else hparams.resume)
                trainer.test(model,ckpt_path="best")
        else:
            trainer.fit(model,ckpt_path= None if hparams.resume=="" else hparams.resume)
            trainer.test(model,ckpt_path="best")

    elif(hparams.eval_only!=""):#eval only
    #else:
        if(MLFLOW_AVAILABLE):
            with mlflow.start_run(run_name=hparams.metadata) as run:
                for k,v  in dict(hparams._get_kwargs()).items():
                    mlflow.log_param(k," " if v=="" else v)#mlflow as issues with empty strings
                #trainer.fit(model)#mock fit call as mlflow logging is only invoked for fit
                trainer.test(model,ckpt_path=hparams.eval_only)
        else:
            trainer.test(model,ckpt_path=hparams.eval_only)
