__all__ = ['nn_upsample','resample_labels','butter_filter', 'butter_filter_frequency_response', 'apply_butter_filter', 'save_dataset', 'load_dataset',
           'dataset_add_chunk_col', 'dataset_add_length_col', 'dataset_add_labels_col', 'dataset_add_mean_col',
           'dataset_add_median_col', 'dataset_add_std_col', 'dataset_add_iqr_col', 'dataset_get_stats',
           'npys_to_memmap_batched', 'npys_to_memmap', 'reformat_as_memmap', 'ConcatDatasetTimeseriesDatasetCrops', 'TimeseriesDatasetCrops', 'RandomCrop',
           'CenterCrop', 'GaussianNoise', 'Rescale', 'ToTensor', 'Normalize', 'NormalizeBatch', 'ButterFilter',
           'ChannelFilter', 'Transform', 'StaticTransform', 'TupleTransform', 'SequenceToSampleLabelTransform', 'aggregate_predictions']


import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import Compose

from pathlib import Path
from scipy.stats import iqr


#workaround for windows pickles
from sys import platform
import pathlib
if platform == "linux" or platform == "linux2":
    pathlib.WindowsPath = pathlib.PosixPath


try:
    import pickle5 as pickle
except ImportError as e:
    import pickle

#Note: due to issues with the numpy rng for multiprocessing (https://github.com/pytorch/pytorch/issues/5059) that could be fixed by a custom worker_init_fn we use random throught for convenience
import random

#Note: multiprocessing issues with python lists and dicts (https://github.com/pytorch/pytorch/issues/13246) and pandas dfs (https://github.com/pytorch/pytorch/issues/5902)
import multiprocessing as mp

from skimage import transform

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from scipy.signal import butter, sosfilt, sosfiltfilt, sosfreqz
from scipy.interpolate import interp1d

from tqdm.auto import tqdm

from collections import namedtuple

tsdata_static = namedtuple("tsdata_static",("data","label","static"))
tsdata = namedtuple("tsdata",("data","label"))

def nn_upsample(xin, yin, xout):
    '''performs nearest neighbor upsampling of the integer array yin with values at xin for new datapoints at xout'''
    f = interp1d(xin,yin, kind="nearest",bounds_error=False,fill_value="extrapolate")
    return f(xout).astype(np.int64)

def resample_labels(startpts, labels, startpts_to_mid, startpts_new, startpts_to_mid_new):
    '''resamples integer labels labels at starpts+startpts_to_mid to new anchor points at startpts_new+startpts_to_mid_new'''
    if(isinstance(startpts_to_mid,float) or isinstance(startpts_to_mid,int)):
        startpts_to_mid = np.ones_like(startpts)*startpts_to_mid
    if(isinstance(startpts_to_mid_new,float) or isinstance(startpts_to_mid_new,int)):
        startpts_to_mid_new = np.ones_like(startpts_new)*startpts_to_mid_new
    midpts = np.array(startpts)+startpts_to_mid
    midpts_new = np.array(startpts_new)+startpts_to_mid_new
    return nn_upsample(midpts, labels, midpts_new)

#https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
def butter_filter(lowcut=10, highcut=20, fs=50, order=5, btype='band'):
    '''returns butterworth filter with given specifications'''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    sos = butter(order, [low, high] if btype=="band" else (low if btype=="low" else high), analog=False, btype=btype, output='sos')
    return sos

def butter_filter_frequency_response(filter):
    '''returns frequency response of a given filter (result of call of butter_filter)'''
    w, h = sosfreqz(filter)
    #gain vs. freq(Hz)
    #plt.plot((fs * 0.5 / np.pi) * w, abs(h))
    return w,h

def apply_butter_filter(data, filter, forwardbackward=True):
    '''pass filter from call of butter_filter to data (assuming time axis at dimension 0)'''
    if(forwardbackward):
        return sosfiltfilt(filter, data, axis=0)
    else:
        data = sosfilt(filter, data, axis=0)


def save_dataset(df,lbl_itos,mean,std,target_root,filename_postfix="",protocol=4):
    target_root = Path(target_root)
    df.to_pickle(target_root/("df"+filename_postfix+".pkl"), protocol=protocol)

    if(isinstance(lbl_itos,dict)):#dict as pickle
        outfile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "wb")
        pickle.dump(lbl_itos, outfile, protocol=protocol)
        outfile.close()
    else:#array
        np.save(target_root/("lbl_itos"+filename_postfix+".npy"),lbl_itos)

    np.save(target_root/("mean"+filename_postfix+".npy"),mean)
    np.save(target_root/("std"+filename_postfix+".npy"),std)

def load_dataset(target_root,filename_postfix="",df_mapped=True):
    target_root = Path(target_root)

    if(df_mapped):
        df = pickle.load(open(target_root/("df_memmap"+filename_postfix+".pkl"), "rb"))
    else:
        df = pickle.load(open(target_root/("df"+filename_postfix+".pkl"), "rb"))


    if((target_root/("lbl_itos"+filename_postfix+".pkl")).exists()):#dict as pickle
        infile = open(target_root/("lbl_itos"+filename_postfix+".pkl"), "rb")
        lbl_itos=pickle.load(infile)
        infile.close()
    else:#array
        lbl_itos = np.load(target_root/("lbl_itos"+filename_postfix+".npy"))


    mean = np.load(target_root/("mean"+filename_postfix+".npy"))
    std = np.load(target_root/("std"+filename_postfix+".npy"))
    return df, lbl_itos, mean, std


def dataset_add_chunk_col(df, col="data"):
    '''add a chunk column to the dataset df'''
    df["chunk"]=df.groupby(col).cumcount()

def dataset_add_length_col(df, col="data", data_folder=None):
    '''add a length column to the dataset df'''
    df[col+"_length"]=df[col].apply(lambda x: len(np.load(x if data_folder is None else data_folder/x, allow_pickle=True)))

def dataset_add_labels_col(df, col="label", data_folder=None):
    '''add a column with unique labels in column col'''
    df[col+"_labels"]=df[col].apply(lambda x: list(np.unique(np.load(x if data_folder is None else data_folder/x, allow_pickle=True))))

def dataset_add_mean_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_mean"]=df[col].apply(lambda x: np.mean(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_median_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with median'''
    df[col+"_median"]=df[col].apply(lambda x: np.median(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_std_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_std"]=df[col].apply(lambda x: np.std(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_add_iqr_col(df, col="data", axis=(0), data_folder=None):
    '''adds a column with mean'''
    df[col+"_iqr"]=df[col].apply(lambda x: iqr(np.load(x if data_folder is None else data_folder/x, allow_pickle=True),axis=axis))

def dataset_get_stats(df, col="data", simple=True):
    '''creates (weighted) means and stds from mean, std and length cols of the df'''
    if(simple):
        return df[col+"_mean"].mean(), df[col+"_std"].mean()
    else:
        #https://notmatthancock.github.io/2017/03/23/simple-batch-stat-updates.html
        #or https://gist.github.com/thomasbrandon/ad5b1218fc573c10ea4e1f0c63658469
        def combine_two_means_vars(x1,x2):
            (mean1,var1,n1) = x1
            (mean2,var2,n2) = x2
            mean = mean1*n1/(n1+n2)+ mean2*n2/(n1+n2)
            var = var1*n1/(n1+n2)+ var2*n2/(n1+n2)+n1*n2/(n1+n2)/(n1+n2)*np.power(mean1-mean2,2)
            return (mean, var, (n1+n2))

        def combine_all_means_vars(means,vars,lengths):
            inputs = list(zip(means,vars,lengths))
            result = inputs[0]

            for inputs2 in inputs[1:]:
                result= combine_two_means_vars(result,inputs2)
            return result

        means = list(df[col+"_mean"])
        vars = np.power(list(df[col+"_std"]),2)
        lengths = list(df[col+"_length"])
        mean,var,length = combine_all_means_vars(means,vars,lengths)
        return mean, np.sqrt(var)


def npys_to_memmap_batched(npys, target_filename, max_len=0, delete_npys=True, batched_npy=False, batch_length=900000):
    '''
    analogous to npys_to_memmap but processes batches of files before flushing them into memmap for faster processing
    '''
    memmap = None
    start = np.array([0])#start_idx in current memmap file (always already the next start- delete last token in the end)
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]#shapes of all memmap files

    data = []
    data_lengths=[]
    dtype = None

    target_filename = Path(target_filename)

    for idx,npy in tqdm(enumerate(npys),total=len(npys)):
        data_batched = np.load(npy, allow_pickle=True)

        for data_tmp in (tqdm(data_batched,leave=False) if batched_npy else [data_batched]):
            data.append(data_tmp)
            data_lengths.append(len(data[-1]))
            if(idx==len(npys)-1 or np.sum(data_lengths)>batch_length):#flush
                data = np.concatenate(data,axis=0)#concatenate along time axis (still axis 0 at this stage)
                if(memmap is None or (max_len>0 and start[-1]>max_len)):#new memmap file has to be created
                    if(max_len>0):
                        filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames))+".npy"))
                    else:
                        filenames.append(target_filename)

                    shape.append([np.sum(data_lengths)]+[l for l in data.shape[1:]])#insert present shape

                    if(memmap is not None):#an existing memmap exceeded max_len
                        del memmap
                    #create new memmap
                    start[-1] = 0
                    start = np.concatenate([start,np.cumsum(data_lengths)])
                    length = np.concatenate([length,data_lengths])

                    memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
                else:
                    #append to existing memmap
                    start = np.concatenate([start,start[-1]+np.cumsum(data_lengths)])
                    length = np.concatenate([length,data_lengths])
                    shape[-1] = [start[-1]]+[l for l in data.shape[1:]]
                    memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple(shape[-1]))

                #store mapping memmap_id to memmap_file_id
                file_idx=np.concatenate([file_idx,[(len(filenames)-1)]*len(data_lengths)])
                #insert the actual data
                memmap[start[-len(data_lengths)-1]:start[-len(data_lengths)-1]+len(data)]=data[:]
                memmap.flush()
                dtype = data.dtype
                data = []#reset data storage
                data_lengths = []

    start= start[:-1]#remove the last element
    #cleanup
    for npy in npys:
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=dtype,filenames=filenames)


def npys_to_memmap(npys, target_filename, max_len=0, delete_npys=True, batched_npy=False):
    '''
    converts list of filenames pointing to npy files into a memmap file with target_filename
    max_len: restricts filesize per memmap file (0 no restriction)
    delete_npys: deletes original npys after processing to save space
    batched_npy: assumes first axis in the npy file enumerates samples (otherwise just a single sample per npy file)
    '''
    memmap = None
    start = []#start_idx in current memmap file
    length = []#length of segment
    filenames= []#memmap files
    file_idx=[]#corresponding memmap file for sample
    shape=[]

    target_filename = Path(target_filename)

    for _,npy in tqdm(enumerate(npys),total=len(npys)):
        data_batched = np.load(npy, allow_pickle=True)
        for data in tqdm(data_batched,leave=False) if batched_npy else[data_batched]:
            if(memmap is None or (max_len>0 and start[-1]+length[-1]>max_len)):
                if(max_len>0):
                    filenames.append(target_filename.parent/(target_filename.stem+"_"+str(len(filenames))+".npy"))
                else:
                    filenames.append(target_filename)

                if(memmap is not None):#an existing memmap exceeded max_len
                    shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
                    del memmap
                #create new memmap
                start.append(0)
                length.append(data.shape[0])
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='w+', shape=data.shape)
            else:
                #append to existing memmap
                start.append(start[-1]+length[-1])
                length.append(data.shape[0])
                memmap = np.memmap(filenames[-1], dtype=data.dtype, mode='r+', shape=tuple([start[-1]+length[-1]]+[l for l in data.shape[1:]]))

            #store mapping memmap_id to memmap_file_id
            file_idx.append(len(filenames)-1)
            #insert the actual data
            memmap[start[-1]:start[-1]+length[-1]]=data[:]
            memmap.flush()
        if(delete_npys is True):
            npy.unlink()
    del memmap

    #append final shape if necessary
    if(len(shape)<len(filenames)):
        shape.append([start[-1]+length[-1]]+[l for l in data.shape[1:]])
    #convert everything to relative paths
    filenames= [f.name for f in filenames]
    #save metadata
    np.savez(target_filename.parent/(target_filename.stem+"_meta.npz"),start=start,length=length,shape=shape,file_idx=file_idx,dtype=data.dtype,filenames=filenames)

def reformat_as_memmap(df, target_filename, data_folder=None, annotation=False, max_len=0, delete_npys=True,col_data="data",col_lbl="label", batch_length=0, skip_export_signals=False):
    #note: max_len refers to time steps to keep consistency between labels and signals
    target_filename = Path(target_filename)
    data_folder = Path(data_folder)
    
    npys_data = []
    npys_label = []

    for _,row in df.iterrows():
        npys_data.append(data_folder/row[col_data] if data_folder is not None else row[col_data])
        if(annotation):
            npys_label.append(data_folder/row[col_lbl] if data_folder is not None else row[col_lbl])
    if(not(skip_export_signals)):
        if(batch_length==0):
            npys_to_memmap(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_data, target_filename, max_len=max_len, delete_npys=delete_npys,batch_length=batch_length)
    if(annotation):
        if(batch_length==0):
            npys_to_memmap(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys)
        else:
            npys_to_memmap_batched(npys_label, target_filename.parent/(target_filename.stem+"_label.npy"), max_len=max_len, delete_npys=delete_npys, batch_length=batch_length)

    #replace data(filename) by integer
    df_mapped = df.copy()
    df_mapped[col_data+"_original"]=df_mapped[col_data]
    df_mapped[col_data]=np.arange(len(df_mapped))
    if(annotation):#for consistency also map labels (even though indexing will be via col_data)
        df_mapped[col_lbl+"_original"]=df_mapped[col_lbl]
        df_mapped[col_lbl]=np.arange(len(df_mapped))
        
    df_mapped.to_pickle(target_filename.parent/("df_"+target_filename.stem+".pkl"))
    return df_mapped

class ConcatDatasetTimeseriesDatasetCrops(torch.utils.data.ConcatDataset):
    '''ConcatDataset that handles id mapping correctly (to allow to aggregate predictions)'''
    def __init__(self, datasets):
        super(ConcatDatasetTimeseriesDatasetCrops, self).__init__(datasets)
        idmaps = []
        for dataset_idx,ds in enumerate(self.datasets):
            idmap = ds.get_id_mapping()
            remap_dict = {x:j+(self.cumulative_sizes[dataset_idx-1] if dataset_idx>0 else 0) for j,x in enumerate(np.unique(idmap))}
            idmaps.append(np.array([remap_dict[x] for x in idmap]))
        self.df_idx_mapping = np.concatenate(idmaps)

    def get_id_mapping(self):
        return self.df_idx_mapping
        

class TimeseriesDatasetCrops(torch.utils.data.Dataset):
    """timeseries dataset with partial crops."""

    def __init__(self, df, output_size, chunk_length, min_chunk_length, memmap_filename=None, memmap_label_filename=None, npy_data=None, random_crop=True, data_folder=None, copies=0, col_data="data", col_lbl="label", cols_static=None, stride=None, start_idx=0, annotation=False, transforms=None, sample_items_per_record=1, fs_annotation_over_fs_data=1):
        """
        accepts three kinds of input:
        1) filenames pointing to aligned numpy arrays [timesteps,channels,...] for data and either integer labels or filename pointing to numpy arrays[timesteps,...] e.g. for annotations
        2) memmap_filename to memmap file (same argument that was passed to reformat_as_memmap) for data [concatenated,...] and labels- data column in df corresponds to index in this memmap; memmap_label_filename can normally kept as None (will use the memmap_label file in the same directory in this case)
        3) npy_data [samples,ts,...] (either path or np.array directly- also supporting variable length input) - data column in df corresponds to sampleid

        transforms: list of callables (transformations) or single instance e.g. from torchvision.transforms.Compose (applied in the specified order i.e. leftmost element first)
        
        col_lbl = None: return dummy label 0 (e.g. for unsupervised pretraining)
        cols_static: (optional) list of cols with extra static information
        fs_annotation_over_fs_data over ratio of sampling frequencies
        """
        assert not((memmap_filename is not None) and (npy_data is not None))
        # require integer entries if using memmap or npy
        assert (memmap_filename is None and npy_data is None) or (df[col_data].dtype==np.int64 or df[col_data].dtype==np.int32 or df[col_data].dtype==np.int16)
        # keys (in column data) have to be unique
        assert(len(df[col_data].unique())==len(df))

        self.timeseries_df_data = np.array(df[col_data])
        if(self.timeseries_df_data.dtype not in [np.int16, np.int32, np.int64]):
            assert(memmap_filename is None and npy_data is None) #only for filenames in mode files
            self.timeseries_df_data = np.array(df[col_data].astype(str)).astype(np.string_)

        if(col_lbl is None):# use dummy labels
            self.timeseries_df_label = np.zeros(len(df))
        else: # use actual labels
            if(isinstance(df[col_lbl].iloc[0],list) or isinstance(df[col_lbl].iloc[0],np.ndarray)):#stack arrays/lists for proper batching
                self.timeseries_df_label = np.stack(df[col_lbl])
            else: # single integers/floats
                self.timeseries_df_label = np.array(df[col_lbl])

            if(not(annotation and memmap_filename is not None)):#skip if memmap and annotation        
                if(self.timeseries_df_label.dtype not in [np.int16, np.int32, np.int64, np.float32, np.float64]): #everything else cannot be batched anyway mp.Manager().list(self.timeseries_df_label)
                    assert(annotation and memmap_filename is None and npy_data is None)#only for filenames in mode files
                    self.timeseries_df_label = np.array(df[col_lbl].apply(lambda x:str(x))).astype(np.string_)

        if(cols_static is not None):
            if(len(cols_static)==1):
                self.timeseries_df_static = np.array(df[cols_static[0]])
            else:
                self.timeseries_df_static = np.array(df[cols_static])#.astype(np.float32)
            self.static = True
        else:
            self.static = False

        self.output_size = output_size
        self.data_folder = data_folder
        self.transforms = Compose(transforms) if isinstance(transforms,list) else transforms
        #if(isinstance(self.transforms,list) or isinstance(self.transforms,np.ndarray)):
        #    print("Warning: the use of lists as arguments for transforms is discouraged")
        self.annotation = annotation
        self.col_lbl = col_lbl

        self.mode="files"
        self.fs_annotation_over_fs_data = fs_annotation_over_fs_data

        if(memmap_filename is not None):
            self.memmap_meta_filename = memmap_filename.parent/(memmap_filename.stem+"_meta.npz")
            self.mode="memmap"
            memmap_meta = np.load(self.memmap_meta_filename, allow_pickle=True)
            self.memmap_start = memmap_meta["start"].astype(np.int64)# cast as integers to be on the safe side
            self.memmap_shape = memmap_meta["shape"].astype(np.int64)
            self.memmap_length = memmap_meta["length"].astype(np.int64)
            self.memmap_file_idx = memmap_meta["file_idx"].astype(np.int64)
            self.memmap_dtype = np.dtype(str(memmap_meta["dtype"]))
            self.memmap_filenames = np.array(memmap_meta["filenames"]).astype(np.string_)#save as byte to avoid issue with mp
            if(annotation):
                #by default use the memmap_label.npy in the same directory as the signal memmap file
                memmap_label_filename=memmap_label_filename if memmap_label_filename is not None else self.memmap_meta_filename.parent/("_".join(self.memmap_meta_filename.stem.split("_")[:-1])+"_label.npy")
                self.memmap_meta_filename_label = memmap_filename.parent/(memmap_label_filename.stem+"_meta.npz")
                memmap_meta_label =np.load(self.memmap_meta_filename_label, allow_pickle=True)
                self.memmap_start_label = memmap_meta_label["start"].astype(np.int64)
                self.memmap_shape_label = memmap_meta_label["shape"].astype(np.int64)
                self.memmap_length_label = memmap_meta_label["length"].astype(np.int64)
                self.memmap_file_idx_label = memmap_meta_label["file_idx"].astype(np.int64)
                self.memmap_dtype_label = np.dtype(str(memmap_meta_label["dtype"]))
                self.memmap_filenames_label = np.array(memmap_meta_label["filenames"]).astype(np.string_)
        elif(npy_data is not None):
            self.mode="npy"
            if(isinstance(npy_data,np.ndarray) or isinstance(npy_data,list)):
                self.npy_data = np.array(npy_data)
                assert(annotation is False)
            else:
                self.npy_data = np.load(npy_data, allow_pickle=True)
            if(annotation):
                self.npy_data_label = np.load(npy_data.parent/(npy_data.stem+"_label.npy"), allow_pickle=True)

        self.random_crop = random_crop
        self.sample_items_per_record = sample_items_per_record

        self.df_idx_mapping=[]
        self.start_idx_mapping=[]
        self.end_idx_mapping=[]

        for df_idx,(id,row) in enumerate(df.iterrows()):
            if(self.mode=="files"):
                data_length = row["data_length"]
            elif(self.mode=="memmap"):
                data_length= self.memmap_length[row[col_data]]
            else: #npy
                data_length = len(self.npy_data[row[col_data]])

            if(chunk_length == 0):#do not split
                idx_start = [start_idx]
                idx_end = [data_length]
            else:
                idx_start = list(range(start_idx,data_length,chunk_length if stride is None else stride))
                idx_end = [min(l+chunk_length, data_length) for l in idx_start]

            #remove final chunk(s) if too short
            for i in range(len(idx_start)):
                if(idx_end[i]-idx_start[i]< min_chunk_length):
                    del idx_start[i:]
                    del idx_end[i:]
                    break
            #append to lists
            for _ in range(copies+1):
                for i_s,i_e in zip(idx_start,idx_end):
                    self.df_idx_mapping.append(df_idx)
                    self.start_idx_mapping.append(i_s)
                    self.end_idx_mapping.append(i_e)
        #convert to np.array to avoid mp issues with python lists
        self.df_idx_mapping = np.array(self.df_idx_mapping)
        self.start_idx_mapping = np.array(self.start_idx_mapping)
        self.end_idx_mapping = np.array(self.end_idx_mapping)
            
    def __len__(self):
        return len(self.df_idx_mapping)

    @property
    def is_empty(self):
        return len(self.df_idx_mapping)==0

    def __getitem__(self, idx):
        lst=[]
        for _ in range(self.sample_items_per_record):
            #determine crop idxs
            timesteps= self.get_sample_length(idx)

            if(self.random_crop):#random crop
                if(timesteps==self.output_size):
                    start_idx_rel = 0
                else:
                    start_idx_rel = random.randint(0, timesteps - self.output_size -1)#np.random.randint(0, timesteps - self.output_size)
            else:
                start_idx_rel =  (timesteps - self.output_size)//2
            if(self.sample_items_per_record==1):
                return self._getitem(idx,start_idx_rel)
            else:
                lst.append(self._getitem(idx,start_idx_rel))
        return tuple(lst)

    def _getitem(self, idx,start_idx_rel):
        #low-level function that actually fetches the data
        df_idx = self.df_idx_mapping[idx]
        start_idx = self.start_idx_mapping[idx]
        end_idx = self.end_idx_mapping[idx]
        #determine crop idxs
        timesteps= end_idx - start_idx
        assert(timesteps>=self.output_size)
        start_idx_crop = start_idx + start_idx_rel
        end_idx_crop = start_idx_crop+self.output_size
        if(self.annotation):
            start_idx_crop_label = int(np.round(start_idx_crop*self.fs_annotation_over_fs_data))
            end_idx_crop_label = start_idx_crop_label+int(np.round(self.output_size*self.fs_annotation_over_fs_data))

        #print(idx,start_idx,end_idx,start_idx_crop,end_idx_crop)
        #load the actual data
        if(self.mode=="files"):#from separate files
            data_filename = str(self.timeseries_df_data[df_idx],encoding='utf-8') #todo: fix potential issues here
            if self.data_folder is not None:
                data_filename = self.data_folder/data_filename
            data = np.load(data_filename, allow_pickle=True)[start_idx_crop:end_idx_crop] #data type has to be adjusted when saving to npy

            ID = data_filename.stem

            if(self.annotation is True):
                label_filename = str(self.timeseries_df_label[df_idx],encoding='utf-8')
                if self.data_folder is not None:
                    label_filename = self.data_folder/label_filename
                label = np.load(label_filename, allow_pickle=True)[start_idx_crop_label:end_idx_crop_label] #data type has to be adjusted when saving to npy
            else:
                label = self.timeseries_df_label[df_idx] #input type has to be adjusted in the dataframe
        elif(self.mode=="memmap"): #from one memmap file
            memmap_idx = self.timeseries_df_data[df_idx] #grab the actual index (Note the df to create the ds might be a subset of the original df used to create the memmap)
            memmap_file_idx = self.memmap_file_idx[memmap_idx]
            idx_offset = self.memmap_start[memmap_idx]

            #wi = torch.utils.data.get_worker_info()
            #pid = 0 if wi is None else wi.id#os.getpid()
            #print("idx",idx,"ID",ID,"idx_offset",idx_offset,"start_idx_crop",start_idx_crop,"df_idx", self.df_idx_mapping[idx],"pid",pid)
            mem_filename = str(self.memmap_filenames[memmap_file_idx],encoding='utf-8')
            mem_file = np.memmap(self.memmap_meta_filename.parent/mem_filename, self.memmap_dtype, mode='r', shape=tuple(self.memmap_shape[memmap_file_idx]))
            data = np.copy(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            del mem_file
            #print(mem_file[idx_offset + start_idx_crop: idx_offset + end_idx_crop])
            if(self.annotation):
                memmap_file_idx_label = self.memmap_file_idx_label[memmap_idx]
                idx_offset_label = self.memmap_start_label[memmap_idx]

                mem_filename_label = str(self.memmap_filenames_label[memmap_file_idx_label],encoding='utf-8')
                mem_file_label = np.memmap(self.memmap_meta_filename_label.parent/mem_filename_label, self.memmap_dtype_label, mode='r', shape=tuple(self.memmap_shape_label[memmap_file_idx]))
                
                label = np.copy(mem_file_label[idx_offset_label + start_idx_crop_label: idx_offset_label + end_idx_crop_label])
                del mem_file_label
            else:
                label = self.timeseries_df_label[df_idx]
        else:#single npy array
            ID = self.timeseries_df_data[df_idx]

            data = self.npy_data[ID][start_idx_crop:end_idx_crop]

            if(self.annotation):
                label = self.npy_data_label[ID][start_idx_crop:end_idx_crop]
            else:
                label = self.timeseries_df_label[df_idx]

        sample = (data, label, self.timeseries_df_static[df_idx] if self.static else None)
        
        # consistency check: make sure that data and annotation lengths match (check here because transforms might change the shape of the annotation)
        assert(self.annotation is False or len(sample[1])==int(np.round(self.fs_annotation_over_fs_data*len(sample[0]))))

        if(isinstance(self.transforms,list)):#transforms passed as list
            for t in self.transforms:
                sample = t(sample)
        elif(self.transforms is not None):#single transform e.g. from torchvision.transforms.Compose
            sample = self.transforms(sample)
        
        if(self.static is True):
            return tsdata_static(sample[0],sample[1], sample[2])
        else:
            return tsdata(sample[0], sample[1])
        

    def get_sampling_weights(self, class_weight_dict,length_weighting=False, timeseries_df_group_by_col=None):
        '''
        class_weight_dict: dictionary of class weights
        length_weighting: weigh samples by length
        timeseries_df_group_by_col: column of the pandas df used to create the object'''
        assert(self.annotation is False)
        assert(length_weighting is False or timeseries_df_group_by_col is None)
        weights = np.zeros(len(self.df_idx_mapping),dtype=np.float32)
        length_per_class = {}
        length_per_group = {}
        for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
            label = self.timeseries_df_label[i]
            weight = class_weight_dict[label]
            if(length_weighting):
                if label in length_per_class.keys():
                    length_per_class[label] += e-s
                else:
                    length_per_class[label] = e-s
            if(timeseries_df_group_by_col is not None):
                group = timeseries_df_group_by_col[i]
                if group in length_per_group.keys():
                    length_per_group[group] += e-s
                else:
                    length_per_group[group] = e-s
            weights[iw] = weight

        if(length_weighting):#need second pass to properly take into account the total length per class
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                label = self.timeseries_df_label[i]
                weights[iw]= (e-s)/length_per_class[label]*weights[iw]
        if(timeseries_df_group_by_col is not None):
            for iw,(i,s,e) in enumerate(zip(self.df_idx_mapping,self.start_idx_mapping,self.end_idx_mapping)):
                group = timeseries_df_group_by_col[i]
                weights[iw]= (e-s)/length_per_group[group]*weights[iw]

        weights = weights/np.min(weights)#normalize smallest weight to 1
        return weights

    def get_id_mapping(self):
        return self.df_idx_mapping

    def get_sample_id(self,idx):
        return self.df_idx_mapping[idx]

    def get_sample_length(self,idx):
        return self.end_idx_mapping[idx]-self.start_idx_mapping[idx]

    def get_sample_start(self,idx):
        return self.start_idx_mapping[idx]


class RandomCrop(object):
    """Crop randomly the image in a sample.
    """

    def __init__(self, output_size,annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static = sample

        timesteps= len(data)
        assert(timesteps>=self.output_size)
        if(timesteps==self.output_size):
            start=0
        else:
            start = random.randint(0, timesteps - self.output_size-1) #np.random.randint(0, timesteps - self.output_size)

        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return data, label, static


class CenterCrop(object):
    """Center crop the image in a sample.
    """

    def __init__(self, output_size, annotation=False):
        self.output_size = output_size
        self.annotation = annotation

    def __call__(self, sample):
        data, label, static = sample

        timesteps= len(data)
        start = (timesteps - self.output_size)//2

        data = data[start: start + self.output_size]
        if(self.annotation):
            label = label[start: start + self.output_size]

        return data, label, static


class GaussianNoise(object):
    """Add gaussian noise to sample.
    """

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, sample):
        if self.scale ==0:
            return sample
        else:
            data, label, static = sample
            data = data + np.reshape(np.array([random.gauss(0,self.scale) for _ in range(np.prod(data.shape))]),data.shape)#np.random.normal(scale=self.scale,size=data.shape).astype(np.float32)
            return data, label, static


class Rescale(object):
    """Rescale by factor.
    """

    def __init__(self, scale=0.5,interpolation_order=3):
        self.scale = scale
        self.interpolation_order = interpolation_order

    def __call__(self, sample):
        if self.scale ==1:
            return sample
        else:
            data, label, static = sample
            timesteps_new = int(self.scale * len(data))
            data = transform.resize(data,(timesteps_new,data.shape[1]),order=self.interpolation_order).astype(np.float32)
            return data, label, static


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, transpose_data=True, transpose_label=False):
        #swap channel and time axis for direct application of pytorch's convs
        self.transpose_data=transpose_data
        self.transpose_label=transpose_label

    def __call__(self, sample):

        def _to_tensor(data,transpose=False):
            if(isinstance(data,np.ndarray)):
                if(transpose):#seq,[x,y,]ch -> [x,y,]ch, seq
                    return torch.from_numpy(np.moveaxis(data,0,-1))
                else:
                    return torch.from_numpy(data)
            else:#default_collate will take care of it
                return data

        data, label, static = sample
        if not isinstance(data,tuple):
            data = _to_tensor(data,self.transpose_data)
        else:
            data = tuple(_to_tensor(x,self.transpose_data) for x in data)

        if not isinstance(label,tuple):
            label = _to_tensor(label,self.transpose_label)
        else:
            label = tuple(_to_tensor(x,self.transpose_label) for x in label)
        #default collation for static  
        return data, label, static #returning as a tuple (potentially of lists)


class Normalize(object):
    """Normalize using given stats.
    """
    def __init__(self, stats_mean, stats_std, input=True, channels=[]):
        self.stats_mean=stats_mean.astype(np.float32) if stats_mean is not None else None
        self.stats_std=stats_std.astype(np.float32)+1e-8 if stats_std is not None else None
        self.input = input
        if(len(channels)>0):
            for i in range(len(stats_mean)):
                if(not(i in channels)):
                    self.stats_mean[:,i]=0
                    self.stats_std[:,i]=1

    def __call__(self, sample):
        datax, labelx, static = sample
        data = datax if self.input else labelx
        #assuming channel last
        if(self.stats_mean is not None):
            data = data - self.stats_mean
        if(self.stats_std is not None):
            data = data/self.stats_std

        if(self.input):
            return (data, labelx, static)
        else:
            return (datax, data, static)


class NormalizeBatch(object):
    """Normalize using batch statistics.
    axis: tuple of integers of axis numbers to be normalized over (by default everything but the last)
    """
    def __init__(self, input=True, channels=[],axis=None):
        self.channels = channels
        self.channels_keep = None
        self.input = input
        self.axis = axis

    def __call__(self, sample):
        datax, labelx, static = sample
        data = datax if self.input else labelx
        #assuming channel last
        #batch_mean = np.mean(data,axis=tuple(range(0,len(data)-1)))
        #batch_std = np.std(data,axis=tuple(range(0,len(data)-1)))+1e-8
        batch_mean = np.mean(data,axis=self.axis if self.axis is not None else tuple(range(0,len(data.shape)-1)))
        batch_std = np.std(data,axis=self.axis if self.axis is not None else tuple(range(0,len(data.shape)-1)))+1e-8

        if(len(self.channels)>0):
            if(self.channels_keep is None):
                self.channels_keep = np.setdiff(range(data.shape[-1]),self.channels)

            batch_mean[self.channels_keep]=0
            batch_std[self.channels_keep]=1

        data = (data - batch_mean)/batch_std

        if(self.input):
            return (data, labelx, static)
        else:
            return (datax, data, static)


class ButterFilter(object):
    """Apply filter
    """

    def __init__(self, lowcut=50, highcut=50, fs=100, order=5, btype='band', forwardbackward=True, input=True):
        self.filter = butter_filter(lowcut,highcut,fs,order,btype)
        self.input = input
        self.forwardbackward = forwardbackward

    def __call__(self, sample):
        datax, labelx, static = sample
        data = datax if self.input else labelx

        if(self.forwardbackward):
            data = sosfiltfilt(self.filter, data, axis=0)
        else:
            data = sosfilt(self.filter, data, axis=0)

        if(self.input):
            return (data, labelx, static)
        else:
            return (datax, data, static)


class ChannelFilter(object):
    """Select certain channels.
    axis: axis index of the channel axis
    """

    def __init__(self, channels=[0], axis=-1, input=True):
        self.channels = channels
        self.input = input
        self.axis = axis

    def __call__(self, sample):
        data, label, static = sample
        if(self.input):
            return (np.take(data,self.channels,axis=self.axis), label, static) #(data[...,self.channels], label, static)
        else:
            return (data, np.take(label,self.channels,axis=self.axis), static)


class Transform(object):
    """Transforms data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)
    """

    def __init__(self, func, input=False):
        self.func = func
        self.input = input

    def __call__(self, sample):
        data, label, static = sample
        if(self.input):
            return (self.func(data), label, static)
        else:
            return (data, self.func(label), static)

class StaticTransform(object):
    """Transforms static data using a given function i.e. data_new = func(data) for input is True else label_new = func(label)
    """
    def __init__(self, func):
        self.func = func
        
    def __call__(self, sample):
        data, label, static = sample
        return (data, label, self.func(static))

class TupleTransform(object):
    """Transforms data using a given function (operating on both data and label and return a tuple) i.e. data_new, label_new = func(data_old, label_old)
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, sample):
        data,label,static = sample
        return  self.func(data,label,static)

class SequenceToSampleLabelTransform(object):
    """Transforms sequence-level to sample-level labels
    majority vote: pick the most frequent label as segment label (i.e. suitable for single-label classification)
    num_classes: number of output classes
    binary: binary instead of floating point outputs (where the latter represent fractions)
    epoch_length: split the original sequence in ts//epoch_length fragments
    """

    def __init__(self, majority_vote=False, num_classes=2, binary=False,epoch_length=0):
        self.majority_vote = majority_vote
        self.num_classes = num_classes
        self.binary = binary
        self.epoch_length = epoch_length

    def __call__(self, sample):
        data, label, static = sample
        
        epoch_length = self.epoch_length if self.epoch_length>0 else len(label)
        if(len(label.shape)==1):#each time step is single-labeled
            label = np.eye(self.num_classes)[label] #now label has shape ts,num_classes
        cnts = np.sum(label.reshape((-1,epoch_length,label.shape[-1])),axis=1)#segments,classes
        if(self.majority_vote):
            label = np.argmax(cnts,axis=-1)#segments
        else:
            if(self.binary):
                label=(cnts>0).astype(np.float32)
            else:
                label = (cnts/epoch_length).astype(np.float32)
        if(self.epoch_length>0):
            return (data, label, static)
        else:#just one segment
            return (data, label[0], static)

def aggregate_predictions(preds,targs=None,idmap=None,aggregate_fn = np.mean,verbose=False):
    '''
    aggregates potentially multiple predictions per sample (can also pass targs for convenience)
    idmap: idmap as returned by TimeSeriesCropsDataset's get_id_mapping
    preds: ordered predictions as returned by learn.get_preds()
    aggregate_fn: function that is used to aggregate multiple predictions per sample (most commonly np.amax or np.mean)
    '''
    if(idmap is not None and len(idmap)!=len(np.unique(idmap))):
        if(verbose):
            print("aggregating predictions...")
        preds_aggregated = []
        targs_aggregated = []
        for i in np.unique(idmap):
            preds_local = preds[np.where(idmap==i)[0]]
            preds_aggregated.append(aggregate_fn(preds_local,axis=0))
            if targs is not None:
                targs_local = targs[np.where(idmap==i)[0]]
                assert(np.all(targs_local==targs_local[0])) #all labels have to agree
                targs_aggregated.append(targs_local[0])
        if(targs is None):
            return np.array(preds_aggregated)
        else:
            return np.array(preds_aggregated),np.array(targs_aggregated)
    else:
        if(targs is None):
            return preds
        else:
            return preds,targs
