__all__ = ['empirical_bootstrap']

import numpy as np
from sklearn.utils import resample
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm

def _eval(ids, input_tuple, score_fn, input_tuple2=None,score_fn_kwargs={}):
    return score_fn(*[t[ids] for t in input_tuple],**score_fn_kwargs) if input_tuple2 is None else score_fn(*[t[ids] for t in input_tuple],**score_fn_kwargs)-score_fn(*[t[ids] for t in input_tuple2],**score_fn_kwargs)

def empirical_bootstrap(input_tuple, score_fn, ids=None, n_iterations=1000, alpha=0.95, score_fn_kwargs={},threads=None, input_tuple2=None, ignore_nans=False, chunksize=50):
    '''
        performs empirical bootstrap https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
        
        input_tuple: tuple of inputs for the score function typically something like (labels,predictions)
        score_function: scoring function that takes the individual entries of input tuple as argument e.g. f1_score
        id: list of previously sampled ids (if None new ids will be sampled)
        n_iterations: number of bootstrap iterations
        alpha: alpha-level for the confidence intervals
        score_fn_kwargs: additional (static) kwargs to be passed to the score_fn
        threads: number of threads (None uses os.cpu_count()); 0 no multithreading
        input_tuple2: if not None this is a second input of the same shape as input_tuple- in that case the function bootstraps the score difference between both inputs (this is just a convenience function- the same could be achieved by passing a tuple of the form (label,preds1,preds2) and computing the difference in the score_function itself)
        ignore_nans: ignore nans (e.g. no positives during during AUC evaluation) for score evaluation
        chunksize: process in chunks of size chunksize
    '''
    
    if(not(isinstance(input_tuple,tuple))):
        input_tuple = (input_tuple,)
    if(input_tuple2 is not None and not(isinstance(input_tuple2,tuple))):
        input_tuple2 = (input_tuple2,)
        
    score_point = score_fn(*input_tuple,**score_fn_kwargs) if input_tuple2 is None else score_fn(*input_tuple,**score_fn_kwargs)-score_fn(*input_tuple2,**score_fn_kwargs)

    if(n_iterations==0):
        return score_point,np.zeros(score_point.shape),np.zeros(score_point.shape),[]
    
    if(ids is None):
        ids = []
        for _ in range(n_iterations):
            ids.append(resample(range(len(input_tuple[0])), n_samples=len(input_tuple[0])))
        ids = np.array(ids)

    fn = partial(_eval,input_tuple=input_tuple,score_fn=score_fn,input_tuple2=input_tuple2,score_fn_kwargs=score_fn_kwargs)

    if(threads is not None and threads==0):
        results= np.array(fn(ids)).astype(np.float32)#shape: bootstrap_iterations, number_of_evaluation_metrics
    else:
        results=[]       
        for istart in tqdm(np.arange(0,n_iterations,chunksize)):
            iend = min(n_iterations,istart+chunksize)
            pool = Pool(threads)
            results.append(np.array(pool.map(fn, ids[istart:iend])).astype(np.float32))
            pool.close()
            pool.join()
    
        results = np.concatenate(results,axis=0)
    
    percentile_fn = np.nanpercentile if ignore_nans else np.percentile        
    score_diff = np.array(results)- score_point
    score_low = score_point + percentile_fn(score_diff, ((1.0-alpha)/2.0) * 100,axis=0)
    score_high = score_point + percentile_fn(score_diff, (alpha+((1.0-alpha)/2.0)) * 100,axis=0)

    if(ignore_nans):#in this case return the number of nans in each score rather than the sampled ids (which could be different when evaluating several metrics at once)
        return score_point, score_low, score_high, np.sum(np.isnan(score_diff),axis=0)
    else:
        return score_point, score_low, score_high, ids

