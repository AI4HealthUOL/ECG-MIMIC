__all__ = ['auc_prrc_uninterpolated', 'multiclass_roc_curve', 'single_eval_prrc', 'eval_prrc', 'eval_prrc_parallel',
           'eval_scores', 'eval_scores_bootstrap']

# Cell
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.utils import resample

from tqdm import tqdm

# Cell
def auc_prrc_uninterpolated(recall,precision):
    '''uninterpolated auc as used by sklearn https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/ranking.py see also the discussion at https://github.com/scikit-learn/scikit-learn/pull/9583'''
    #print(-np.sum(np.diff(recall) * np.array(precision)[:-1]),auc(recall,precision))
    return -np.sum(np.diff(recall) * np.array(precision)[:-1])

# Cell
#label-centric metrics
def multiclass_roc_curve(y_true, y_pred, classes=None, precision_recall=False):
    '''Compute ROC curve and ROC area for each class "0"..."n_classes - 1" (or classnames passed via classes), "micro", "macro"
    returns fpr,tpr,roc (dictionaries) for ROC
    returns recall,precision,average_precision for precision_recall
    '''

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes=len(y_pred[0])
    if(classes is None):
        classes = [str(i) for i in range(n_classes)]

    for i,c in enumerate(classes):
        if(precision_recall):
            tpr[c], fpr[c], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
            roc_auc[c] = auc_prrc_uninterpolated(fpr[c], tpr[c])
        else:
            fpr[c], tpr[c], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[c] = auc(fpr[c], tpr[c])

    # Compute micro-average curve and area
    if(precision_recall):
        tpr["micro"], fpr["micro"], _ = precision_recall_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc_prrc_uninterpolated(fpr["micro"], tpr["micro"])
    else:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average curve and area (linear interpolation is incorrect for PRRC- therefore just for ROC)
    if(precision_recall is False):
        # 1. First aggregate all unique x values (false positive rates for ROC)
        all_fpr = np.unique(np.concatenate([fpr[c] for c in classes]))

        # 2. Then interpolate all curves at this points
        mean_tpr=None
        for c in classes:
            f = interp1d(fpr[c], tpr[c])
            if(mean_tpr is None):
                mean_tpr = f(all_fpr)
            else:
                mean_tpr += f(all_fpr)

        # 3. Finally average it and compute area
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        #macro2 differs slightly from macro due to interpolation effects
        #roc_auc["macro2"] = auc(fpr["macro"], tpr["macro"])

    #calculate macro auc directly by summing
    roc_auc_macro = 0
    for c in classes:
        roc_auc_macro += roc_auc[c]
    roc_auc["macro"]=roc_auc_macro/n_classes

    #calculate macro auc directly by summing
    roc_auc_macro = 0
    macro_auc_nans = 0 #due to an insufficient amount of pos/neg labels
    for c in classes:
        if(np.isnan(roc_auc[c])):#conservative choice: replace auc by 0.5 if it could not be calculated
            roc_auc_macro += 0.5
            macro_auc_nans += 1
        else:
            roc_auc_macro += roc_auc[c]
    roc_auc["macro"]=roc_auc_macro/n_classes
    roc_auc["macro_nans"] = macro_auc_nans

    return fpr, tpr, roc_auc

# Cell
def single_eval_prrc(y_true,y_pred,threshold):
    '''evaluate instance-wise scores for a single sample and a single threshold'''
    y_pred_bin = (y_pred >= threshold)
    TP = np.sum(np.logical_and(y_true == y_pred_bin,y_true>0))
    count = np.sum(y_pred_bin)#TP+FP

    # Find precision: TP / (TP + FP)
    precision = TP / count if count > 0 else np.nan
    # Find recall/TPR/sensitivity: TP / (TP + FN)
    recall = TP/np.sum(y_true>0)
    # Find FPR/specificity: FP/ (FP + TN)=FP/N
    FP = np.sum(np.logical_and(y_true != y_pred_bin,y_pred_bin>0))
    specificity = FP/ np.sum(y_true==0)
    return precision, recall, specificity

# Cell
def eval_prrc(y_true,y_pred,threshold):
    '''eval instance-wise scores across all samples for a single threshold'''
    # Initialize Variables
    PR = 0.0
    RC = 0.0
    SP = 0.0

    counts_above_threshold = 0

    for i in range(len(y_true)):
        pr,rc,sp = single_eval_prrc(y_true[i],y_pred[i],threshold)
        if pr is not np.nan:
            PR += pr
            counts_above_threshold += 1
        RC += rc
        SP += sp

    recall = RC/len(y_true)
    specificity = SP/len(y_true)

    if counts_above_threshold > 0:
        precision = PR/counts_above_threshold
    else:
        precision = np.nan
        if(threshold<1.0):
            print("No prediction is made above the %.2f threshold\n" % threshold)
    return precision, recall, specificity, counts_above_threshold/len(y_true)

# Cell
def eval_prrc_parallel(y_true,y_pred,thresholds):

    y_pred_bin = np.repeat(y_pred[None, :, :], len(thresholds), axis=0)>=thresholds[:,None,None]#thresholds, samples, classes
    TP = np.sum(np.logical_and( y_true == True, y_pred_bin== True),axis=2)#threshold, samples

    with np.errstate(divide='ignore', invalid='ignore'):
        den = np.sum(y_pred_bin,axis=2)>0
        precision = TP/np.sum(y_pred_bin,axis=2)
        precision[den==0] = np.nan

    recall = TP/np.sum(y_true==True, axis=1)#threshold,samples/samples=threshold,samples

    FP = np.sum(np.logical_and((y_true ==False),(y_pred_bin==True)),axis=2)
    specificity = FP/np.sum(y_true==False, axis=1)

    with warnings.catch_warnings(): #for nan slices
        warnings.simplefilter("ignore", category=RuntimeWarning)
        av_precision = np.nanmean(precision,axis=1)

    av_recall = np.mean(recall,axis=1)
    av_specificity = np.mean(specificity,axis=1)
    av_coverage = np.mean(den,axis=1)

    return av_precision, av_recall, av_specificity, av_coverage


# Cell
def eval_scores(y_true,y_pred,classes=None,num_thresholds=100,full_output=False,parallel=True):
    '''returns a dictionary of performance metrics:
    sample centric c.f. https://github.com/ashleyzhou972/CAFA_assessment_tool/blob/master/precrec/precRec.py
    https://www.nature.com/articles/nmeth.2340 vs https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3694662/ and https://arxiv.org/pdf/1601.00891
    * Fmax, sample AUC, sample Average Precision (as in sklearn)

    label-centric: micro,macro,individual AUC and Average Precision
    '''
    results = {}

    # thresholds = np.arange(0.00, 1.01, 1./num_thresholds, float)
    # if(parallel is False):
    #     PR = np.zeros(len(thresholds))
    #     RC = np.zeros(len(thresholds))
    #     SP = np.zeros(len(thresholds))
    #     COV = np.zeros(len(thresholds))

    #     for i,t in enumerate(thresholds):
    #         PR[i],RC[i],SP[i],COV[i] = eval_prrc(y_true,y_pred,t)
    #     F =  (2*PR*RC)/(PR+RC)
    # else:
    #     PR,RC,SP,COV = eval_prrc_parallel(y_true,y_pred,thresholds)
    #     F = (2*PR*RC)/(PR+RC)

    # if(full_output is True):
    #     results["PR"] = PR
    #     results["RC"] = RC
    #     results["SP"] = SP
    #     results["F"] = F
    #     results["COV"] = COV

    # if np.isnan(F).sum() == len(F):
    #     results["Fmax"] = 0
    #     results["precision_at_Fmax"] = 0
    #     results["recall_at_Fmax"] = 0
    #     results["threshold_at_Fmax"] = 0
    #     results["coverage_at_Fmax"]= 0
    # else:
    #     imax = np.nanargmax(F)
    #     results["Fmax"] = F[imax]
    #     results["precision_at_Fmax"] = PR[imax]
    #     results["recall_at_Fmax"] = RC[imax]
    #     results["threshold_at_Fmax"] = thresholds[imax]
    #     results["coverage_at_Fmax"]=COV[imax]

    # results["sample_AUC"]=auc(1-SP,RC)
    # #https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/metrics/ranking.py set final PR value to 1
    # PR[-1]=1
    # results["sample_APR"]=auc_prrc_uninterpolated(RC,PR)#skip last point with undefined precision
    ###########################################################
    #label-centric
    #"micro","macro",i=0...n_classes-1
    fpr, tpr, roc_auc = multiclass_roc_curve(y_true, y_pred,classes=classes,precision_recall=False)
    if(full_output is True):
        results["fpr"]=fpr
        results["tpr"]=tpr
    results["label_AUC"]=roc_auc

    # rc, pr, prrc_auc = multiclass_roc_curve(y_true, y_pred,classes=classes,precision_recall=True)
    # if(full_output is True):
    #     results["pr"]=pr
    #     results["rc"]=rc
    # results["label_APR"]=prrc_auc

    return results

# Cell
def eval_scores_bootstrap(y_true, y_pred,classes=None, n_iterations = 10000, alpha=0.95):
    #https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf empirical bootstrap rather than bootstrap percentiles
    Fmax_diff = []
    sample_AUC_diff = []
    sample_APR_diff = []
    label_AUC_diff = []
    label_APR_diff = []
    label_AUC_keys = None

    #point estimate
    res_point = eval_scores(y_true,y_pred,classes=classes)
    Fmax_point = res_point["Fmax"]
    sample_AUC_point = res_point["sample_AUC"]
    sample_APR_point = res_point["sample_APR"]
    label_AUC_point = np.array(list(res_point["label_AUC"].values()))
    label_APR_point = np.array(list(res_point["label_APR"].values()))

    #bootstrap
    for i in tqdm(range(n_iterations)):
        ids = resample(range(len(y_true)), n_samples=len(y_true))
        res = eval_scores(y_true[ids],y_pred[ids],classes=classes)
        Fmax_diff.append(res["Fmax"]-Fmax_point)
        sample_AUC_diff.append(res["sample_AUC"]-sample_AUC_point)
        sample_APR_diff.append(res["sample_APR"]-sample_APR_point)
        label_AUC_keys = list(res["label_AUC"].keys())
        label_AUC_diff.append(np.array(list(res["label_AUC"].values()))-label_AUC_point)
        label_APR_diff.append(np.array(list(res["label_APR"].values()))-label_APR_point)

    p = ((1.0-alpha)/2.0) * 100
    Fmax_low = Fmax_point + np.percentile(Fmax_diff, p)
    sample_AUC_low = sample_AUC_point + np.percentile(sample_AUC_diff, p)
    sample_APR_low = sample_APR_point + np.percentile(sample_APR_diff, p)
    label_AUC_low = label_AUC_point + np.percentile(label_AUC_diff,p,axis=0)
    label_APR_low = label_APR_point + np.percentile(label_APR_diff,p,axis=0)
    p = (alpha+((1.0-alpha)/2.0)) * 100
    Fmax_high = Fmax_point + np.percentile(Fmax_diff, p)
    sample_AUC_high = sample_AUC_point + np.percentile(sample_AUC_diff, p)
    sample_APR_high = sample_APR_point + np.percentile(sample_APR_diff, p)
    label_AUC_high = label_AUC_point + np.percentile(label_AUC_diff,p,axis=0)
    label_APR_high = label_APR_point + np.percentile(label_APR_diff,p,axis=0)

    return {"Fmax":[Fmax_low,Fmax_point,Fmax_high], "sample_AUC":[sample_AUC_low,sample_AUC_point,sample_AUC_high], "sample_APR":[sample_APR_low,sample_APR_point,sample_APR_high], "label_AUC":{k:[v1,v2,v3] for k,v1,v2,v3 in zip(label_AUC_keys,label_AUC_low,label_AUC_point,label_AUC_high)}, "label_APR":{k:[v1,v2,v3] for k,v1,v2,v3 in zip(label_AUC_keys,label_APR_low,label_APR_point,label_APR_high)}}