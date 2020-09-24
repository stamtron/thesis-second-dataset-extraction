import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
import numpy as np
import pandas as pd

# we assume that y contains a tuple of y_pred and targets
def nsea_compute_thresholds(y_true, y_pred, classes):
#     y_pred = numpy.asarray(y[0])
#     y_true = numpy.asarray(y[1])
    precisions = dict()
    recalls = dict()
    Thresholds = dict()
    for i in range(5):
        precisions[i], recalls[i], Thresholds[i] = precision_recall_curve(y_true[:, i], y_pred[:, i])

    
    
    result = {}
    ###############
    ###############  FIX THE UGLY STAFF BELOW
    ###############
    opt_id = []
    for i,event_type in enumerate(classes): 
        re = recalls[i]
        pre = precisions[i]
        dist = [ np.sqrt((1-re)**2 + (1-pre)**2) for re, pre in zip(re, pre) ]
        opt_id.append(dist.index(min(dist)))
        t = Thresholds[i]
        opt_thres = t[opt_id[i]]
        result[event_type] = opt_thres
    return result

def new_compute_metrics(y_true, y_pred, thresholds, classes):
    #y_pred = y_pred.numpy()
    #y_true = y_true.numpy()
    th = np.array([thresholds[key] for key in thresholds])
    
    ## Apply digitisation on the outputs
    for idx, event_type in enumerate(classes):
        y_pred[:,idx] = np.where(y_pred[:,idx] >= thresholds[event_type], 1, 0)
    acc = []
    for idx, event in enumerate(classes):
        acc.append(accuracy_score(y_true[:,idx], y_pred[:, idx]))
    acc = np.array(acc)
    agg_acc = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    
    cm=multilabel_confusion_matrix(y_true, y_pred)
    cmm=cm.reshape(-1,4)
    
    res_labels=pd.DataFrame({'Event': classes, 'Threshold': th, 'Exact Matching Score': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
    
    res_labels = pd.concat([res_labels, pd.DataFrame(cmm, columns=['tn', 'fp', 'fn', 'tp'])], axis=1)
    
    agg_precision, agg_recall, agg_f1, agg_support = precision_recall_fscore_support(y_true, y_pred, average='micro')
    agg=pd.DataFrame({'Event': ['Aggregate'], 'Threshold': [np.nan], 'Exact Matching Score': agg_acc, 'Precision': agg_precision, 'Recall': agg_recall, 'F1-Score': agg_f1})
    
    
    agg = pd.concat([agg, pd.DataFrame(data=[np.nan]*4, index=['tn', 'fp', 'fn', 'tp']).T], axis=1)

    res=pd.concat([res_labels, agg]).reset_index(drop=True)
    
    return res


def write_dictionary_to_csv(csv_file, res):
    with open(csv_file, 'w') as csvfile:
        for key in res:
            csvfile.write(key)
            for metric in res[key]:
                csvfile.write(',')
                csvfile.write(metric)
                csvfile.write(',')
                csvfile.write(str(res[key][metric]))
            csvfile.write('\n')
            
            
def truncate_decimals(arr, places=3):
    return np.floor(arr*(10**places))/(10**places)