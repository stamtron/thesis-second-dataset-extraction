import os
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from itertools import cycle
import pandas as pd

plt.rcParams['figure.figsize'] = (20,8)
font = {'family' : 'DejaVu Sans', 'weight' : 'normal', 'size'  : 22}
plt.rc('font', **font)


def predictions_csv(y_pr, y_tr, thresholds, classes):
    df_labels = pd.DataFrame(y_tr, columns=['exp','bur','fj','ande','fs'])
    df_labels['label'] = 'exp'
    for i in range(len(df_labels)):
        if df_labels.bur[i] == 1.0:
            df_labels.label[i] = 'bur'
        if df_labels.ande[i] == 1.0:
            df_labels.label[i] = 'anode'
        if df_labels.fj[i] == 1.0:
            df_labels.label[i] = 'fj'
        if df_labels.fs[i] == 1.0:
            df_labels.label[i] = 'fs'
    df_conf = pd.DataFrame(y_pr, columns=['exp_conf','bur_conf','fj_conf','ande_conf','fs_conf'])
    result = pd.concat([df_labels, df_conf], axis=1)
    th = np.array([thresholds[key] for key in thresholds])
    y_final_pred = np.zeros_like(y_pr)
    ## Apply digitisation on the outputs
    for idx, event_type in enumerate(classes):
        y_final_pred[:,idx] = np.where(y_pr[:,idx] >= thresholds[event_type], 1, 0)
    df_pred_final = pd.DataFrame(y_final_pred, columns=['exp_pred','bur_pred','fj_pred','ande_pred','fs_pred'])
    result = pd.concat([result, df_pred_final], axis=1)
    return result

def conf_and_true_csv(y_pr, y_tr, classes):
    df_labels = pd.DataFrame(y_tr, columns=['exp','bur','fj','ande','fs'])
    df_conf = pd.DataFrame(y_pr, columns=['exp_conf','bur_conf','fj_conf','ande_conf','fs_conf'])
    result = pd.concat([df_labels, df_conf], axis=1)
    return result


def case_defined_smart_thresholding(y_pr):
    for i in range(len(y_pr)):
        if y_pr[i][0] >= y_pr[i][1]:
            y_pr[i][0] = 1.0
            y_pr[i][1] = 0.0
        else:
            y_pr[i][0] = 0.0
            y_pr[i][1] = 1.0

        if y_pr[i][1] == 1.0:
            y_pr[i][2] = 0.0
            y_pr[i][3] = 0.0
            y_pr[i][4] = 0.0

        if y_pr[i][0] == 1.0:
            if y_pr[i][2] > y_pr[i][3]:
                y_pr[i][2] = 1.0
                y_pr[i][3] = 0.0
            else:
                y_pr[i][2] = 1.0
                y_pr[i][3] = 0.0

        if y_pr[i][4] >= 0.85:
            y_pr[i][4] = 1.0
            y_pr[i][2] = 0.0
            y_pr[i][3] = 0.0
        else:
            y_pr[i][4] = 0.0
    
    return y_pr
    
    
def metrics_after_thresholding(y_true, y_pred, classes):
    acc = []
    for idx, event in enumerate(classes):
        acc.append(accuracy_score(y_true[:,idx], y_pred[:, idx]))
    acc = np.array(acc) 
    agg_acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    cm=multilabel_confusion_matrix(y_true, y_pred)
    cmm=cm.reshape(-1,4)
    
    res_labels=pd.DataFrame({'Event': classes, 'Exact Matching Score': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
    
    res_labels = pd.concat([res_labels, pd.DataFrame(cmm, columns=['tn', 'fp', 'fn', 'tp'])], axis=1)
    
    agg_precision, agg_recall, agg_f1, agg_support = precision_recall_fscore_support(y_true, y_pred, average='samples')
    agg=pd.DataFrame({'Event': ['Aggregate'], 'Exact Matching Score': agg_acc, 'Precision': agg_precision, 'Recall': agg_recall, 'F1-Score': agg_f1})
       
    agg = pd.concat([agg, pd.DataFrame(data=[np.nan]*4, index=['tn', 'fp', 'fn', 'tp']).T], axis=1)

    res=pd.concat([res_labels, agg]).reset_index(drop=True)
    
    return res

# we assume that y contains a tuple of y_pred and targets
def pr_curves_compute_thresholds(y_true, y_pred, classes):
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


def roc_curves_compute_thresholds(y_true, y_pred, classes):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    Thresholds = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], Thresholds[i] = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    result = {}
    opt_id = []
    for i,event_type in enumerate(classes):
        fprr = fpr[i]
        tprr = tpr[i]
        dist = [ np.sqrt((0-f)**2 + (1-t)**2) for f, t in zip(fprr, tprr) ]
        opt_id.append(dist.index(min(dist)))
        t = Thresholds[i]
        opt_thres = t[opt_id[i]]
        result[event_type] = opt_thres
    return result

def plot_micro_macro_roc_curves(y_tr, y_pr, classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_tr[:, i], y_pr[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_tr.ravel(), y_pr.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))

# Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw = 2
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Micro and Micro as well')
    plt.legend(loc="lower right")
    plt.show()

def compute_label_metrics(y_true, y_pred, threshold, classes):
    for idx, event_type in enumerate(classes):
        y_pred[:,idx] = np.where(y_pred[:,idx] >= threshold, 1, 0)
    acc = []
    for idx, event in enumerate(classes):
        acc.append(accuracy_score(y_true[:,idx], y_pred[:, idx]))
    acc = np.array(acc) 
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    return acc, f1


def compute_label_metrics_validation(y_true, y_pred, threshold, classes):
    for idx, event_type in enumerate(classes):
        y_pred[:,idx] = np.where(y_pred[:,idx] >= threshold, 1, 0)
    acc = []
    for idx, event in enumerate(classes):
        acc.append(accuracy_score(y_true[:,idx], y_pred[:, idx]))
    acc = np.array(acc) 
    agg_acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    cm=multilabel_confusion_matrix(y_true, y_pred)
    cmm=cm.reshape(-1,4)
    
    res_labels=pd.DataFrame({'Event': classes, 'Threshold': threshold, 'Exact Matching Score': acc, 'Precision': precision, 'Recall': recall, 'F1-Score': f1})
    
    res_labels = pd.concat([res_labels, pd.DataFrame(cmm, columns=['tn', 'fp', 'fn', 'tp'])], axis=1)
    
    agg_precision, agg_recall, agg_f1, agg_support = precision_recall_fscore_support(y_true, y_pred, average='samples')
    agg=pd.DataFrame({'Event': ['Aggregate'], 'Threshold': [np.nan], 'Exact Matching Score': agg_acc, 'Precision': agg_precision, 'Recall': agg_recall, 'F1-Score': agg_f1})
    
    
    agg = pd.concat([agg, pd.DataFrame(data=[np.nan]*4, index=['tn', 'fp', 'fn', 'tp']).T], axis=1)

    res=pd.concat([res_labels, agg]).reset_index(drop=True)
    
    return res

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
    
    agg_precision, agg_recall, agg_f1, agg_support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
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




# dist = [ np.sqrt((1-re)**2 + (1-pre)**2) for re, pre in zip(re, pre) ]
# opt_id.append(dist.index(min(dist)))
# t = Thresholds[i]
# opt_thres = t[opt_id[i]]
# result[event_type] = opt_thres

def plot_roc_curves(first_lim, y_true, y_pred, classes, model_type):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    Thresholds = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], Thresholds[i] = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    opt_id = []
    for i in range(len(classes)): 
        fprr = fpr[i]
        tprr = tpr[i]
        dist = [ np.sqrt((0-f)**2 + (1-t)**2) for f, t in zip(fprr, tprr) ]
        opt_id.append(dist.index(min(dist)))
        t = Thresholds[i]
        opt_thres = t[opt_id[i]]
        #f1_score_opt = 2.0*fprr[opt_id[i]]*tprr[opt_id[i]] / (re[opt_id[i]]+pre[opt_id[i]])
        #print(re[opt_id[i]], pre[opt_id[i]], f1_score_opt)
        print("Optimal ",classes[i]," Threshold = ", opt_thres )
        print()
        
    cmaps = OrderedDict()
    plt.figure(figsize=(12,8))
    for i in range(5): 
        plt.plot(fpr[i], tpr[i], label = classes[i], linewidth=3.0)
        #plt.plot(recalls[i][3:-4], precisions[i][3:-4], label = classes[i], linewidth=3.0)
        plt.plot([fpr[i][opt_id[i]],0],[tpr[i][opt_id[i]],1], 'ro--')
        plt.scatter(fpr[i][opt_id[i]],tpr[i][opt_id[i]], marker='*', s=900)
        plt.text(0.85, 0.8, 'opt. threshold', color='r', size=18)
    plt.xlabel('False Positive Rate', size=20)
    plt.ylabel('True Positive Rate', size=20)
    plt.xlim(first_lim,1.05)
    plt.ylim(first_lim,1.05)
    plt.title("ROC Curves")
    plt.legend()
    #plt.axis('off')
    #plt.axis('equal')
    #plt.gca().set_aspect('equal')
    plt.savefig('figures/ROC_curves_zoom' + model_type +'.png', dpi=300)
    plt.show()    

    

def plot_pr_curves(first_lim, y_true, y_pred, classes, model_type):
    precisions = dict()
    recalls = dict()
    Thresholds = dict()
    for i in range(5):
        precisions[i], recalls[i], Thresholds[i] = precision_recall_curve(y_true[:, i], y_pred[:, i])
    
    opt_id = []
    for i in range(5): 
        re = recalls[i]
        pre = precisions[i]
        dist = [ np.sqrt((1-re)**2 + (1-pre)**2) for re, pre in zip(re, pre) ]
        opt_id.append(dist.index(min(dist)))
        t = Thresholds[i]
        opt_thres = t[opt_id[i]]
        f1_score_opt = 2.0*re[opt_id[i]]*pre[opt_id[i]] / (re[opt_id[i]]+pre[opt_id[i]])
        print(re[opt_id[i]], pre[opt_id[i]], f1_score_opt)
        print("Optimal ",classes[i]," Threshold = ", opt_thres )
        print()

    cmaps = OrderedDict()
    plt.figure(figsize=(12,8))
    for i in range(5): 
        plt.plot(recalls[i], precisions[i], label = classes[i], linewidth=3.0)
        #plt.plot(recalls[i][3:-4], precisions[i][3:-4], label = classes[i], linewidth=3.0)
        plt.plot([recalls[i][opt_id[i]],1],[precisions[i][opt_id[i]],1], 'ro--')
        plt.scatter(recalls[i][opt_id[i]],precisions[i][opt_id[i]], marker='*', s=900)
        plt.text(0.85, 0.8, 'opt. threshold', color='r', size=18)
    plt.xlabel('Recall', size=20)
    plt.ylabel('Precision', size=20)
    plt.xlim(first_lim,1.05)
    plt.ylim(first_lim,1.05)
    plt.title("Precision Recall Curves")
    plt.legend()
    #plt.axis('off')
    #plt.axis('equal')
    #plt.gca().set_aspect('equal')
    plt.savefig('figures/precision_recall_curve_zoom' + model_type +'.png', dpi=300)
    plt.show()    

    
def table_to_latex(table):
    print(table.to_latex(float_format=lambda x: '%.3f' % truncate_decimals(x,3)))