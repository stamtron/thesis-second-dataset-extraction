#from fastai.vision import *
import os
#import fastai
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
import numpy as np

test_seeds = [23,68,90,164,2001]
train_seeds = [867,18412,375,45674,6201]

DEBUG = False

from pathlib import Path
def list_dir(self, pattern='*'):
    import glob
    return [Path(x) for x in glob.glob(str(self/pattern))]
Path.ls  = list_dir

classes = ['Anode', 'Burial', 'Exposure', 'Field Joint', 'Free Span']


def nsea_load_data_from_excel(xls_path):
    df_train = pd.read_excel(xls_path, sheet_name='Train')
    df_valid = pd.read_excel(xls_path, sheet_name='Validation')
    df_test = pd.read_excel(xls_path, sheet_name='Test')

    df_train['is_valid'] = False
    df_valid['is_valid'] = True
    df_train_val = pd.concat([df_train, df_valid]).reset_index(drop=True)
    
    return df_train_val, df_test


def split_data(csvfile, random_state=None, test_size=0.2):
    df = pd.read_csv(csvfile)
    X=df.image_name.values
    Y=df.tags.values
    X_train_plus_val, X_test, y_train_plus_val, y_test = train_test_split(X,Y,test_size=test_size, random_state=random_state, shuffle=True)
    df_test = pd.DataFrame({'image_name': X_test, 'tags': y_test})
    df_train_plus_val = pd.DataFrame({'image_name': X_train_plus_val, 'tags': y_train_plus_val})
    
    return df_train_plus_val, df_test

def create_db_train_val(df_train_plus_val, path, folder):
    # Now create a databunch for training and validation
    src = (ImageList.from_df(df_train_plus_val, path, folder=folder).
           split_from_df().label_from_df('tags', label_delim= ' '))
    tfms = get_transforms(max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    size=src.train.x[0].shape[1:]
    dt = src.transform(tfms, size=size).databunch(bs=4, num_workers=0)
    data = dt.normalize(imagenet_stats)
    return data


def create_db_test(df_test, path, folder):
    src_test = (ImageList.from_df(df_test, path, folder=folder).split_none().label_from_df('tags', label_delim= ' '))
    size=src_test.train.x[0].shape[1:]
    dt_test = src_test.transform([], size=size).databunch(bs=4, num_workers=0)
    test_data = dt_test.normalize(imagenet_stats)
    return test_data



def nsea_train_model(data, epochs, lr):
    
    #arch = models.wrn_22
    metrics = []
    callback_fns = []
    if DEBUG:
        metrics.append(partial(accuracy_thresh, thresh=0.8), 
                      partial(fbeta, thresh=0.8))
        callback_fns = ShowGraph
    #learn = cnn_learner(data, arch, metrics=metrics, callback_fns = callback_fns)
    learn = Learner(data, models.wrn_22(), metrics=metrics, callback_fns = callback_fns, loss_func = torch.nn.BCEWithLogitsLoss())
    learn.model.features[14] = torch.nn.Linear(384,5)
    (learn.model).to("cuda:1")
    
    learn.fit_one_cycle(epochs[0], lr[0])
    learn.unfreeze()
    learn.fit_one_cycle(epochs[1], max_lr=lr[1])
    
    return learn
    
def nsea_load_learner(weights, db):
    arch = models.resnet152
    learn = cnn_learner(db, arch)
    learn.load(weights)
    return learn

def nsea_get_preds(weights, train_validation_data=None, test_data=None):
    y_preds_valid = None
    if train_validation_data:
        learn = nsea_load_learner(weights, train_validation_data)
        y_preds_valid = learn.get_preds(DatasetType.Valid)
    y_preds_test = None
    if test_data:
        learn = nsea_load_learner(weights, test_data)
        y_preds_test = learn.get_preds(DatasetType.Train)
    return y_preds_valid, y_preds_test
    
    
# we assume that y contains a tuple of y_pred and targets
def nsea_compute_thresholds(y_true, y_pred):
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
        
        


def new_compute_metrics(y_true, y_pred, thresholds):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
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
    
    agg_precision, agg_recall, agg_f1, agg_support = precision_recall_fscore_support(y_true, y_pred, average='samples')
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
    
    
if __name__ == "__main__": 
    
    # To create the datasets call Do-CV-Splits 
    # Note the datasets will be create for the test_seeds and train_seeds seeds defined in this
    # file (helpers.py -- top of the file)

    
    # To do the training call script train.py in the same folder...
    # Note: the Do-CV-Splits create the XLSX files in the current directory and the
    # train.py assumes that the data in the directory scratch/cvs/. So after you regenerate
    # the datasets you need to move them there. We do not put them there directly to avoid
    # unintentional overwriting of the files.
    
    """
    for i in range(1):
        df_train_val, df_test = split_data(str(path/csvfile), test_size=0.2)
        data_test = create_db_test(df_test, path, folder)

        for k in range(5):
            # train validation data
            data=create_db_train_val(df_train_val, path, folder, validation_size=0.2)

            learn = nsea_train_model(data, epochs=(4,2), lr=(slice(0.01), slice(1e-6,1e-4)))

            model_name = "full-model-{}-{}".format(i, k)
            learn.save(model_name)

            y_preds_valid, y_preds_test = nsea_get_preds(model_name, data, data_test)

            thresholds = nsea_compute_thresholds(y_preds_valid)
            res = compute_metrics(y_preds_valid, thresholds)
            write_dictionary_to_csv('validation-metrics-{}-{}.csv'.format(i,k), res)
            res = compute_metrics(y_preds_test, thresholds)
            write_dictionary_to_csv('test-metrics-{}-{}.csv'.format(i,k), res)

    
    """
    

    



