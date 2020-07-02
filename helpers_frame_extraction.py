import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
import pandas as pd
import numpy as np
import os
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import re
import glob

def frame_extraction(center_video_path, path_to_save, i, df):
    m = re.search('tos/(.+?).mpg', center_video_path)
    if m:
        name = m.group(1)
    name = name.replace('/','__')    
    path_to_save = path_to_save + name + '__' + str(int(i/2))
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    path_to_save = path_to_save + '/'

    start = float(df_new['offset'][i])*1000 + 20 
    end = float(df_new['offset'][i+1])*1000 - 100
    cap = cv2.VideoCapture(center_video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, start)
    #success,image = cap.read()
    s = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.set(cv2.CAP_PROP_POS_MSEC, end)
    #success,image = cap.read()
    e = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frames = np.arange(s, e, 4)
    #if not os.path.exists(path_to_save):
    for j in range(len(frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[j])
        suc,im = cap.read()
        cv2.imwrite(path_to_save + 'frame' + str(j) + '.png', im)