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
from datetime import datetime, timedelta

########################################################

######################################
# fn to parse timestamp
######################################
def parseTimestamp(ts):
    return datetime.strptime(ts,'%Y%m%d%H%M%S%f')

######################################
# fn to create timestamp
######################################
def createTimestamp(dt):
    ts = str(dt.year)
    ts = ts+"{0:0=2d}".format(dt.month)
    ts = ts+"{0:0=2d}".format(dt.day)
    ts = ts+"{0:0=2d}".format(dt.hour)
    ts = ts+"{0:0=2d}".format(dt.minute)
    ts = ts+"{0:0=2d}".format(dt.second)
    ts = ts+"{0:0=3d}".format(int(dt.microsecond/1000))

    return ts

########################################################

def frame_extraction(center_video_path, path_to_save, i, df):
    m = re.search('tos/(.+?).mpg', center_video_path)
    if m:
        name = m.group(1)
    name = name.replace('/','__')
    fileTS = parseTimestamp(center_video_path[-31:-14]) 
    # extract timestamp of video file and convert to datetime
    path_to_save = path_to_save + name + '__' + str(int(i/2))
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    path_to_save = path_to_save + '/'

    cap = cv2.VideoCapture(center_video_path)
    
    if(i==-1):
        start = 0
        end = float(df_new['offset'][i+1])*1000 - 100
    elif(i < 9999):
        start = float(df_new['offset'][i])*1000 + 20 
        end = float(df_new['offset'][i+1])*1000 - 100
    else:
        start = float(df_new['offset'][len(df_new)-1])*1000 + 20
        fps = cap.get(cv2.CAP_PROP_FPS)
        totalNoFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        end = 1000*float(totalNoFrames) / float(fps)


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
        
        frameTS = createTimestamp(fileTS + timedelta(seconds = frames[j]/25))
        
        cv2.imwrite(path_to_save + 'frameTS' + frameTS + '.png', im)

########################################################

codes = ['EXE','EXS','FJS','FJE','ANS','ANE','SUS', 'SUE']

########################################################

event_paths = np.sort(glob.glob('/home/cmccaig/Jupyter/N-Sea/csv_for_frame_extraction/'+'*'))

########################################################

videos_root_path = '/media/data/astamoulakatos/Survey-2-2012/Project 1/IC2/'

video_paths = np.sort(glob.glob(videos_root_path  + 'KP*'))[1:]

#event_paths = event_paths[range(6,7)]
#video_paths = video_paths[range(6,7)]

print("TYPE: ", type(event_paths))
print("SHAPE: ", event_paths.shape)
print("DIMS: ", event_paths.ndim)

########################################################
print('len(event_paths): ',len(event_paths),'\n')
#for ep in      range(len(event_paths)):
for ep in tqdm(range(len(event_paths))):
    event_path = event_paths[ep]
    print('Path: ', event_path)
    videoName = video_paths[ep]
    
    df = pd.read_csv(event_path)
    
    df_new = df[df['Observation Code'].isin(codes)]
    
    df_new = df_new.reset_index(drop=True)
    
    folders = np.unique(df_new['folder'])
    
    print('EVENT FILE: ',event_path,'\nFOLDERS: ',folders,'\n\n')
    
    # for each events file
    for f in range(len(folders)):
        print('\n!! folder: ',folders[f],'\t',f)
        df_newF = df_new[df_new['folder'] == folders[f]]
        df_newF.loc[:,('KP')] = df_newF.loc[:,('KP')].replace('-',np.NaN)
        df_newF.loc[:,('KP')] = df_newF.loc[:,('KP')].astype(float)
        df_newF = df_newF.reset_index(drop=True)

        KPS = df_newF['KP'].replace('-',np.NaN).dropna().reset_index(drop=True) # exract KP and keep only numeric

        if(KPS[0] < KPS[len(KPS)-1]):
            forward = True
        else:
            forward = False

        exsT = df_newF[df_newF['Observation Code'] == 'EXS']['VWTimestamp']
        exeT = df_newF[df_newF['Observation Code'] == 'EXE']['VWTimestamp']

        if((not len(exsT)) and (not len(exeT))):
            if(len(df_newF)):              # no start or end of exposure but other events
                StartExp = True            # pipe exposed throughout   !!!THIS MAY NOT BE TRUE IF PIPE
            else:                          #                           !!!IS COVERED BY SAND BUT LASER
                StartExp = False           # pipe starts buried        !!!LINE CURVED THEN COULD BE NO 
                                           #                           !!!EVENTS BUT IT'S EXPOSED
                continue                   # just going to ignore these 
                                           # files since only know if it
                                           # starts exposed by looking
                                           # !!!MIGHT MISS A LARGE NUMBER OF FRAMES
                                           # !!!THAT ARE BURIED OR PARTLY EXPOSED

        elif((len(exsT)) and (not len(exeT))):
            if(forward):                   # start of exposure but no end
                StartExp = False           # pipe starts buried
            else:
                StartExp = True            # pipe starts exposed

        elif((len(exeT)) and (not len(exsT))):
            if(forward):                   # end of exposure but no start                   
                StartExp = True            # pipe starts exposed
            else:
                StartExp = False           # pipe starts buried

        else:
            if(forward):                   # start and end of exposure
                if(np.min(exsT) < np.min(exeT)): # KP going forward
                    StartExp = False       # pipe starts buried
                else:
                    StartExp = True        # pipe starts exposed
            else:
                if(np.min(exsT) < np.min(exeT)): # KP going backward
                    StartExp = True        # pipe starts exposed
                else:
                    StartExp = False       # pipe starts buried


        center_video = os.path.join(videoName,folders[f])
        for name in glob.glob(center_video + '/*Ch3.mpg'):
            center_video_path = name

        # collect frames before first event (I'm assuming these frames are either exposure only or burial)
        if(StartExp):
            if (StartExp):
                if (forward):
                    print('Exposure forward start')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'

                    frame_extraction(center_video_path, path_to_save, -1, df_newF)
                    curClass = [0, 0, 1, 0, 0]
                else:
                    print('Exposure backward start')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'

                    frame_extraction(center_video_path, path_to_save, -1, df_newF)
                    curClass = [0, 0, 1, 0, 0]

            else:
                if (forward):
                    print('Burial forward start')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/bur/'

                    frame_extraction(center_video_path, path_to_save, -1, df_newF)
                    curClass = [0, 1, 0, 0, 0]
                else:
                    print('Burial backward start')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/bur/'

                    frame_extraction(center_video_path, path_to_save, -1, df_newF)
                    curClass = [0, 1, 0, 0, 0]


        # collect frames between each pair of events
        for i in tqdm(range(len(df_newF)-1)): 

            if (df_newF['Observation Code'][i] == 'EXS'):
                if (forward):
                    print('Exposure forward')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'

                    curClass = [0, 0, 1, 0, 0]
                    frame_extraction(center_video_path, path_to_save, i, df_newF)

                else:
                    print('Burial backward')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/bur/'

                    frame_extraction(center_video_path, path_to_save, i, df_newF)
                    curClass = [0, 1, 0, 0, 0]

            elif (df_newF['Observation Code'][i] == 'EXE'):
                if (forward):
                    print('Burial forward')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/bur/'

                    frame_extraction(center_video_path, path_to_save, i, df_newF)
                    curClass = [0, 1, 0, 0, 0]

                else:
                    print('Exposure backward')
                    path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'

                    frame_extraction(center_video_path, path_to_save, i, df_newF)
                    curClass = [0, 0, 1, 0, 0]


            # this could miss out cases where a field joint or anode occur on a free span
            elif (df_newF['Observation Code'][i] == 'SUS'):
                if (forward):
                    print('Free span forward')
                    if(curClass[0]==1):    # Anode already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_and_fs/'
                        curClass = [1, 0, 1, 0, 1]
                    elif(curClass[3]==1):  # Field Joint already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fj_fs/'
                        curClass = [0, 0, 1, 1, 1]
                    else:                  # Free Span only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fs/'
                        curClass = [0, 0, 1, 0, 1]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)
                else:
                    print('Exposure after Free span backward')
                    if(curClass[0]==1):    # Anode already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_and/'
                        curClass = [1, 0, 1, 0, 0]
                    elif(curClass[0]==1):  # Field Joint already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fj/'
                        curClass = [0, 0, 1, 1, 0]
                    else:                  # Exposure only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'
                        curClass = [0, 0, 1, 0, 0]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)

            elif (df_newF['Observation Code'][i] == 'SUE'):
                if (forward):
                    print('Exposure after free span forward')
                    if(curClass[0]==1):    # Anode already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_and/'
                        curClass = [1, 0, 1, 0, 0]
                    elif(curClass[0]==1):  # Field Joint already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fj/'
                        curClass = [0, 0, 1, 1, 0]
                    else:                  # Exposure only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'
                        curClass = [0, 0, 1, 0, 0]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)
                else:
                    print('Free span backward')
                    if(curClass[0]==1):    # Anode already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_and_fs/'
                        curClass = [1, 0, 1, 0, 1]
                    elif(curClass[3]==1):  # Field Joint already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fj_fs/'
                        curClass = [0, 0, 1, 1, 1]
                    else:                  # Free Span only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fs/'
                        curClass = [0, 0, 1, 0, 1]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)


            # assumming after FJ, AN or FS we have exposure        
            elif (df_newF['Observation Code'][i] == 'FJE'):
                    print('Exposure after FJ')
                    if(curClass[4]==1):    # Field Joint ends during free span
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fs/'
                        curClass = [0, 0, 1, 0, 1]
                    else:    # exposure only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'
                        curClass = [0, 0, 1, 0, 0]


                    frame_extraction(center_video_path, path_to_save, i, df_newF)

            elif (df_newF['Observation Code'][i] == 'ANE'):
                    print('Exposure after AN')
                    if(curClass[4]==1):    # Anode during free span
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fs/'
                        curClass = [0, 0, 1, 0, 1]
                    else:    # exposure only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'
                        curClass = [0, 0, 1, 0, 0]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)

            elif (df_newF['Observation Code'][i] == 'FJS'):
                    print('Field joint')
                    if(curClass[4]==1):    # Free span already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fj_fs/'
                        curClass = [0, 0, 1, 1, 1]
                    else:                  # Field Joint only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fj/'
                        curClass = [0, 0, 1, 1, 0]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)


            elif (df_newF['Observation Code'][i] == 'ANS'):
                    print('Anode')
                    if(curClass[0]==1):    # Free span already visible
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_and_fs/'
                        curClass = [1, 0, 1, 0, 1]
                    else:                  # Anode only
                        path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp_fs/'
                        curClass = [1, 0, 1, 0, 0]

                    frame_extraction(center_video_path, path_to_save, i, df_newF)
        
        
        # collect after last event (I'm assuming these frames are either exposure only or burial)
        lastObsCode = df_newF['Observation Code'][len(df_newF)-1]
        if (lastObsCode == 'EXE' and forward):
            path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/bur/'

            frame_extraction(center_video_path, path_to_save, 9999, df_newF)

        elif (lastObsCode == 'EXS' and (not forward)):
            print('Burial backward end')
            path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/bur/'

            frame_extraction(center_video_path, path_to_save, 9999, df_newF)

        else:
            print('Exposure end')
            path_to_save = '/media/scratch/cmccaig/nsea_video_jpegs/right/exp/'

            frame_extraction(center_video_path, path_to_save, 9999, df_newF)
            
########################################################
