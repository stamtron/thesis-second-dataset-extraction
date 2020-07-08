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
        
from convenience import is_cv3
import cv2

def count_frames(path, override=False):
	# grab a pointer to the video file and initialize the total
	# number of frames read
	video = cv2.VideoCapture(path)
	total = 0

	# if the override flag is passed in, revert to the manual
	# method of counting frames
	if override:
		total = count_frames_manual(video)

	# otherwise, let's try the fast way first
	else:
		# lets try to determine the number of frames in a video
		# via video properties; this method can be very buggy
		# and might throw an error based on your OpenCV version
		# or may fail entirely based on your which video codecs
		# you have installed
		try:
			# check if we are using OpenCV 3
			if is_cv3():
				total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

			# otherwise, we are using OpenCV 2.4
			else:
				total = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

		# uh-oh, we got an error -- revert to counting manually
		except:
			total = count_frames_manual(video)

	# release the video file pointer
	video.release()

	# return the total number of frames in the video
	return total

def count_frames_manual(video):
	# initialize the total number of frames read
	total = 0

	# loop over the frames of the video
	while True:
		# grab the current frame
		(grabbed, frame) = video.read()
	 
		# check to see if we have reached the end of the
		# video
		if not grabbed:
			break

		# increment the total number of frames read
		total += 1

	# return the total number of frames in the video file
	return total

def columns_of_interest(df):
    
    col_list = ['folder', 'video1', 'offset_Ch1', 'video2', 'offset_Ch2',
            'video3', 'offset_Ch3', 'VWTimestamp', 'Description', 'KP', 'Observation Code']
    
    df = df[col_list]
    codes = ['EXE','EXS','FJ','FJS','FJE','AN','ANS','ANE','SUS', 'SUE']
    df = df[df['Observation Code'].isin(codes)]
    
    return df

def fill_in_KP(df):
    
    df['KP'] = df['KP'].replace('-',np.NaN)
    df['KP'] = df['KP'].fillna(method='bfill')
    df = df.sort_values(by=['KP'])
    return df

def add_event_for_start(df):
    df_start = df[0:1].copy()
    df_start['offset_Ch1'].iloc[0] = 0.0
    df_start['offset_Ch2'].iloc[0] = 0.0
    df_start['offset_Ch3'].iloc[0] = 0.0
    codes_start = ['EXE','FJ','FJS','FJE','AN','ANS','ANE','SUS', 'SUE']
    
    if df_start['Observation Code'][0] in codes_start:
        df_start['Observation Code'][0] = 'EXS'
    else:
        df_start['Observation Code'][0] = 'EXE'
        
    df = pd.concat([df_start, df])  
    df = df.reset_index(drop=True)
    
    return df
   

def add_event_for_end(df, videos):
    
    df_end = df[(len(df)-1):len(df)].copy()
    codes_end = ['EXS','FJ','FJS','FJE','AN','ANS','ANE','SUS', 'SUE']
    
    if df_end['Observation Code'][len(df)-1] not in codes_end:
        df_end['Observation Code'][len(df)-1] = 'EXS'
    else:
        df_end['Observation Code'][len(df)-1] = 'EXE'
    
    for i in range(len(videos)):
        if 'Ch2' in videos[i].parts[-1]:
            frames_ch2 = count_frames(str(videos[i]))
        if 'Ch1' in videos[i].parts[-1]:
            frames_ch1 = count_frames(str(videos[i]))
        if 'Ch3' in videos[i].parts[-1]:
            frames_ch3 = count_frames(str(videos[i]))
            
    df_end['offset_Ch1'][len(df)-1] = frames_ch1 / 25
    df_end['offset_Ch2'][len(df)-1] = frames_ch2 / 25 
    df_end['offset_Ch3'][len(df)-1] = frames_ch3 / 25    
    
    df = pd.concat([df, df_end])
    df = df.reset_index(drop=True)
    
    return df
    