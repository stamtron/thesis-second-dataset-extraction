#import sys
#sys.path.append('../helpers/')
from new_extraction_function import *

def suspension_extraction(df, videos, path):
    codes = ['EXE','EXS','FJ','AN','SUS','SUE']
    df_new = df[df['Observation Code'].isin(codes)]
    df_new = df_new.reset_index(drop=True)
    
    #videos = Path('/media/data/astamoulakatos/Survey-2-2012/Project 1/IC2/KP155.800-186.495_D/')
    
    #videos = videos / df_new['folder'][0]
    
    video_paths = videos
    
    for k in tqdm(range(0,len(df_new)-1)): 
        for i in range(len(video_paths)):
            if 'Ch2' in video_paths[i].parts[-1]:
                ch2_video = video_paths[i]
            if 'Ch1' in video_paths[i].parts[-1]:
                ch1_video = video_paths[i]
            if 'Ch3' in video_paths[i].parts[-1]:
                ch3_video = video_paths[i]

        if (df_new['KP'][k] <= df_new['KP'][k+1]):
            if (df_new['Observation Code'][k] == 'SUS') and (df_new['Observation Code'][k+1] == 'SUE'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                start = df_new['offset_Ch2'][k]
                stop = df_new['offset_Ch2'][k+1]
                extract_frames(str(ch2_video), path_to_save, 2, start, stop, nframes=None)      
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                start = df_new['offset_Ch1'][k]
                stop = df_new['offset_Ch1'][k+1]
                extract_frames(str(ch2_video), path_to_save, 1, start, stop, nframes=None)
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                start = df_new['offset_Ch3'][k]
                stop = df_new['offset_Ch3'][k+1]
                extract_frames(str(ch2_video), path_to_save, 3, start, stop, nframes=None)     
            if (df_new['Observation Code'][k] == 'SUS') and (df_new['Observation Code'][k+1] == 'FJ'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                start = df_new['offset_Ch2'][k]
                stop = df_new['offset_Ch2'][k+1]
                extract_frames(str(ch2_video), path_to_save, 2, start, stop, nframes=None)      
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                start = df_new['offset_Ch1'][k]
                stop = df_new['offset_Ch1'][k+1]
                extract_frames(str(ch2_video), path_to_save, 1, start, stop, nframes=None)
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                start = df_new['offset_Ch3'][k]
                stop = df_new['offset_Ch3'][k+1]
                extract_frames(str(ch2_video), path_to_save, 3, start, stop, nframes=None)    
            if (df_new['Observation Code'][k] == 'FJ') and (df_new['Observation Code'][k+1] == 'SUE'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                start = df_new['offset_Ch2'][k]
                stop = df_new['offset_Ch2'][k+1]
                extract_frames(str(ch2_video), path_to_save, 2, start, stop, nframes=None)      
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                start = df_new['offset_Ch1'][k]
                stop = df_new['offset_Ch1'][k+1]
                extract_frames(str(ch2_video), path_to_save, 1, start, stop, nframes=None)
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                start = df_new['offset_Ch3'][k]
                stop = df_new['offset_Ch3'][k+1]
                extract_frames(str(ch2_video), path_to_save, 3, start, stop, nframes=None)  
        else:
            if (df_new['Observation Code'][k] == 'SUE') and (df_new['Observation Code'][k+1] == 'SUS'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                start = df_new['offset_Ch2'][k]
                stop = df_new['offset_Ch2'][k+1]
                extract_frames(str(ch2_video), path_to_save, 2, start, stop, nframes=None)      
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                start = df_new['offset_Ch1'][k]
                stop = df_new['offset_Ch1'][k+1]
                extract_frames(str(ch2_video), path_to_save, 1, start, stop, nframes=None)      
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                start = df_new['offset_Ch3'][k]
                stop = df_new['offset_Ch3'][k+1]
                extract_frames(str(ch2_video), path_to_save, 3, start, stop, nframes=None)   
            if (df_new['Observation Code'][k] == 'SUE') and (df_new['Observation Code'][k+1] == 'FJ'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                start = df_new['offset_Ch2'][k]
                stop = df_new['offset_Ch2'][k+1]
                extract_frames(str(ch2_video), path_to_save, 2, start, stop, nframes=None)      
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                start = df_new['offset_Ch1'][k]
                stop = df_new['offset_Ch1'][k+1]
                extract_frames(str(ch2_video), path_to_save, 1, start, stop, nframes=None)      
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                start = df_new['offset_Ch3'][k]
                stop = df_new['offset_Ch3'][k+1]
                extract_frames(str(ch2_video), path_to_save, 3, start, stop, nframes=None) 
            if (df_new['Observation Code'][k] == 'FJ') and (df_new['Observation Code'][k+1] == 'SUS'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                start = df_new['offset_Ch2'][k]
                stop = df_new['offset_Ch2'][k+1]
                extract_frames(str(ch2_video), path_to_save, 2, start, stop, nframes=None)      
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                start = df_new['offset_Ch1'][k]
                stop = df_new['offset_Ch1'][k+1]
                extract_frames(str(ch2_video), path_to_save, 1, start, stop, nframes=None)      
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                start = df_new['offset_Ch3'][k]
                stop = df_new['offset_Ch3'][k+1]
                extract_frames(str(ch2_video), path_to_save, 3, start, stop, nframes=None) 