#import sys
#sys.path.append('../helpers/')
from helpers_frame_extraction import *

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
        if (df_new['offset_Ch1'][k] < df_new['offset_Ch1'][k+1]):
            if (df_new['Observation Code'][k] == 'SUS') & (df_new['Observation Code'][k+1] == 'SUE'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                frame_extraction_with_sns(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                frame_extraction_with_sns(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                frame_extraction_with_sns(str(ch3_video), path_to_save, k, df_new, 3, 'up')
            if (df_new['Observation Code'][k] == 'SUS') & (df_new['Observation Code'][k+1] == 'FJ' or df_new['Observation Code'][k+1] == 'AN'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'up')
            if (df_new['Observation Code'][k] == 'FJ' or df_new['Observation Code'][k] == 'AN') & (df_new['Observation Code'][k+1] == 'SUE'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'up')

        else:
            if (df_new['Observation Code'][k] == 'SUE') & (df_new['Observation Code'][k+1] == 'SUS'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                frame_extraction_with_sns(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                frame_extraction_with_sns(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                frame_extraction_with_sns(str(ch3_video), path_to_save, k, df_new, 3, 'down')
            if (df_new['Observation Code'][k] == 'SUE') & (df_new['Observation Code'][k+1] == 'FJ' or df_new['Observation Code'][k+1] == 'AN'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'down')
            if (df_new['Observation Code'][k+1] == 'FJ' or df_new['Observation Code'][k+1] == 'AN') & (df_new['Observation Code'][k] == 'SUE'):
                path_to_save = path + 'centre_Ch2/exp_fs/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp_fs/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp_fs/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'down')
