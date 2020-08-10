#import sys
#sys.path.append('../helpers/')
from helpers_frame_extraction import *

def exposure_extraction(df, videos, path):
    codes = ['EXE','EXS','FJ','AN','SUS', 'SUE']
    df_new = df[df['Observation Code'].isin(codes)]
    df_new = df_new.reset_index(drop=True)
    
#     videos = Path('/media/data/astamoulakatos/Survey-2-2012/Project 1/IC2/KP078.553-119.732_A/')
#     videos = videos / df_new['folder'][0]
    
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
            if (df_new['Observation Code'][k] == 'EXS') and (df_new['Observation Code'][k+1] == 'EXE' or df_new['Observation Code'][k+1] == 'SUS'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_with_sns(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_with_sns(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_with_sns(str(ch3_video), path_to_save, k, df_new, 3, 'up')
                # works fine
            if (df_new['Observation Code'][k] == 'EXS') and (df_new['Observation Code'][k+1] == 'AN' or df_new['Observation Code'][k+1] == 'FJ'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_special_cases_after_startend(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_special_cases_after_startend(str(ch3_video), path_to_save, k, df_new, 3, 'up')  
                # works fine
            if (df_new['Observation Code'][k] == 'FJ' or df_new['Observation Code'][k] == 'AN') and (df_new['Observation Code'][k+1] == 'AN' or df_new['Observation Code'][k+1] == 'FJ'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_between_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_between_fjan(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_between_fjan(str(ch3_video), path_to_save, k, df_new, 3, 'up')
            if (df_new['Observation Code'][k] == 'FJ' or df_new['Observation Code'][k] == 'AN') and (df_new['Observation Code'][k+1] == 'SUS' or df_new['Observation Code'][k+1] == 'EXE'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_special_cases_after_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_special_cases_after_fjan(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_special_cases_after_fjan(str(ch3_video), path_to_save, k, df_new, 3, 'up')
        else:
            if (df_new['Observation Code'][k+1] == 'EXS') and (df_new['Observation Code'][k] == 'EXE' or df_new['Observation Code'][k] == 'SUS'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_with_sns(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_with_sns(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_with_sns(str(ch3_video), path_to_save, k, df_new, 3, 'down')
                # works fine
            if (df_new['Observation Code'][k+1] == 'EXS') and (df_new['Observation Code'][k] == 'AN' or df_new['Observation Code'][k] == 'FJ'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_special_cases_after_startend(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_special_cases_after_startend(str(ch3_video), path_to_save, k, df_new, 3, 'down')
                # works fine
            if (df_new['Observation Code'][k] == 'FJ' or df_new['Observation Code'][k] == 'AN') and (df_new['Observation Code'][k+1] == 'AN' or df_new['Observation Code'][k+1] == 'FJ'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_between_fjan(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_between_fjan(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_between_fjan(str(ch3_video), path_to_save, k, df_new, 3, 'down')
            if (df_new['Observation Code'][k+1] == 'FJ' or df_new['Observation Code'][k+1] == 'AN') and (df_new['Observation Code'][k] == 'EXE' or df_new['Observation Code'][k+1] == 'SUS'):
                path_to_save = path + 'centre_Ch2/exp/'
                frame_extraction_special_cases_after_startend(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = path + 'left_port_Ch1/exp/'
                frame_extraction_special_cases_after_startend(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = path + 'right_starboard_Ch3/exp/'
                frame_extraction_special_cases_after_startend(str(ch3_video), path_to_save, k, df_new, 3, 'down')


