from helpers_frame_extraction import *

def burial_extraction(df, videos, folder):
    codes = ['EXE','EXS','FJS','ANS', 'FJE', 'ANE']
    df_new = df[df['Observation Code'].isin(codes)]
    df_new = df_new.reset_index(drop=True)
    df_new = df_new.sort_values(by=['KP'])
    df_new = df_new.reset_index(drop=True)
    
    videos = Path('/media/data/astamoulakatos/Survey-2-2012/Project 1/IC2/KP155.800-186.495_D/')
    videos = videos / df_new['folder'][0]
    
    video_paths = list(videos.glob('*'))
    
    for k in tqdm(range(0,len(df_new)-1)): 
        for i in range(len(video_paths)):
            if 'Ch2' in video_paths[i].parts[-1]:
                ch2_video = video_paths[i]
            if 'Ch1' in video_paths[i].parts[-1]:
                ch1_video = video_paths[i]
            if 'Ch3' in video_paths[i].parts[-1]:
                ch3_video = video_paths[i]
        if (df_new['offset_Ch1'][k] < df_new['offset_Ch1'][k+1]):
            if (df_new['Observation Code'][k] == 'ANS') & (df_new['Observation Code'][k+1] == 'ANE'):
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/centre_Ch2/exp_and/'
                frame_extraction(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/left_port_Ch1/exp_and/'
                frame_extraction(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/right_starboard_Ch3/exp_and/'
                frame_extraction(str(ch3_video), path_to_save, k, df_new, 3, 'up')
            if (df_new['Observation Code'][k] == 'FJS') & (df_new['Observation Code'][k+1] == 'FJE'):
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/centre_Ch2/exp_fj/'
                frame_extraction(str(ch2_video), path_to_save, k, df_new, 2, 'up')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/left_port_Ch1/exp_fj/'
                frame_extraction(str(ch1_video), path_to_save, k, df_new, 1, 'up')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/right_starboard_Ch3/exp_fj/'
                frame_extraction(str(ch3_video), path_to_save, k, df_new, 3, 'up')
        else:
            if (df_new['Observation Code'][k] == 'ANE') & (df_new['Observation Code'][k+1] == 'ANS'):
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/centre_Ch2/exp_and/'
                frame_extraction(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/left_port_Ch1/exp_and/'
                frame_extraction(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/right_starboard_Ch3/exp_and/'
                frame_extraction(str(ch3_video), path_to_save, k, df_new, 3, 'down')
            if (df_new['Observation Code'][k] == 'FJE') & (df_new['Observation Code'][k+1] == 'FJS'):
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/centre_Ch2/exp_fj/'
                frame_extraction(str(ch2_video), path_to_save, k, df_new, 2, 'down')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/left_port_Ch1/exp_fj/'
                frame_extraction(str(ch1_video), path_to_save, k, df_new, 1, 'down')
                path_to_save = '/media/scratch/astamoulakatos/nsea_frame_test/right_starboard_Ch3/exp_fj/'
                frame_extraction(str(ch3_video), path_to_save, k, df_new, 3, 'down')