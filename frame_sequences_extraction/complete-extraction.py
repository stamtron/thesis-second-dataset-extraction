import sys
sys.path.append('../helpers/')

from helpers_frame_extraction import *
from fs_extraction import *
from fj_and_extraction import *
from bur_extraction import *
from exp_extraction import *

csv_path = Path('../csv_for_frame_extraction/')

csv_paths = list(csv_path.glob('*'))

folder_path = Path('/media/data/astamoulakatos/Survey-2-2012/Project 1/IC2')

folders_path = list(folder_path.glob('*'))

folders_path = [folders_path[9], folders_path[0], folders_path[5], folders_path[1], folders_path[10], folders_path[4], folders_path[7], folders_path[3], folders_path[8], folders_path[2]]
for c, vf in zip(csv_paths, folders_path):
    print(c , vf)
    
path = '/media/raid/astamoulakatos/nsea_frame_sequences/'
counter = 0
for c, vf in zip(csv_paths, folders_path):
    events_csv = pd.read_csv(str(c))
    folders = events_csv['folder'].unique()
    for f in folders:
        df_vid = events_csv[events_csv['folder'] == f] 
        df_vid = columns_of_interest(df_vid)
        df_vid = df_vid.reset_index(drop = True)
        df_vid = fill_in_KP(df_vid)
        df_vid = df_vid.reset_index(drop = True)
        if df_vid.empty:
            print('DataFrame is empty!')
        else:
            codes  = df_vid['Observation Code'].unique()
            if 'FJS' or 'ANS' in codes:
                df_vid = add_event_for_start(df_vid)
                vid_path = vf/f
                videos = list(vid_path.glob('*'))
                df_vid = add_event_for_end(df_vid, videos)
                df_vid = convert_to_ms(df_vid)
                #print(videos)
                counter += 1
                print(counter)
                #suspension_extraction(df_vid, videos, path)
                fieldjoint_anode_extraction(df_vid, videos, path)
                #burial_extraction(df_vid, videos, path)
                #exposure_extraction(df_vid, videos, path)