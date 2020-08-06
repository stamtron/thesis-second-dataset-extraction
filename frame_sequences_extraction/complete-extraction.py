import sys
sys.path.append('../helpers/')

from helpers_frame_extraction import *
from fs_extraction import *
from fj_and_extraction import *
from bur_extraction import *
from exp_extraction import *

csv_path = Path('../csvs_with_startends/')

csv_paths = list(csv_path.glob('*'))

folder_path = Path('/media/data/astamoulakatos/Survey-2-2012/Project 1/IC2')

folders_path = list(folder_path.glob('*'))

folders_path = [folders_path[9], folders_path[3], folders_path[2]]

for c, vf in zip(csv_paths, folders_path):
    print(c , vf)
    
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
            df_vid = add_event_for_start(df_vid)
            df_vid = add_event_for_end(df_vid)
            vid_path = vf/f
            videos = list(vid_path.glob('*'))
            #print(videos)
            counter += 1
            print(counter)
            suspension_extraction(df, videos)
            fieldjoint_anode_extraction(df, videos)
            burial_extraction(df, videos)
            exposure_extraction(df, videos)