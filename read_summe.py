import glob
import numpy as np 
from scipy.io import loadmat
import pandas as pd

GT_PATH = "./SumMe/GT/"

def get_groundtruth(VIDEO_ID):
    data = loadmat(GT_PATH + VIDEO_ID)
    return np.ravel(data['gt_score'])

def get_videos_id():
    return glob.glob1(GT_PATH, "*.m*")

def get_video_name(VIDEO_ID):
    return VIDEO_ID.split('.')[0]

def get_memorability(VIDEO_NAME):
    mems = pd.read_csv('./SumMe-memorability.csv', sep=';', header=0)
    mems_of_video = mems[mems.video_name == VIDEO_NAME  + ".webm"]

    if mems_of_video.empty:
        return []
    else:
        mems_of_video = mems_of_video.memorability_scores.iloc[0]
        return list(map(float, mems_of_video.split(',')))

def get_iovc(VIDEO_NAME):
    iovc_videos = pd.read_json("./SumMe-iovc.json", lines=True)
    return iovc_videos[VIDEO_NAME + ".webm"].iloc[0]

def get_summary(VIDEO_KEY):
    summaries = pd.read_csv('./SumMe-PGLSUM-summary.csv', sep=';', header=0)
    summary_row = summaries[summaries['video_key'] == 'video_' + str(VIDEO_KEY)]
    if summary_row.empty:
        return []
    else:
        summary_row = summary_row.summary.iloc[0]
        return np.array(list(map(int, summary_row.split(','))))


if __name__ == "__main__":
    videos_id = get_videos_id()
    for index in range(len(videos_id)):
        video_id = videos_id[index]
        video_name = get_video_name(video_id)

        gt = get_groundtruth(video_id)
        memorability = get_memorability(video_name)
        iovc = get_iovc(video_name)
        summary = get_summary(index+1)
        
        print(video_name, len(gt), "-", len(summary), len(gt) == len(summary))