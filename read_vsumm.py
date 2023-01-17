import numpy as np
import pandas as pd
import glob
import cv2

def get_frames_importance(VIDEO_ID):
    video_path = glob.glob("./vsumm/youtube/{}.*".format(VIDEO_ID))[0]
    video = cv2.VideoCapture(video_path)
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    binarized_vect =  nb_frames * [0]

    for user_path in glob.glob("./vsumm/youtube/UserSummary/{}/*/".format(VIDEO_ID)):
        frames = []
        for frame in glob.glob1(user_path, "*"):
            number = ''.join(filter(str.isdigit, frame))
            if number != '':
                frames.append(int(number))
        frames.sort()
        for i in frames:
            print(nb_frames, i)
            binarized_vect[i-1] = 1

    return np.array(binarized_vect)

def read_memorability():
    mems = pd.read_csv('./vsumm-mem-score.csv', sep=';', header=0)

    avg_avg_selected = []
    avg_avg_not_selected = []
    for index in range(len(mems)):
        video_id = mems.loc[index, 'video_name'].split('.')[0]
        binarized_vect = get_frames_importance(video_id)
        print(binarized_vect)
        memorability = list(map(float, mems.iloc[index]['memorability_scores'].split(',')))
        selected_indices = np.where(binarized_vect == 1)[0].tolist()
        not_selected_indices = np.where(binarized_vect == 0)[0].tolist()
        avg_selected = np.average(np.array(memorability).take(selected_indices))
        avg_not_selected = np.average(np.array(memorability).take(not_selected_indices))
        avg_avg_selected.append(avg_selected)
        avg_avg_not_selected.append(avg_not_selected)
        print("Moyenne de mémorabilité sur les frames sélectionnées :", avg_selected)
        print("Moyenne de mémorabilité sur les frames non sélectionnées :", avg_not_selected)

if __name__ == "__main__":
    read_memorability()