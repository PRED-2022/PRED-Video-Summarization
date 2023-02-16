"""
    Script pour lire les données inférées
    et la vérité-terrain de VSumm
"""
import numpy as np
import pandas as pd
import glob
import cv2

bases = ["youtube", "open-video"]

def get_frames_importance(VIDEO_ID, base="youtube"):
    """
    Retourne la vérité terrain de la vidéo d'une certaine base.
    
    Parameters
    ----------
    VIDEO_ID : identifiant de la vidéo
    base : bases "youtube" ou "open-video"
    """
    video_path = glob.glob("./vsumm/{}/{}.*".format(base, VIDEO_ID))[0]
    video = cv2.VideoCapture(video_path)
    nb_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    binarized_vect =  nb_frames * [0]
    for user_path in glob.glob("./vsumm/{}/UserSummary/{}/*/".format(base, VIDEO_ID)):
        frames = []
        for frame in glob.glob1(user_path, "*"):
            number = ''.join(filter(str.isdigit, frame))
            if number != '':
                frames.append(int(number))
        frames.sort()
        for i in frames:
            binarized_vect[i-1] = 1

    return np.array(binarized_vect)

def read_iovc():
    """
    Lit l'iovc des bases de VSUMM
    """
    for base in bases:
        iovc_videos = pd.read_json('./vsumm-{}-iovc.json'.format(base), lines=True)
        avg_avg_selected = []
        avg_avg_not_selected = []
        for video_id in iovc_videos.columns:
            iovc_video = iovc_videos[video_id].iloc[0]
            binarized_vect = get_frames_importance(video_id.split('.')[0], base=base)
            binarized_vect = binarized_vect[:len(iovc_video)]

            selected_indices = np.where(binarized_vect == 1)[0].tolist()
            for idx in selected_indices:
                binarized_vect[idx-12:idx+12] = 1

            selected_indices = np.where(binarized_vect == 1)[0].tolist()
            not_selected_indices = np.where(binarized_vect == 0)[0].tolist()
            

            avg_selected = np.average(np.array(iovc_video).take(selected_indices))
            avg_not_selected = np.average(np.array(iovc_video).take(not_selected_indices))
            avg_avg_selected.append(avg_selected)
            avg_avg_not_selected.append(avg_not_selected)
        print(base, ":")
        print("Moyenne d'IOVC sur les frames sélectionnées :", np.average(avg_avg_selected))
        print("Moyenne d'IOVC sur les frames non sélectionnées :", np.average(avg_avg_not_selected))
        


def read_memorability():
    """
    Lit la mémorabilité des bases de VSUMM
    """
    for base in bases:
        mems = pd.read_csv('./Memorability/vsumm-{}-mem-score.csv'.format(base), sep=';', header=0)
        avg_avg_selected = []
        avg_avg_not_selected = []
        for index in range(len(mems)):
            video_id = mems.loc[index, 'video_name'].split('.')[0]
            binarized_vect = get_frames_importance(video_id, base=base)
            memorability = list(map(float, mems.iloc[index]['memorability_scores'].split(',')))
            binarized_vect = binarized_vect[:len(memorability)]

            mask = binarized_vect == 1
            indices = np.where(mask)[0]

            for idx in indices:
                binarized_vect[idx-12:idx+12] = 1

            selected_indices = np.where(binarized_vect == 1)[0].tolist()
            not_selected_indices = np.where(binarized_vect == 0)[0].tolist()
            
            avg_selected = np.average(np.array(memorability).take(selected_indices))
            avg_not_selected = np.average(np.array(memorability).take(not_selected_indices))
            avg_avg_selected.append(avg_selected)
            avg_avg_not_selected.append(avg_not_selected)
        print(base, ":")
        print("Moyenne de mémorabilité sur les frames sélectionnées :", np.average(avg_avg_selected))
        print("Moyenne de mémorabilité sur les frames non sélectionnées :", np.average(avg_avg_not_selected))
        

if __name__ == "__main__":
    read_memorability()
    read_iovc()