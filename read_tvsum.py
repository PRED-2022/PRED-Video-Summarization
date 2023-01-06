import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import ceil
from scipy.signal import savgol_filter

# Jeu de données TVSum
# Source : http://people.csail.mit.edu/yalesong/tvsum/
# Dossiers : ./tvsum/
#                    video/ : les vidéos annotées
#                    data/ : annotations de l'importance des images des vidéos selon 20 observateurs
#                            Importance entre 1 (basse) et 5 (forte)

def get_frames_importance(VIDEO_ID):
    ground_truth = pd.read_csv('./tvsum/data/ydata-tvsum50-anno.tsv', sep='\t', header=None)
    ground_truth.columns = ['id', 'category', 'importance']

    video_ground_truth = ground_truth[ground_truth['id'] == VIDEO_ID]['importance']

    annotations_list = []

    for importances_observer_i in video_ground_truth:
        importances_observer_i = list(map(int, importances_observer_i.split(',')))
        annotations_list.append(importances_observer_i)

    return np.average(annotations_list, axis=0)


def get_frames_memorability(VIDEO_ID):
    mems = pd.read_csv('./tv-sum-mem-score.csv', sep=';', header=0)
    mems_of_video = mems[mems['video_name'] == VIDEO_ID + '.mp4']

    if mems_of_video.empty:
        return []
    else:
        mems_of_video = mems_of_video.memorability_scores.iloc[0]
        return list(map(float, mems_of_video.split(',')))


def get_all_videos_id():
    videos_infos = pd.read_csv('./tvsum/data/ydata-tvsum50-info.tsv', sep='\t', header=0)
    print(len(videos_infos['video_id']))
    return videos_infos['video_id']


if __name__ == "__main__":

    all_frames_importances = []
    all_frames_memorability = {}

    haveMemorabilityScore = []
    videos_id = get_all_videos_id()
    for index in range(len(videos_id)):
        video_id = videos_id[index]
        frames_importance = get_frames_importance(video_id)
        all_frames_importances.append(frames_importance)

        frames_memorability = get_frames_memorability(video_id)
        hasMemorability = len(frames_memorability) > 0
        haveMemorabilityScore.append(hasMemorability)
        if hasMemorability:
            all_frames_memorability[index] = frames_memorability
    """
        Vidéos de différents durées donc on étire (répète) les scores de chaque frame
        des videos pour que toutes les vidéos aient une durée de STRETCH_TO_N_FRAMES
    """
    STRETCH_TO_N_FRAMES = 1000000

    # Pertes dues à l'étirement
    losses = []

    for video_index in range(len(all_frames_importances)):
        # Vérité-terrain d'une vidéo (score de chaque frame)
        frames_importances = all_frames_importances[video_index]
        if haveMemorabilityScore[video_index]:
            frames_memorability = all_frames_memorability[video_index]

        # Nombre de frames de la vidéo = nombre de scores d'importance
        nb_frames_video = len(frames_importances)
        
        # Combien de fois il faut répéter chaque frame pour atteindre STRETCH_TO_N_FRAMES :
        repeat_n_times = ceil(STRETCH_TO_N_FRAMES / nb_frames_video)
        # On répète les (scores des) frames pour atteindre STRETCH_TO_N_FRAMES :
        all_frames_importances[video_index] = np.repeat(frames_importances, repeat_n_times)
        if haveMemorabilityScore[video_index]:
            all_frames_memorability[video_index] = np.repeat(frames_memorability, repeat_n_times)

        loss = (len(all_frames_importances[video_index])-STRETCH_TO_N_FRAMES) / len(all_frames_importances[video_index])
        losses.append(loss)

        # On enlève les frames >= STRETCH_TO_N_FRAMES (= perte) :
        all_frames_importances[video_index] = all_frames_importances[video_index][:STRETCH_TO_N_FRAMES]
        if haveMemorabilityScore[video_index]:
            all_frames_memorability[video_index] = all_frames_memorability[video_index][:STRETCH_TO_N_FRAMES]

    print("Pertes moyenne : {:.2%}, écart-type : {:.2%}. Perte minimale : {:.2%}, maximale : {:.2%}".format(np.average(losses), np.std(losses), np.min(losses), np.max(losses)))

    """
        Application d'une méthode de clustering : K-Means
    """

    kmeans_groundtruth = KMeans(n_clusters=3, random_state=0).fit(all_frames_importances)
    print(kmeans_groundtruth.labels_)
    for center in kmeans_groundtruth.cluster_centers_:
        plt.plot(savgol_filter(center, window_length=100000, polyorder=3, mode="nearest"))

    plt.xlabel("Frames de la vidéo au cours du temps")
    plt.ylabel("Scores d'importance")
    plt.savefig('groundtruth-tvsum-clusters.png')
    
    plt.clf()
    
    if len(frames_memorability) > 0:
        kmeans_memorability = KMeans(n_clusters=3, random_state=0).fit(list(all_frames_memorability.values()))
        print(kmeans_memorability.labels_)
        for center in kmeans_memorability.cluster_centers_:
            plt.plot(savgol_filter(center, window_length=100000, polyorder=3, mode="nearest"))

        plt.xlabel("Frames de la vidéo au cours du temps")
        plt.ylabel("Memorabilité")
        plt.savefig('memorability-tvsum-clusters.png')
