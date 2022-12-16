import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import ceil

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


def get_all_videos_id():
    videos_infos = pd.read_csv('./tvsum/data/ydata-tvsum50-info.tsv', sep='\t', header=0)
    return videos_infos['video_id']


if __name__ == "__main__":

    all_frames_importances = []

    for video_id in get_all_videos_id():
        frames_importance = get_frames_importance(video_id)
        nb_frames_video = len(frames_importance)
        # print(nb_frames_video)
        all_frames_importances.append(frames_importance)

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

        # Nombre de frames de la vidéo = nombre de scores d'importance
        nb_frames_video = len(frames_importances)
        
        # Combien de fois il faut répéter chaque frame pour atteindre STRETCH_TO_N_FRAMES :
        repeat_n_times = ceil(STRETCH_TO_N_FRAMES / nb_frames_video)
        # On répète les (scores des) frames pour atteindre STRETCH_TO_N_FRAMES :
        all_frames_importances[video_index] = np.repeat(frames_importances, repeat_n_times)
        
        loss = (len(all_frames_importances[video_index])-STRETCH_TO_N_FRAMES) / len(all_frames_importances[video_index])
        losses.append(loss)

        # On enlève les frames >= STRETCH_TO_N_FRAMES (= perte) :
        all_frames_importances[video_index] = all_frames_importances[video_index][:STRETCH_TO_N_FRAMES]

    print("Pertes moyenne : {:.2%}, écart-type : {:.2%}. Perte minimale : {:.2%}, maximale : {:.2%}".format(np.average(losses), np.std(losses), np.min(losses), np.max(losses)))

    """
        Application d'une méthode de clustering : K-Means
    """

    kmeans = KMeans(n_clusters=3, random_state=0).fit(all_frames_importances)
    print(kmeans.labels_)
    for center in kmeans.cluster_centers_:
        plt.plot(center)
    
    plt.xlabel("Frames de la vidéo au cours du temps")
    plt.ylabel("Scores d'importance")

    plt.show()