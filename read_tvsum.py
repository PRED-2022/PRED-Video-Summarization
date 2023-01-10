from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from math import ceil
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

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
    WINDOW_IN_SECOND = 30
    STRIDE_IN_SECOND = 10

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


    windows_importance = []
    windows_memorability = []
    for video_index in range(len(all_frames_importances)):
        # Vérité-terrain d'une vidéo (score de chaque frame)
        frames_importances = all_frames_importances[video_index]

        # Mémorabilité des frames d'une vidéo
        frames_memorability = all_frames_memorability[video_index]

        # Nombre de frames de la vidéo = nombre de scores d'importance
        nb_frames_video = len(frames_importances)

        for l_window in range(0, len(frames_importances), 24*STRIDE_IN_SECOND):
            r_window = l_window + 24*WINDOW_IN_SECOND
            if r_window > len(frames_importances):
                break
            else:
                windows_importance.append(frames_importances[l_window:r_window])
                windows_memorability.append(frames_memorability[l_window:r_window])
    

    """
        Analyse en composantes principales
    """
    pca = PCA(n_components=4)

    data_2d = pca.fit_transform(windows_memorability)

    x_bar = range(len(pca.explained_variance_))
    y_bar = list(pca.explained_variance_)

    y_cumulated = np.cumsum(y_bar)/sum(y_bar)*100

    fig, ax = plt.subplots()

    ax.bar(x_bar, y_bar, tick_label=['PC'+str(i) for i in range(1,len(pca.components_)+1)])
    ax.tick_params(axis="y", colors="C0")

    ax2 = ax.twinx()
    ax2.plot(x_bar, y_cumulated, color="C1", marker=".",)
    ax2.yaxis.set_major_formatter(PercentFormatter())
    ax2.tick_params(axis="y", colors="C1")

    ax.set_xlabel("Composantes principales")
    ax.set_ylabel("Variance expliquée")

    plt.show()

    for pc_x, pc_y in [[0,1],[2,3]]:

        xs = data_2d[:,pc_x]
        ys = data_2d[:,pc_y]

        coeff = pca.components_[pc_x:pc_y+1, :].T
        n = coeff.shape[0]

        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        fig, ax = plt.subplots()
        ax.scatter(data_2d[:][:,pc_x]*scalex, data_2d[:][:,pc_y]*scaley, marker=".", alpha=0.5)
        ax.set_xlabel("PC{}".format(pc_x+1))
        ax.set_ylabel("PC{}".format(pc_y+1))

        print(len(data_2d[:]))

        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.005, head_width=0.003)

        plt.show()


    """
        Application d'une méthode de clustering : K-Means
    """
    """
    kmeans_groundtruth = KMeans(n_clusters=3, random_state=0).fit(windows_importance)
    print(kmeans_groundtruth.labels_)
    for center in kmeans_groundtruth.cluster_centers_:
        plt.plot(center)

    plt.xlabel("Fenêtres")
    plt.ylabel("Scores d'importance")
    plt.savefig('groundtruth-tvsum-clusters.png')
    
    plt.clf()
    
    if len(frames_memorability) > 0:
        kmeans_memorability = KMeans(n_clusters=3, random_state=0).fit(windows_memorability)
        print(kmeans_memorability.labels_)
        for center in kmeans_memorability.cluster_centers_:
            plt.plot(center)

        plt.xlabel("Frames de la vidéo au cours du temps")
        plt.ylabel("Memorabilité")
        plt.savefig('memorability-tvsum-clusters.png')
    """