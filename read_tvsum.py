import math
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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


def read_correlations():
    mems = pd.read_csv('./tv-sum-mem-score.csv', sep=';', header=0)

    for index in range(len(mems)):
        video_id = mems.loc[index, 'video_name'].split('.')[0]
        groundtruth = get_frames_importance(video_id)
        memorability = list(map(float, mems.iloc[index]['memorability_scores'].split(',')))
        if len(groundtruth) != len(memorability):
            memorability = memorability[:len(groundtruth)]

        memorability_per_sec = []
        gt_per_sec = []
        n_frames = 1
        for i in range(0, len(memorability), n_frames):
            memorability_per_sec.append(np.average(memorability[i:i+n_frames]))
            gt_per_sec.append(np.average(groundtruth[i:i+n_frames]))

        mems.loc[index, 'pearson'] = stats.pearsonr(memorability_per_sec, gt_per_sec).statistic
        mems.loc[index, 'spearman'] = stats.spearmanr(memorability_per_sec, gt_per_sec).correlation

    pearsons = mems['pearson']
    spearmans = mems['spearman']

    index_max_corr = np.argmax(spearmans)
    print(index_max_corr)
    mem_max = list(map(float, mems.loc[index_max_corr, 'memorability_scores'].split(',')))
    plt.plot(mem_max, c='blue')
    plt.plot(savgol_filter(mem_max, 72, 3), c='orange')
    plt.show()

    bins = np.arange(-1, 1, .1)

    plt.subplot(1, 2, 1)
    plt.hist(spearmans, bins=bins)
    plt.xlabel('Corrélation de Spearman')
    plt.ylabel('Nombre de vidéos')
    avg_spearman = np.average(mems['spearman'])
    plt.axvline(x=avg_spearman, color='black', linestyle='--')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)

    plt.subplot(1, 2, 2)
    plt.hist(pearsons, bins=bins, color="orange")
    plt.xlabel('Corrélation de Pearson')
    plt.ylabel('Nombre de vidéos')
    avg_pearson = np.average(mems['pearson'])
    plt.axvline(x=avg_pearson, color='black', linestyle='--')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)

    print("Moyenne des corrélations absolues : Pearson = {} - Spearman = {}".format(np.average(np.abs(mems['pearson'])), np.average(np.abs(mems['spearman']))))
    print("Moyenne des corrélations : Pearson = {} - Spearman = {}".format(avg_pearson, avg_spearman))
    plt.show()


READ_CORRELATIONS = True
DIM_REDUCTIONS_TEST = False

if __name__ == "__main__":
    if READ_CORRELATIONS:
        read_correlations()
    elif DIM_REDUCTIONS_TEST:
        # Taille de la fenêtre vidéo en seconde :
        WINDOW_IN_SECOND = 10
        # Pas entre deux fenêtres, si WINDOW >= STRIDE -> pas de superposition de fenêtres
        STRIDE_IN_SECOND = 5

        all_frames_importances = []
        all_frames_memorability = []

        videos_id = get_all_videos_id()
        for index in range(len(videos_id)):
            video_id = videos_id[index]
            frames_importance = get_frames_importance(video_id)
            all_frames_importances.append(frames_importance)

            frames_memorability = get_frames_memorability(video_id)
            if len(frames_memorability) > len(frames_importance):
                frames_memorability = frames_memorability[:len(frames_importance)]
            # lissage de la mémorabilité qui oscille beaucoup
            frames_memorability = savgol_filter(frames_memorability, 72, 3)
            all_frames_memorability.append(frames_memorability)

        x = np.concatenate(all_frames_importances)
        indices = np.where(x >= 3)
        y = np.concatenate(all_frames_memorability)
        plt.scatter(x[indices], y[indices], alpha=0.8, marker='.')
        a, b = np.polyfit(x[indices], y[indices], 1)
        plt.plot(x[indices], a*x[indices]+b, color='red')

        #coefficients = np.polyfit(x[indices], y[indices], 2)
        #x_reg = np.linspace(x[indices].min(), x[indices].max(), 25)
        #y_reg = np.polyval(coefficients, x_reg)
        #plt.plot(x_reg, y_reg, color='red')

        plt.xlabel("Importance (vérité-terrain)")
        plt.ylabel("Mémorabilité")
        plt.show()

        """
            Fenêtrage des vidéos pour pouvoir les comparer
            (vidéos de différentes longueurs)
        """

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
            Réduction de dimensions (ACP et T-SNE)
        """
        """
            Analyse en composantes principales
        """
        pca = PCA(n_components=2)

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

        pc_x = 0
        pc_y = 1
        xs = data_2d[:,pc_x]
        ys = data_2d[:,pc_y]

        coeff = pca.components_[pc_x:pc_y+1, :].T
        n = coeff.shape[0]

        scalex = 1.0/(xs.max() - xs.min())
        scaley = 1.0/(ys.max() - ys.min())

        fig, ax = plt.subplots()
        ax.scatter(data_2d[:,pc_x]*scalex, data_2d[:,pc_y]*scaley, marker=".", alpha=0.5)
        ax.set_xlabel("PC{}".format(pc_x+1))
        ax.set_ylabel("PC{}".format(pc_y+1))

        print(len(data_2d))

        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1], color='r', alpha=0.005, head_width=0.003)

        plt.show()

        """
            T-SNE
        """
        tsne = TSNE(n_components=2, random_state=0)
        tsne_data = tsne.fit_transform(np.array(windows_memorability))
        plt.scatter(tsne_data[:,0], tsne_data[:,1], marker=".", alpha=0.5)
        plt.show()


        """
            Clustering KMeans
        """

        cluster_count = 9
        kmeans = KMeans(n_clusters=cluster_count)

        labels = kmeans.fit_predict(tsne_data)
        plot_count = math.ceil(math.sqrt(cluster_count))

        fig, axs = plt.subplots(plot_count,plot_count,figsize=(15,15))
        fig.suptitle('Clusters')
        row_i=0
        column_j=0
        for label in set(labels):
            cluster = []
            for i in range(len(labels)):
                    if(labels[i]==label):
                        axs[row_i, column_j].plot(windows_memorability[i],c="gray",alpha=0.4)
                        cluster.append(windows_memorability[i])
            if len(cluster) > 0:
                axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)),c="blue")
                axs[row_i, column_j].plot(np.average(np.vstack(cluster), axis=0), c="red", alpha=0.8)
            axs[row_i, column_j].set_title("Cluster "+str((row_i*plot_count)+column_j))
            column_j+=1
            if column_j%plot_count == 0:
                row_i+=1
                column_j=0
                
        plt.show()


        """
            Application d'une méthode de clustering : K-Means
        """
        """
        kmeans_groundtruth = KMeans(n_clusters=3, random_state=0).fit(windows_importance)
        print(kmeans_groundtruth.labels_)
        for center in kmeans_groundtruth.cluster_centers_:
            plt.plot(center)

        plt.xlabel("Fenêtre de la vidéo")
        plt.ylabel("Scores d'importance")
        plt.savefig('groundtruth-tvsum-clusters.png')
        
        plt.clf()
        
        if len(frames_memorability) > 0:
            kmeans_memorability = KMeans(n_clusters=3, random_state=0).fit(windows_memorability)
            print(kmeans_memorability.labels_)
            for center in kmeans_memorability.cluster_centers_:
                plt.plot(center)

            plt.xlabel("Fenêtre de la vidéo")
            plt.ylabel("Memorabilité")
            plt.savefig('memorability-tvsum-clusters.png')
        """