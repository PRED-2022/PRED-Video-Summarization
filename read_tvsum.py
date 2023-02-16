"""
    Script pour lire les données inférées
    et la vérité-terrain de TVSum
"""
import math
from os import path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from tslearn.barycenters import dtw_barycenter_averaging
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import csv
from tslearn.clustering import TimeSeriesKMeans
from rich.progress import Progress
import cv2
from scipy.stats import wasserstein_distance

# Jeu de données TVSum
# Source : http://people.csail.mit.edu/yalesong/tvsum/
# Dossiers : ./tvsum/
#                    video/ : les vidéos annotées
#                    data/ : annotations de l'importance des images des vidéos selon 20 observateurs
#                            Importance entre 1 (basse) et 5 (forte)

def get_video_framerate(video_id):
    """
    Retourne la fréquence d'images de la vidéo.
    
    Parameters
    ----------
    video_id : identifiant de la vidéo
    """
    return cv2.VideoCapture("./tvsum/video/{}.mp4".format(video_id)).get(cv2.CAP_PROP_FPS)

def get_frames_importance(VIDEO_ID):
    """
    Retourne la vérité terrain de la vidéo.
    
    Parameters
    ----------
    VIDEO_ID : identifiant de la vidéo
    """
    ground_truth = pd.read_csv('./tvsum/data/ydata-tvsum50-anno.tsv', sep='\t', header=None)
    ground_truth.columns = ['id', 'category', 'importance']

    video_ground_truth = ground_truth[ground_truth['id'] == VIDEO_ID]['importance']

    annotations_list = []

    for importances_observer_i in video_ground_truth:
        importances_observer_i = list(map(int, importances_observer_i.split(',')))
        annotations_list.append(importances_observer_i)

    return np.mean(annotations_list, axis=0)

def get_frames_summary(VIDEO_KEY):
    """
    Retourne le résumé vidéo de la vidéo.
    
    Parameters
    ----------
    VIDEO_KEY : clé de la vidéo
    """
    summaries = pd.read_csv('./TVSum-PGLSUM-summary.csv', sep=';', header=0)
    summary_row = summaries[summaries['video_key'] == 'video_' + str(VIDEO_KEY)]
    if summary_row.empty:
        return []
    else:
        summary_row = summary_row.summary.iloc[0]
        return np.array(list(map(int, summary_row.split(','))))


def get_frames_iovc(VIDEO_ID):
    """
    IOVC de la vidéo.
    
    Parameters
    ----------
    VIDEO_ID : id de la vidéo
    """
    iovc_videos = pd.read_json("./TVSum-iovc.json", lines=True)
    return iovc_videos[VIDEO_ID + '.mp4'].iloc[0]

def get_frames_memorability(VIDEO_ID):
    """
    Mémorabilité de la vidéo.
    
    Parameters
    ----------
    VIDEO_ID : id de la vidéo
    """
    mems = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0)
    mems_of_video = mems[mems['video_name'] == VIDEO_ID + '.mp4']

    if mems_of_video.empty:
        return []
    else:
        mems_of_video = mems_of_video.memorability_scores.iloc[0]
        return list(map(float, mems_of_video.split(',')))


def get_all_videos_id():
    """
    Retourne les identifiants des vidéos de la base
    """
    videos_infos = pd.read_csv('./tvsum/data/ydata-tvsum50-info.tsv', sep='\t', header=0)
    print(len(videos_infos['video_id']))
    return videos_infos['video_id']

def get_all_frames_importance():
    """
    Ecrit dans un fichier CSV les vérités terrains
    de chaque vidéo de la base
    """
    ground_truth = pd.read_csv('./tvsum/data/ydata-tvsum50-anno.tsv', sep='\t', header=None)
    ground_truth.columns = ['id', 'category', 'importance']
    with open('./TVSum-groundtruth.csv', 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['id', 'importance'])

        for video_id in get_all_videos_id():
            video_ground_truth = ground_truth[ground_truth['id'] == video_id]['importance']

            annotations_list = []

            for importances_observer_i in video_ground_truth:
                importances_observer_i = list(map(int, importances_observer_i.split(',')))
                annotations_list.append(importances_observer_i)

            writer.writerow([video_id, ','.join(map(str, np.mean(annotations_list, axis=0)))])


def read_correlations():
    """
    Teste les corrélations linéaires de mémorabilité et d'IOVC
    """
    df = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0, usecols=['video_name', 'memorability_scores'])

    for index in range(len(df)):
        video_id = df.loc[index, 'video_name'].split('.')[0]
        groundtruth = get_frames_importance(video_id)
        memorability = list(map(float, df.iloc[index]['memorability_scores'].split(',')))
        iovc = get_frames_iovc(video_id)
        if len(groundtruth) != len(memorability):
            memorability = memorability[:len(groundtruth)]
            iovc = iovc[:len(groundtruth)]

        df.loc[index, 'pearson_mem'] = stats.pearsonr(memorability, groundtruth).statistic
        df.loc[index, 'spearman_mem'] = stats.spearmanr(memorability, groundtruth).correlation
        df.loc[index, 'pearson_iovc'] = stats.pearsonr(iovc, groundtruth).statistic
        df.loc[index, 'spearman_iovc'] = stats.spearmanr(iovc, groundtruth).correlation

    bins = np.arange(-1, 1, .1)

    plt.subplot(2, 2, 1)
    plt.hist(df['spearman_mem'], bins=bins)
    plt.xlabel('Mémorabilité - Corrélation de Spearman')
    plt.ylabel('Nombre de vidéos')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)

    plt.subplot(2, 2, 2)
    plt.hist(df['pearson_mem'], bins=bins, color="orange")
    plt.xlabel('Mémorabilité - Corrélation de Pearson')
    plt.ylabel('Nombre de vidéos')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)

    plt.subplot(2, 2, 3)
    plt.hist(df['spearman_iovc'], bins=bins)
    plt.xlabel('IOVC - Corrélation de Spearman')
    plt.ylabel('Nombre de vidéos')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)

    plt.subplot(2, 2, 4)
    plt.hist(df['pearson_iovc'], bins=bins, color="orange")
    plt.xlabel('IOVC - Corrélation de Pearson')
    plt.ylabel('Nombre de vidéos')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)

    print("Mémorabilité - Moyenne des corrélations : Pearson = {} - Spearman = {}".format(np.mean(df['pearson_mem']), np.mean(df['spearman_mem'])))
    print("IOVC - Moyenne des corrélations : Pearson = {} - Spearman = {}".format(np.mean(df['pearson_iovc']), np.mean(df['spearman_iovc'])))
    plt.show()


def save_boxplot(video_id, data, feature_name, bottom=-1, top=-1):
    """
    Sauvegarde les boîtes à moustaches
    dans un dossier (à créer) : ./Figures/Boxplots/TVSum/{feature_name}/{video_id}
    """
    if top != -1 and bottom != -1:
        plt.ylim(top=top, bottom=bottom)
    elif top != -1:
        plt.ylim(top=top)
    elif bottom != -1:
        plt.ylim(bottom=bottom)
    plt.boxplot(data, labels=["{} - {}".format(feature_name, video_id)])
    plt.savefig('./Figures/Boxplots/TVSum/{}/{}.jpg'.format(feature_name, video_id))
    plt.clf()

def save_plot(video_id, data, feature_name):
    """
    Sauvegarde le graphique d'évolution de la caractéristique
    dans un dossier (à créer) : ./Figures/Plots/TVSum/{feature_name}
    """
    framerate = get_video_framerate(video_id)
    nb_seconds = len(data) / framerate
    plt.xlabel("Video in seconds")
    plt.ylabel(feature_name)
    plt.xticks(ticks=np.arange(0, len(data), 30 * framerate), labels=[str(int(n)) for n in np.arange(0, nb_seconds, 30)], fontsize=8)
    plt.plot(data)
    plt.savefig('./Figures/Plots/TVSum/{}/{}.jpg'.format(feature_name, video_id))
    plt.clf()


"""
Paramètre à activer/désactiver pour tester :
    le clustering T-SNE : DIM_REDUCTIONS_TEST
    test de variance : ANOVA_TEST
    la similarité : WASSERSTEIN_TEST
    si l'on souhaite sauvegarder les figures : SAVE_FIG
"""

WASSERSTEIN_TEST = "IOVC" # ou "Memorability" ou False pour ne pas le faire
ANOVA_TEST = False
READ_CORRELATIONS = False
DIM_REDUCTIONS_TEST = False
SAVE_FIG = False

if __name__ == "__main__":
    if WASSERSTEIN_TEST:
        videos_id = get_all_videos_id()
        # Création d'un CSV / matrice des distances entre les pairs de distribution d'IOVC de vidéos
        if not path.isfile(f'./TVSum-{WASSERSTEIN_TEST}-Wasserstein.csv'):
            with open(f'./TVSum-{WASSERSTEIN_TEST}-Wasserstein.csv', 'w', newline='') as csv_file:
                dist_matrix = []
                writer = csv.writer(csv_file, delimiter=';')
                with Progress() as progress:
                    video_task = progress.add_task("[red]Boucle video...", total=len(videos_id))
                    writer.writerow(pd.concat([pd.Series(["videos_name"]), videos_id]))
                    for a in range(len(videos_id)):
                        progress.advance(video_task)
                        video_a = videos_id[a]
                        iovc_a = get_frames_iovc(video_a) if WASSERSTEIN_TEST == "IOVC" else get_frames_memorability(video_a)
                        dist_matrix.append([])
                        video_analyse_task = progress.add_task("[blue]Calcul video : " + video_a, total=len(videos_id))
                        for b in range(len(videos_id)):
                            progress.advance(video_analyse_task)
                            if a != b:
                                video_b = videos_id[b]
                                iovc_b = get_frames_iovc(video_b) if WASSERSTEIN_TEST == "IOVC" else get_frames_memorability(video_b)
                                w_dt = wasserstein_distance(iovc_a, iovc_b)
                                dist_matrix[a].append(w_dt)
                            else:
                                dist_matrix[a].append("")
                        writer.writerow([video_a] + dist_matrix[a])

    if ANOVA_TEST:
        all_groundtruth = []
        all_memorability = []
        all_summary = []
        all_iovc = []

        videos_id = get_all_videos_id()

        if path.isfile('./TVSum-average.csv'):
            df = pd.read_csv('./TVSum-average.csv', sep=';', header=0)
            max_iovc = df.iloc[df.iovc_avg.argmax(), 0]
            min_iovc = df.iloc[df.iovc_avg.argmin(), 0]
            max_memorability = df.iloc[df.memorability_avg.argmax(), 0]
            min_memorability = df.iloc[df.memorability_avg.argmin(), 0]
            max_groundtruth = df.iloc[df.groundtruth_avg.argmax(), 0]
            min_groundtruth = df.iloc[df.groundtruth_avg.argmin(), 0]

            print("IOVC -> max : {} | min : {}".format(max_iovc, min_iovc))
            print("Memorability -> max : {} | min : {}".format(max_memorability, min_memorability))
            print("Groundtruth -> max : {} | min : {}".format(max_groundtruth, min_groundtruth))
        else:
            with open('./TVSum-average.csv', 'w', newline='') as csv_file:
                writer = csv.writer(csv_file, delimiter=';')
                writer.writerow(['video_id', 'groundtruth_avg', 'memorability_avg', 'iovc_avg', 'summary_avg'])
                with Progress() as progress:
                    video_task = progress.add_task("[red]Boucle video...", total=len(videos_id))
                    for index in range(len(videos_id)):
                        progress.advance(video_task)
                        video_id = videos_id[index]

                        groundtruth = get_frames_importance(video_id)
                        all_groundtruth.append(np.mean(groundtruth))

                        summary = get_frames_summary(index + 1)
                        all_summary.append(np.mean(summary))

                        memorability = get_frames_memorability(video_id)
                        if len(memorability) > len(groundtruth):
                            memorability = memorability[:len(groundtruth)]

                        all_memorability.append(np.mean(memorability))

                        iovc = get_frames_iovc(video_id)
                        iovc = iovc[:len(groundtruth)]
                        all_iovc.append(np.mean(iovc))

                        if SAVE_FIG:
                            save_boxplot(video_id, memorability, 'Memorability', 0.4, 1)
                            save_boxplot(video_id, groundtruth, 'Groundtruth', 1, 5)
                            save_boxplot(video_id, iovc, 'IOVC', 0, 4)
                            save_plot(video_id, memorability, 'Memorability')
                            save_plot(video_id, groundtruth, 'Groundtruth')
                            save_plot(video_id, iovc, 'IOVC')

                        writer.writerow([video_id, np.mean(groundtruth), np.mean(memorability), np.mean(iovc), np.mean(summary)])

            plt.ylabel('Nombre de vidéos')

            plt.xlabel('Vérité-terrain moyenne')
            plt.hist(all_groundtruth)
            plt.show()

            plt.hist(all_memorability)
            plt.xlabel('Mémorabilité moyenne')
            plt.show()

            plt.hist(all_iovc)
            plt.xlabel('IOVC moyen')
            plt.show()

    if READ_CORRELATIONS:
        read_correlations()
    if DIM_REDUCTIONS_TEST:
        # Taille de la fenêtre vidéo en seconde :
        WINDOW_IN_SECOND = 25
        # Pas entre deux fenêtres, si WINDOW >= STRIDE -> pas de superposition de fenêtres
        STRIDE_IN_SECOND = 5

        all_frames_importances = []
        all_frames_memorability = []
        all_frames_iovc = []

        videos_id = get_all_videos_id()
        for index in range(len(videos_id)):
            video_id = videos_id[index]
            frames_importance = get_frames_importance(video_id)
            all_frames_importances.append(frames_importance)

            frames_memorability = get_frames_memorability(video_id)
            frames_iovc = get_frames_iovc(video_id)
            if len(frames_memorability) > len(frames_importance):
                frames_memorability = frames_memorability[:len(frames_importance)]
            if len(frames_memorability) > len(frames_importance):
                frames_iovc = frames_iovc[:len(frames_importance)]

            if SAVE_FIG:
                plt.plot(frames_memorability)
                plt.xlabel("Frames de la vidéo")
                plt.ylabel("Mémorabilité")
                plt.ylim(top=1, bottom=0.4)
                plt.savefig("./Figures/Memorability/{}.png".format(video_id))
                plt.clf()
            # lissage de la mémorabilité qui oscille beaucoup
            frames_memorability = savgol_filter(frames_memorability, 72, 3).tolist()
            all_frames_memorability.append(frames_memorability)
            frames_iovc = savgol_filter(frames_iovc, 72, 3).tolist()
            all_frames_iovc.append(frames_iovc)
            if SAVE_FIG:
                plt.plot(frames_memorability)
                plt.xlabel("Frames de la vidéo")
                plt.ylabel("Mémorabilité")
                plt.ylim(top=1, bottom=0.4)
                plt.savefig("./Figures/Memorability/{}_filtered.png".format(video_id))
                plt.clf()

        x = np.concatenate(all_frames_importances)
        indices = np.where(x >= 3)
        y = np.concatenate(all_frames_memorability)
        plt.scatter(x[indices], y[indices], alpha=0.8, marker='.')
        a, b = np.polyfit(x[indices], y[indices], 1)
        plt.plot(x[indices], a * x[indices] + b, color='red')

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
        windows_iovc = []
        for video_index in range(len(all_frames_importances)):
            # Vérité-terrain d'une vidéo (score de chaque frame)
            frames_importances = all_frames_importances[video_index]

            # Mémorabilité des frames d'une vidéo
            frames_memorability = all_frames_memorability[video_index]

            # IOVC des frames d'une vidéo
            frames_iovc = all_frames_iovc[video_index]

            # Nombre de frames de la vidéo = nombre de scores d'importance
            nb_frames_video = len(frames_importances)

            for l_window in range(0, len(frames_importances), 24 * STRIDE_IN_SECOND):
                r_window = l_window + 24 * WINDOW_IN_SECOND
                if r_window > len(frames_importances):
                    break
                else:
                    windows_importance.append(frames_importances[l_window:r_window])
                    windows_memorability.append(frames_memorability[l_window:r_window])
                    windows_iovc.append(frames_iovc[l_window:r_window])

        """
            Réduction de dimensions (ACP et T-SNE)
        """
        """
            Analyse en composantes principales
        """

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
        """
            T-SNE
        """

        tsne = TSNE(n_components=2, random_state=0)
        tsne_data = tsne.fit_transform(np.array(windows_memorability))
        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], marker=".", alpha=0.5)
        plt.show()

        """
            Clustering KMeans
        """

        cluster_count = 3

        kmeans = TimeSeriesKMeans(n_clusters=cluster_count, metric="dtw", max_iter=10)

        labels = kmeans.fit_predict(windows_iovc)
        plot_count = math.ceil(math.sqrt(cluster_count))

        plt.scatter(tsne_data[:, 0], tsne_data[:, 1], marker=".", alpha=0.5, c=labels)
        plt.show()

        fig, axs = plt.subplots(plot_count, plot_count, figsize=(15, 15))
        fig.suptitle('Clusters')
        row_i = 0
        column_j = 0
        for label in set(labels):
            cluster = []
            for i in range(len(labels)):
                if (labels[i] == label):
                    axs[row_i, column_j].plot(windows_memorability[i], c="gray", alpha=0.4)
                    cluster.append(windows_memorability[i])
            if len(cluster) > 0:
                axs[row_i, column_j].plot(dtw_barycenter_averaging(np.vstack(cluster)), c="blue")
                axs[row_i, column_j].plot(np.mean(np.vstack(cluster), axis=0), c="red", alpha=0.8)
            axs[row_i, column_j].set_title("Cluster " + str((row_i * plot_count) + column_j))
            column_j += 1
            if column_j % plot_count == 0:
                row_i += 1
                column_j = 0

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
