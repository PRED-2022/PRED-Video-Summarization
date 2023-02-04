from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from math import ceil, isnan


def save_hist(data, feature):
    plt.hist(data, bins=np.arange(0, np.max(data)+0.01, 0.01))
    plt.xlabel("P-Valeur")
    plt.ylabel("Nombre de vidéos")
    plt.title("Test de Student - {}".format(feature))
    plt.savefig('./Figures/Plots/TVSum/T-Test/{}.jpg'.format(feature))
    plt.clf()

def save_hist_correlation(data, feature, correlation):
    bins = np.arange(-1, 1, .1)
    plt.hist(data, bins=bins)
    plt.xlabel('{} - Corrélation de {}'.format(feature, correlation))
    plt.ylabel('Nombre de vidéos')
    plt.xlim(left=-1, right=1)
    plt.ylim(top=8)
    plt.savefig('./Figures/Plots/TVSum/Correlation/{}_{}.jpg'.format(correlation, feature))
    plt.clf()

# Ground Truth
groundtruth_videos = pd.read_csv('./TVSum-groundtruth.csv', sep=';', header=0).set_index('id')

# IOVC
iovc_videos = pd.read_json("./TVSum-iovc.json", lines=True)

# Emotions
emotion_videos = pd.read_json("./PROCESSED-TVsum-face-intensity.json", lines=True)

# Memorability
memorability_videos = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0).set_index("video_name")

ttests_iovc = []
ttests_mem = []
ttests_nbr_face = []
ttests_max_proba = []
ttests_happy = []
ttests_angry = []
ttests_disgust = []
ttests_neutral = []
ttests_fear = []
ttests_sad = []
ttests_surprise = []
pearson_iovc = []
pearson_mem = []
spearman_iovc = []
spearman_mem = []

for key in iovc_videos.keys():
   
    score_gt = np.array(groundtruth_videos.loc[key.replace(".mp4", ""), "importance"].split(",")).astype(float)
    score_iovc = np.array(iovc_videos[key].iloc[0], dtype=float)
    score_mem = np.array(memorability_videos.loc[key, "memorability_scores"].split(",")).astype(float)

    df_emotions = pd.DataFrame(list(filter(lambda x: x is not None, emotion_videos[key].iloc[0])))

    nb_frames = len(score_gt)

    score_gt = pd.Series(score_gt)
    score_iovc = pd.Series(score_iovc[:nb_frames])
    score_mem = pd.Series(score_mem[:nb_frames])
    df_emotions = df_emotions[:nb_frames]

    """
        Sélection des frames retenues pour le résumé vidéo
        selon la vérité-terrain
    """
    # on garde au moins 10 % des meilleures frames en terme d'importance
    nb_selected_frames = ceil(0.10*nb_frames)
    min_importance_for_selection = score_gt.sort_values(ascending=False).iloc[nb_selected_frames]
    selected_frames = score_gt[score_gt >= min_importance_for_selection]
    not_selected_frames = score_gt[score_gt < min_importance_for_selection]

    selected_indices = list(selected_frames.index.values)
    not_selected_indices = list(not_selected_frames.index.values)

    # print("Taille résumé vidéo {} : {:.1f}%".format(key, len(selected_indices)/len(not_selected_indices)*100))

    """
        Test de Student (t-test) :
        Entre les scores des caractéristiques pour les
        frames incluses et les scores de celles non incluses
        dans le résumé vidéo (vérité-terrain)
    """
    selected_iovc, not_selected_iovc = np.array(score_iovc).take(selected_indices), np.array(score_iovc).take(not_selected_indices)
    selected_mem, not_selected_mem = np.array(score_mem).take(selected_indices), np.array(score_mem).take(not_selected_indices)
    selected_nbr_face, not_selected_nbr_face = np.array(df_emotions["nbr_face"]).take(selected_indices), np.array(df_emotions["nbr_face"]).take(not_selected_indices)
    selected_max_proba, not_selected_max_proba = np.array(df_emotions["max_proba"]).take(selected_indices), np.array(df_emotions["max_proba"]).take(not_selected_indices)
    selected_happy, not_selected_happy = np.array(df_emotions["happy"]).take(selected_indices), np.array(df_emotions["happy"]).take(not_selected_indices)
    selected_angry, not_selected_angry = np.array(df_emotions["angry"]).take(selected_indices), np.array(df_emotions["angry"]).take(not_selected_indices)
    selected_disgust, not_selected_disgust = np.array(df_emotions["disgust"]).take(selected_indices), np.array(df_emotions["disgust"]).take(not_selected_indices)
    selected_neutral, not_selected_neutral = np.array(df_emotions["neutral"]).take(selected_indices), np.array(df_emotions["neutral"]).take(not_selected_indices)
    selected_fear, not_selected_fear = np.array(df_emotions["fear"]).take(selected_indices), np.array(df_emotions["fear"]).take(not_selected_indices)
    selected_sad, not_selected_sad = np.array(df_emotions["sad"]).take(selected_indices), np.array(df_emotions["sad"]).take(not_selected_indices)
    selected_surprise, not_selected_surprise = np.array(df_emotions["surprise"]).take(selected_indices), np.array(df_emotions["surprise"]).take(not_selected_indices)

    """
        Test de corrélation linéaire
        Pearson & Spearman
    """
    pearson_iovc.append(stats.pearsonr(selected_frames, selected_iovc).statistic)
    pearson_mem.append(stats.pearsonr(selected_frames, selected_mem).statistic)

    spearman_iovc.append(stats.spearmanr(selected_frames, selected_iovc).correlation)
    spearman_mem.append(stats.spearmanr(selected_frames, selected_mem).correlation)

    ttests_iovc.append(stats.ttest_ind(selected_iovc, not_selected_iovc).pvalue)
    ttests_mem.append(stats.ttest_ind(selected_mem, not_selected_mem).pvalue)
    ttests_nbr_face.append(stats.ttest_ind(selected_nbr_face, not_selected_nbr_face).pvalue)
    ttests_max_proba.append(stats.ttest_ind(selected_max_proba, not_selected_max_proba).pvalue)
    ttests_happy.append(stats.ttest_ind(selected_happy, not_selected_happy).pvalue)
    ttests_angry.append(stats.ttest_ind(selected_angry, not_selected_angry).pvalue)
    ttests_disgust.append(stats.ttest_ind(selected_disgust, not_selected_disgust).pvalue)
    ttests_neutral.append(stats.ttest_ind(selected_neutral, not_selected_neutral).pvalue)
    ttests_fear.append(stats.ttest_ind(selected_fear, not_selected_fear).pvalue)
    ttests_sad.append(stats.ttest_ind(selected_sad, not_selected_sad).pvalue)
    ttests_surprise.append(stats.ttest_ind(selected_surprise, not_selected_surprise).pvalue)

save_hist(ttests_iovc, "IOVC")
save_hist(ttests_mem, "Memorabilité")
save_hist(ttests_nbr_face, "nbr_face")
save_hist(ttests_max_proba, "max_proba")
save_hist(ttests_happy, "happy")
save_hist(ttests_angry, "angry")
save_hist(ttests_disgust, "disgust")
save_hist(ttests_neutral, "neutral")
save_hist(ttests_fear, "fear")
save_hist(ttests_sad, "sad")
save_hist(ttests_surprise, "surprise")

save_hist_correlation(pearson_iovc, "IOVC", "Pearson")
save_hist_correlation(pearson_mem, "Memorabilité", "Pearson")
save_hist_correlation(spearman_iovc, "IOVC", "Spearman")
save_hist_correlation(spearman_mem, "Memorabilité", "Spearman")