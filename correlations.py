from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from math import ceil


def save_hist(data, feature):
    plt.hist(data, bins=np.arange(0, np.max(data)+0.01, 0.01))
    plt.xlabel("P-Valeur")
    plt.ylabel("Nombre de vidéos")
    plt.title("Test de Student - {}".format(feature))
    plt.savefig('./Figures/Plots/TVSum/T-Test/{}.jpg'.format(feature))
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
    ttest_iovc = stats.ttest_ind(np.array(score_iovc).take(selected_indices), np.array(score_iovc).take(not_selected_indices))
    ttest_mem = stats.ttest_ind(np.array(score_mem).take(selected_indices), np.array(score_mem).take(not_selected_indices))
    ttest_nbr_face = stats.ttest_ind(np.array(df_emotions["nbr_face"]).take(selected_indices), np.array(df_emotions["nbr_face"]).take(not_selected_indices))
    ttest_max_proba = stats.ttest_ind(np.array(df_emotions["max_proba"]).take(selected_indices), np.array(df_emotions["max_proba"]).take(not_selected_indices))
    ttest_happy = stats.ttest_ind(np.array(df_emotions["happy"]).take(selected_indices), np.array(df_emotions["happy"]).take(not_selected_indices))
    ttest_angry = stats.ttest_ind(np.array(df_emotions["angry"]).take(selected_indices), np.array(df_emotions["angry"]).take(not_selected_indices))
    ttest_disgust = stats.ttest_ind(np.array(df_emotions["disgust"]).take(selected_indices), np.array(df_emotions["disgust"]).take(not_selected_indices))
    ttest_neutral = stats.ttest_ind(np.array(df_emotions["neutral"]).take(selected_indices), np.array(df_emotions["neutral"]).take(not_selected_indices))
    ttest_fear = stats.ttest_ind(np.array(df_emotions["fear"]).take(selected_indices), np.array(df_emotions["fear"]).take(not_selected_indices))
    ttest_sad = stats.ttest_ind(np.array(df_emotions["sad"]).take(selected_indices), np.array(df_emotions["sad"]).take(not_selected_indices))
    ttest_surprise = stats.ttest_ind(np.array(df_emotions["surprise"]).take(selected_indices), np.array(df_emotions["surprise"]).take(not_selected_indices))
    
    ttests_iovc.append(ttest_iovc.pvalue)
    ttests_mem.append(ttest_mem.pvalue)
    ttests_nbr_face.append(ttest_nbr_face.pvalue)
    ttests_max_proba.append(ttest_max_proba.pvalue)
    ttests_happy.append(ttest_happy.pvalue)
    ttests_angry.append(ttest_angry.pvalue)
    ttests_disgust.append(ttest_disgust.pvalue)
    ttests_neutral.append(ttest_neutral.pvalue)
    ttests_fear.append(ttest_fear.pvalue)
    ttests_sad.append(ttest_sad.pvalue)
    ttests_surprise.append(ttest_surprise.pvalue)

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