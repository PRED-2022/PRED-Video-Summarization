from math import ceil
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Ground Truth
groundtruth_videos = pd.read_csv('./TVSum-groundtruth.csv', sep=';', header=0).set_index('id')

# IOVC
iovc_videos = pd.read_json("./TVSum-iovc.json", lines=True)

# Emotions
emotion_videos = pd.read_json("./PROCESSED-TVsum-face-intensity.json", lines=True)

# Memorability
memorability_videos = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0).set_index("video_name")

# Dataframe 
big_df = pd.DataFrame()
test_df = pd.DataFrame()
vid_df = pd.DataFrame()

# Création du big tableau
for key in iovc_videos.keys():
   
    score_gt = np.array(groundtruth_videos.loc[key.replace(".mp4", ""), "importance"].split(",")).astype(float)
    score_iovc = np.array(iovc_videos[key].iloc[0], dtype=float)
    score_mem = np.array(memorability_videos.loc[key, "memorability_scores"].split(",")).astype(float)

    df_emotions = pd.DataFrame(list(filter(lambda x: x is not None, emotion_videos[key].iloc[0])))

    nbr = len(score_gt)

    score_gt = pd.Series(score_gt)
    score_iovc = pd.Series(score_iovc[:nbr])
    score_mem = pd.Series(score_mem[:nbr])
    df_emotions = df_emotions[:nbr]

    df = df_emotions

    df["iovc"] = score_iovc
    df["memorability"] = score_mem
    df["gt"] = score_gt

    """
        Sélection des frames retenues pour le résumé vidéo
        selon la vérité-terrain
    """
    # on garde au moins 10 % des meilleures frames en terme d'importance
    nb_selected_frames = ceil(0.10*nbr)
    threshold = score_gt.sort_values(ascending=False).iloc[nb_selected_frames]
    selected_frames = score_gt[score_gt >= threshold]
    selected_indices = list(selected_frames.index.values)
    df["gt"] = np.where(score_gt >= threshold, 1, 0)
    big_df = pd.concat([big_df, df], axis= 0)


big_gt = big_df["gt"]
big_df.drop(inplace=True, columns="gt")
big_df = big_df.to_numpy()

# Entrainement du réseau
binary_classif_model = DecisionTreeClassifier(max_depth=10)

auc = []
kf = KFold(n_splits=6, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(big_df)):
    # Récupération des éléments du groupe actuel
    X_train, X_test = big_df[train_index], big_df[test_index]

    y_train, y_test = big_gt.iloc[train_index], big_gt.iloc[test_index]

    # Entrainement de l'arbre
    binary_classif_model = DecisionTreeClassifier(max_depth=10, random_state=0)
    binary_classif_model.fit(X_train, y_train)

    # Prédiction
    y_test_pred = binary_classif_model.predict_proba(X_test)[:, 1]

    # Courbe ROC
    fp_rate, tp_rate, thresholds = metrics.roc_curve(y_test, y_test_pred)
    plt.plot(fp_rate, tp_rate, linestyle='--', marker='o') 

    # AUC
    auc.append(metrics.auc(fp_rate, tp_rate))


plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel("Taux de vrai positif : v(s)")
plt.xlabel("Taux de faux positif : 1-w(s)")
plt.show()

# Calcul AUC
auc = np.array(auc)
print("Valeur AUC de chaque modèle :", auc)
print("Précision - moyennes des AUC :", np.mean(auc))
print("Robustesse - écart-type des AUC :", np.std(auc))