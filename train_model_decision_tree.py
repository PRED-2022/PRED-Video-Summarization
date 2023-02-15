from math import ceil, floor
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from random import shuffle

TEST_VIDEO = ["i3wAGJaaktw.mp4", "98MoyGZKHXc.mp4", "byxOvuiIJV0.mp4", "eQu1rNs0an0.mp4", "Yi4Ij2NM7U4.mp4", "sTEELN-vY30.mp4", "_xMr-HKMfVA.mp4", "WxtbjNsCQ8A.mp4", "gzDbaEs1Rlg.mp4", "WG0MBPpPC6I.mp4"]

# Ground Truth
groundtruth_videos = pd.read_csv('./TVSum-groundtruth.csv', sep=';', header=0).set_index('id')

# IOVC
iovc_videos = pd.read_json("./TVSum-iovc.json", lines=True)

# Emotions
emotion_videos = pd.read_json("./PROCESSED-TVsum-face-intensity.json", lines=True)

# Memorability
memorability_videos = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0).set_index("video_name")

# Dataframe 
train_df = pd.DataFrame()
test_df = pd.DataFrame()

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
    if key in TEST_VIDEO:
        test_df = pd.concat([test_df, df], axis=0)
    else:
        train_df = pd.concat([train_df, df], axis=0)


train_df.set_index(pd.RangeIndex(len(train_df)), inplace=True)
test_df.set_index(pd.RangeIndex(len(test_df)), inplace=True)

"""

    Data imbalance :
    les classes frame selectionnée et frame non sélectionnée sont déséquilibrées.
    Il y a environ 10-15% de sélectionnées parmi toutes les frames

    On garde autant de frames non selectionnes
"""

nb_selected_frames = len(train_df[train_df["gt"] == 1])
nb_not_selected_frames = len(train_df[train_df["gt"] == 0])
NB_TRAIN_SPLITS = floor(nb_not_selected_frames/nb_selected_frames)
selected_indices = list(train_df[train_df["gt"] == 1].index.values)
print("Nb de splits de train :", NB_TRAIN_SPLITS)
not_selected_indices = list(train_df[train_df["gt"] == 0].index.values)
shuffle(not_selected_indices)

train_gt = train_df["gt"]
train_df.drop(inplace=True, columns="gt")
feature_names = train_df.columns

nb_selected_frames_test = len(test_df[test_df["gt"] == 1])
nb_not_selected_frames_test = len(test_df[test_df["gt"] == 0])
NB_TEST_SPLITS = floor(nb_not_selected_frames_test/nb_selected_frames_test)
selected_indices_test = list(test_df[test_df["gt"] == 1].index.values)
print("Nb de splits de test :", NB_TEST_SPLITS)
not_selected_indices_test = list(test_df[test_df["gt"] == 0].index.values)
shuffle(not_selected_indices_test)

test_gt = test_df["gt"]
test_df.drop(inplace=True, columns="gt")

auc = []
fp_tp_rates = []
confusion_matrices = []
for test_i in range(NB_TEST_SPLITS):
    # Récupération des éléments du groupe actuel d'entrainement
    first_indice = test_i*nb_selected_frames_test
    last_indice = first_indice + nb_selected_frames_test

    # Jeu d'entrainement
    train_indices = not_selected_indices[:nb_selected_frames] + selected_indices
    X_train, y_train = train_df.loc[train_indices], train_gt.loc[train_indices]

    test_indices = not_selected_indices_test[first_indice:last_indice] + selected_indices_test
    X_test = test_df.loc[test_indices]
    y_test = test_gt.loc[test_indices]

    # Entrainement de l'arbre
    binary_classif_model = DecisionTreeClassifier(max_depth=10, random_state=0)
    binary_classif_model.fit(X_train, y_train)
    # Prédiction sur les donnees de test
    y_test_pred = binary_classif_model.predict(X_test)

    # Calcul de la matrice de confusion
    confusion_matrices.append(metrics.confusion_matrix(y_test, y_test_pred))

    # Prédiction des scores plutôt que binaire
    y_test_pred_score = binary_classif_model.predict_proba(X_test)[:, 1]

    # Courbe ROC
    fp_rate, tp_rate, thresholds = metrics.roc_curve(y_test, y_test_pred_score)
    fp_tp_rates.append((fp_rate, tp_rate))

    # AUC
    auc.append(metrics.auc(fp_rate, tp_rate))

# Sauvegarde des matrices de confusion
fig, ax = plt.subplots(1, len(confusion_matrices), figsize=(15, 5), sharey='row')

for i, conf_mat in enumerate(confusion_matrices):
    disp = metrics.ConfusionMatrixDisplay(conf_mat)
    disp.plot(ax=ax[i], xticks_rotation=45)
    disp.ax_.set_title("Split " + str(i+1))
    disp.im_.colorbar.remove()
    if i!=0:
        disp.ax_.set_ylabel('')

plt.subplots_adjust(wspace=0.40, hspace=0.1)
fig.colorbar(disp.im_, ax=ax)
fig.savefig("confusion_matrices.png")
plt.close('all')


plt.figure(figsize=(10, 10))
# Affichage Courbe ROC
for fp_rate, tp_rate in fp_tp_rates:
    plt.plot(fp_rate, tp_rate, linestyle='--', marker='o') 

plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel("Taux de vrai positif : v(s)")
plt.xlabel("Taux de faux positif : 1-w(s)")
plt.show()

# Calcul AUC
auc = np.array(auc)
print("Valeur AUC de chaque modèle :", auc)
print("Précision - moyennes des AUC :", np.mean(auc))
print("Robustesse - écart-type des AUC :", np.std(auc))

