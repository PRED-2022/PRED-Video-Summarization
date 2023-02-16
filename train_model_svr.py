"""
Permet d'entrainer un SVR sur un vecteur de caractéristique d'une image seule
"""

import json
from glob import glob
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

TRAINING_MODEL = False
TEST_VIDEOS = ["WxtbjNsCQ8A.mp4", "XzYM3PfTM4w.mp4", "kLxoNp-UchI.mp4", "eQu1rNs0an0.mp4", "VuWGsYPqAX8.mp4"]

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
    
    # Création du jeux de test
    if key in TEST_VIDEOS:
        test_df = pd.concat([test_df, df], axis= 0)

        # On récupère les données de la vidéo pour inférer les frames
        if key == TEST_VIDEOS[0]:
            vid_df = df

    else:
        # Création du jeux train/validation
        big_df = pd.concat([big_df, df], axis= 0)



# Split du jeu d'entrainement / validation
train_df, valid_df = train_test_split(big_df, test_size=0.25, random_state=1)


# Normalisation des données
scaler = StandardScaler()

train_df_gt = train_df["gt"]
train_df.drop(inplace=True, columns="gt")
train_df = scaler.fit_transform(train_df)

valid_df_gt = valid_df["gt"]
valid_df.drop(inplace=True, columns="gt")
valid_df = scaler.transform(valid_df)

test_df_gt = test_df["gt"]
test_df.drop(inplace=True, columns="gt")
test_df = scaler.transform(test_df)

vid_df_gt = vid_df["gt"]
vid_df.drop(inplace=True, columns="gt")
vid_df = scaler.transform(vid_df)


# Entrainement du réseau
regr_model = svm.SVR()

print("Using the model")

if TRAINING_MODEL:
    regr_model.fit(train_df, train_df_gt)
    pickle.dump(regr_model, open('model_regression_SVR_input_normalized.sav', 'wb'))
else:
    regr_model = pickle.load(open('model_regression_SVR_input_normalized.sav', 'rb'))
    print("Model loaded")

print("Accuracy R² sur jeux de validation :", regr_model.score(valid_df, valid_df_gt))
print("Accuracy R² sur jeux de test :", regr_model.score(test_df, test_df_gt))
print("Accuracy R² sur la video à inférer :", regr_model.score(vid_df, vid_df_gt))

print(len(vid_df))

vals_predicted = regr_model.predict(vid_df)
print(vals_predicted)
np.save("result_WxtbjNsCQ8A", vals_predicted)
print(np.max(vals_predicted), np.min(vals_predicted), np.mean(vals_predicted))

import matplotlib.pyplot as plt
plt.plot(np.arange(0, len(vals_predicted)), vals_predicted)
plt.show()
