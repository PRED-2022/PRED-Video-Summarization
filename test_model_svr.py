import json
from glob import glob
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle



# Ground Truth
groundtruth_videos = pd.read_csv('./TVSum-groundtruth.csv', sep=';', header=0).set_index('id')

# IOVC
iovc_videos = pd.read_json("./TVSum-iovc.json", lines=True)

# Emotions
emotion_videos = pd.read_json("./PROCESSED-TVsum-face-intensity.json", lines=True)

# Memorability
memorability_videos = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0).set_index("video_name")

big_df = None

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

    if big_df is None:
        big_df = df
    else:
        big_df = pd.concat([big_df, df], axis= 0)

print(big_df)


# Entrainement du réseau
train_df, test_df = train_test_split(big_df, test_size=0.25, random_state=1)

scaler = StandardScaler()

train_df_gt = train_df["gt"]
train_df.drop(inplace=True, columns="gt")
train_df = scaler.fit_transform(train_df)

test_df_gt = test_df["gt"]
test_df.drop(inplace=True, columns="gt")
test_df = scaler.fit_transform(test_df)

regr_model = svm.SVR()
regr_model.fit(train_df, train_df_gt)

print("Accuracy:", regr_model.score(test_df, test_df_gt))

pickle.dump(regr_model, open('model_regression_SVR_input_normalized.sav', 'wb'))