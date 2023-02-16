"""
Permet de tester un modèle LSTM en inférence sur une vidéo
"""

import json
from glob import glob
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from numpy.lib.stride_tricks import sliding_window_view
from rich.progress import Progress
import matplotlib.pyplot as plt

TEST_VIDEO = "3eYKfiOEJNs.mp4"

# Ground Truth
groundtruth_videos = pd.read_csv('./TVSum-groundtruth.csv', sep=';', header=0).set_index('id')

# IOVC
iovc_videos = pd.read_json("./TVSum-iovc.json", lines=True)

# Emotions
emotion_videos = pd.read_json("./PROCESSED-TVsum-face-intensity.json", lines=True)

# Memorability
memorability_videos = pd.read_csv('./TVSum-memorability.csv', sep=';', header=0).set_index("video_name")

big_df = []

NBR_FEATURES = 11
WINDOW_SIZE = 75

VIDEO_DT = []

# Chargement des données des videos
def readVideoData(key):
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

    global VIDEO_DT
    VIDEO_DT = df

    df = df.to_numpy()
    df_2 = sliding_window_view(df, WINDOW_SIZE, axis=0)
    return df_2


# Donnée de test
test_df = readVideoData(TEST_VIDEO)

###################################################################################

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        b = self.data[idx].T
        gt = b[:, -1][-1]
        data = b[:, :-1]
        return torch.as_tensor(data), torch.as_tensor(gt)

###################################################################################

# Test du réseau

class MLP_LSTM(nn.Module):
    def __init__(self):
        super(MLP_LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(11, 512, batch_first=True)

        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        x = self.layers(output[:, -1, :])
        return x

# Dataset
test_df = MyDataset(test_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the MLP
mlp_lstm = torch.load("./models_sans_dropout/sequence_lstm_nn_512_75_epoch=21_train_loss=0.263709_train_r2=0.160_val_loss=0.271_val_r2=0.161.pt")

video_targets = []
video_output = []

for data in test_df:

    # Get and prepare inputs
    inputs, targets = data
    inputs, targets = inputs.float().unsqueeze(0).to(device), targets.float().unsqueeze(0).to(device)
    targets = targets.reshape((targets.shape[0], 1))

    # Perform forward pass
    outputs = mlp_lstm(inputs)

    video_output.append(outputs.squeeze().cpu().item())
    video_targets.append(targets.squeeze().cpu().item())

video_output = np.array(video_output)
video_targets = np.array(video_targets)




# Affiche un graphique des résultats obtenus / résultats voulus
plt.plot(video_output, label="output")
plt.plot(video_targets, label="video_targets")

plt.legend()
plt.show()




# Affiche un graphique de toutes les caractéristiques + prédictions du modèle
output_data = np.zeros((len(VIDEO_DT[VIDEO_DT.columns[0]])))
index = len(output_data) - len(video_output)
output_data[index:] = video_output

fig, axs = plt.subplots(13)
for i, keys in enumerate(VIDEO_DT.columns, 0):
    color = ""
    if i <= 8:
        color = "tab:orange"
    elif i <= 9:
        color = "tab:red"
    elif i <= 10:
        color = "tab:green"
    else: 
        color = ""

    axs[i].set_title(keys)
    axs[i].plot(VIDEO_DT[keys], color)

axs[12].set_title("prediction")
axs[12].plot(output_data)
plt.show()