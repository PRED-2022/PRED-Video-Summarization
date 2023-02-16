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

SCORE_GT = []

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

    global SCORE_GT
    SCORE_GT = np.array(score_gt)

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
        return torch.as_tensor(data.flatten()), torch.as_tensor(gt)

###################################################################################

# Test du réseau
class MLP_LSTM(nn.Module):
    def __init__(self):
        super(MLP_LSTM, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(WINDOW_SIZE * 11, 512),
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.15),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# Dataset
test_df = MyDataset(test_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the MLP
mlp_lstm = torch.load("./models_sans_dropout/sequence_nn_512_75_epoch=20_train_loss=0.240662_train_r2=0.234_val_loss=0.281_val_r2=0.130.pt")

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

plt.plot(video_output, label="output")
plt.plot(video_targets, label="video_targets")

plt.legend()
plt.show()