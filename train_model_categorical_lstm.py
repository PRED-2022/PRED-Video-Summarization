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
from sklearn.metrics import r2_score, accuracy_score
from numpy.lib.stride_tricks import sliding_window_view
from rich.progress import Progress
from math import ceil, floor

VALIDATION_VIDEO = ["i3wAGJaaktw.mp4", "98MoyGZKHXc.mp4", "byxOvuiIJV0.mp4", "eQu1rNs0an0.mp4", "Yi4Ij2NM7U4.mp4", "sTEELN-vY30.mp4", "_xMr-HKMfVA.mp4", "WxtbjNsCQ8A.mp4", "gzDbaEs1Rlg.mp4", "WG0MBPpPC6I.mp4"]

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
WINDOW_SIZE = 50

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

    nb_selected_frames = ceil(0.10 * nbr)
    threshold = score_gt.sort_values(ascending=False).iloc[nb_selected_frames]
    selected_frames = score_gt[score_gt >= threshold]
    selected_indices = list(selected_frames.index.values)
    df["gt"] = np.where(score_gt >= threshold, 1, 0)

    df = df.to_numpy()
    df_2 = sliding_window_view(df, WINDOW_SIZE, axis=0)
    return df_2


# Donnée d'entrainement
train_df = []
for key in iovc_videos.keys():
    if key not in VALIDATION_VIDEO:
        train_df.append(readVideoData(key))
train_df = np.array([item for sublist in train_df for item in sublist])

# Donnée de validation
val_df = []
for key in VALIDATION_VIDEO:
    val_df.append(readVideoData(key))
val_df = np.array([item for sublist in val_df for item in sublist])


print((train_df[train_df[:, -1] == 1]).shape)
exit()

print("Dataset training of shape :", train_df.shape)
print("Dataset validation of shape :", val_df.shape)

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

class MLP_LSTM(nn.Module):
    def __init__(self):
        super(MLP_LSTM, self).__init__()

        self.lstm = torch.nn.LSTM(11, 512, batch_first=True)

        self.layers = torch.nn.Sequential(
            torch.nn.ReLU(),
            # torch.nn.Dropout(0.15),
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

###################################################################################

# Entrainement du réseau


# Dataset
train_df = MyDataset(train_df)
val_df = MyDataset(val_df)
trainloader = torch.utils.data.DataLoader(train_df, batch_size=256, shuffle=True)
valloader = torch.utils.data.DataLoader(val_df, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the MLP
mlp_lstm = MLP_LSTM()
mlp_lstm.to(device)

# Define the loss function and optimizer
loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(mlp_lstm.parameters(), lr=1e-5)

# Run the training loop
for epoch in range(0, 50):

    # Set current loss value
    current_loss = 0.0
    train_loss = 0.0
    validation_loss = 0.0

    s_o = torch.tensor([])
    s_t = torch.tensor([])

    s_o_v = torch.tensor([])
    s_t_v = torch.tensor([])

    # Iterate over the DataLoader for training data
    with Progress() as progress:
        task = progress.add_task("Epoch %d - Training" % (epoch), total=len(trainloader))
        for i, data in enumerate(trainloader, 0):

            # Get and prepare inputs
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.float().to(device)
            targets = targets.reshape((targets.shape[0], 1))

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp_lstm(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Perform backward pass
            loss.backward()
            train_loss += loss.item()

            # Perform optimization
            optimizer.step()

            s_o = torch.cat((s_o, torch.sigmoid(outputs).squeeze().cpu())).detach()
            s_t = torch.cat((s_t, targets.squeeze().cpu())).detach()

            # Print statistics
            if False:
                current_loss += loss.item()
                if i % 100 == 0:
                    progress.console.print('Loss after batch %5d: %.3f - R²=%.3f' % (i + 1, current_loss / 500, r2_score(s_t, s_o)))
                    current_loss = 0.0

            # Move progress of one batch
            progress.advance(task)

    # Validation loop
    with Progress() as progress:
        task = progress.add_task("Epoch %d - Validation" % (epoch), total=len(valloader))
        with torch.no_grad():

            for y, val_data in enumerate(valloader, 0):
                inputs, targets = val_data
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                targets = targets.reshape((targets.shape[0], 1))
                outputs = mlp_lstm(inputs)
                loss = loss_function(outputs, targets)

                validation_loss += loss.item()

                s_o_v = torch.cat((s_o_v, torch.sigmoid(outputs).squeeze().cpu())).detach()
                s_t_v = torch.cat((s_t_v, targets.squeeze().cpu())).detach()

                progress.advance(task)


    train_acc = accuracy_score(s_t >= 1, s_o >= 0.9)
    train_loss = train_loss / len(trainloader)
    val_acc = accuracy_score(s_t_v >= 1, s_o_v >= 0.9)
    val_loss = validation_loss / len(valloader)

    print(f'Epoch {epoch} - Training Loss={train_loss} - Acc={train_acc} \t Validation Loss={val_loss} - Acc={val_acc}\n')
    torch.save(mlp_lstm, "CLASSIFICATION_sequence_lstm_nn_512_%d_epoch=%d_train_loss=%f_train_acc=%.3f_val_loss=%.3f_val_acc=%.3f.pt" % (WINDOW_SIZE, epoch, train_loss, train_acc, val_loss, val_acc))


# Process is complete.
print('Training process has finished.')
