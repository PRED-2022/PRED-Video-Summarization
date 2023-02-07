import json
from glob import glob
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from numpy.lib.stride_tricks import sliding_window_view
from rich.progress import Progress

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
WINDOW_SIZE = 100

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

    df = df.to_numpy()
    df_2 = sliding_window_view(df, WINDOW_SIZE, axis=0) #.reshape((-1, WINDOW_SIZE, 12))
    big_df.append(df_2)

big_df = np.array([item for sublist in big_df for item in sublist])

print("Dataset of shape :", big_df.shape)

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

class MLP_LSTM(nn.Module):
    def __init__(self):
        super(MLP_LSTM, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(WINDOW_SIZE * 11, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

###################################################################################

# Entrainement du réseau


# Dataset
train_df, test_df = train_test_split(big_df, test_size=0.1, random_state=1)
train_df = MyDataset(train_df)
test_df = MyDataset(test_df)
trainloader = torch.utils.data.DataLoader(train_df, batch_size=256, shuffle=True)
testloader = torch.utils.data.DataLoader(test_df, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the MLP
mlp_lstm = MLP_LSTM()
mlp_lstm.to(device)

# Define the loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mlp_lstm.parameters(), lr=1e-5)

# Run the training loop
for epoch in range(0, 250):  # 5 epochs at maximum

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

            # Perform optimization
            optimizer.step()

            s_o = torch.cat((s_o, outputs.squeeze().cpu())).detach()
            s_t = torch.cat((s_t, targets.squeeze().cpu())).detach()

            # Print statistics
            current_loss += loss.item()
            train_loss += loss.item()
            if i % 100 == 0:
                progress.console.print('Loss after batch %5d: %.3f - R²=%.3f' % (i + 1, current_loss / 500, r2_score(s_t, s_o)))
                current_loss = 0.0

            progress.advance(task)

    # Validation loop
    with Progress() as progress:
        task = progress.add_task("Epoch %d - Validation" % (epoch), total=len(testloader))
        with torch.no_grad():

            for y, test_data in enumerate(testloader, 0):
                inputs, targets = test_data
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                targets = targets.reshape((targets.shape[0], 1))
                outputs = mlp_lstm(inputs)
                loss = loss_function(outputs, targets)

                validation_loss += loss.item()

                s_o_v = torch.cat((s_o_v, outputs.squeeze().cpu())).detach()
                s_t_v = torch.cat((s_t_v, targets.squeeze().cpu())).detach()

                progress.advance(task)

    train_r2 = r2_score(s_t, s_o)
    train_loss = train_loss / len(trainloader)
    val_r2 = r2_score(s_t_v, s_o_v)
    val_loss = validation_loss / len(testloader)

    print(f'Epoch {epoch} \t Training Loss={train_loss} - R²={train_r2} \t Validation Loss={val_loss} - R²={val_r2}\n')
    torch.save(mlp_lstm, "sequence_nn_512_%d_epoch=%d_loss=%f_train_r2=%.3f_val_r2=%.3f.pt" % (WINDOW_SIZE, epoch, train_loss, train_r2, val_r2))


# Process is complete.
print('Training process has finished.')
