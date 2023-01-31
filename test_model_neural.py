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

###################################################################################

class MyDataset(torch.utils.data.Dataset):
 
  def __init__(self, data):
    # scaler = StandardScaler()
    # data_gt = data["gt"]
    # data.drop(inplace=True, columns="gt")
    # data = scaler.fit_transform(data)
    self.data = data

  def __len__(self):
    return len(self.data)
   
  def __getitem__(self, idx):
    b = self.data.iloc[idx]
    x = b["gt"]
    b = b.drop("gt")
    b = b.to_numpy()
    return torch.as_tensor(b), torch.as_tensor(x)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(11, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.layers(x)      
        return x

###################################################################################

# Entrainement du réseau

# Dataset
train_df, test_df = train_test_split(big_df, test_size=0.25, random_state=1)
train_df = MyDataset(train_df)
test_df = MyDataset(test_df)
trainloader = torch.utils.data.DataLoader(train_df, batch_size=256, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize the MLP
mlp = MLP()
mlp.to(device)

# Define the loss function and optimizer
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)

# Run the training loop
for epoch in range(0, 5): # 5 epochs at maximum

    # Print epoch
    print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        targets = targets.reshape((targets.shape[0], 1))

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
        if i % 10 == 0:
            print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500))
            current_loss = 0.0

# Process is complete.
print('Training process has finished.')


