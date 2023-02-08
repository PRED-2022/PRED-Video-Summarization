import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

nn_20 = glob("models/sequence_lstm_nn_512_20_epoch=*.pt")
nn_75 = glob("models/sequence_lstm_nn_512_75_epoch=*.pt")
nn_100 = glob("models/sequence_lstm_nn_512_100_epoch=*.pt")

models = [(nn_20, 20), (nn_75, 75), (nn_100, 100)]

values_dict = dict()

for data, nn_size in models:

    values_dict[nn_size] = dict()
    values = []

    for fn in data:
        epoch, train_loss, train_r2, val_r2 = [eleme.split("_")[0] for eleme in fn.split("=")][1:]
        val_r2 = val_r2.split(".pt")[0]
        epoch, train_loss, train_r2, val_r2 = int(epoch), float(train_loss), float(train_r2), float(val_r2)
        values.append((epoch, train_loss, train_r2, val_r2))

    values.sort(key=lambda x: x[0])
    values = np.array(values)
    values_dict[nn_size]["epoch"] = values[:, 0]
    values_dict[nn_size]["train_loss"] = values[:, 1]
    values_dict[nn_size]["train_r2"] = values[:, 2]
    values_dict[nn_size]["val_r2"] = values[:, 3]

# sns.set_theme(style="whitegrid")


data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["train_loss"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
ax = sns.lineplot(data=data, dashes=False, linewidth=1.5)
ax.set(xlabel='Epoch', ylabel='Mean Squared Error sur jeu d\'entraînement')
plt.show()


data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["train_r2"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
ax = sns.lineplot(data=data, dashes=False, linewidth=1.5)
ax.set(xlabel='Epoch', ylabel='R² sur jeu d\'entraînement')
plt.show()

data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["val_r2"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
ax = sns.lineplot(data=data, dashes=False, linewidth=1.5)
ax.set(xlabel='Epoch', ylabel='R² sur jeu de validation')
plt.show()

y_val = []
x_val = []
for _, nn_size in models:
    x_val.append(nn_size)
    y_val.append(np.max(values_dict[nn_size]["val_r2"]))


data = pd.DataFrame({'Score R² sur jeu de validation': y_val, 'Taille de la séquence': x_val})
ax = sns.lineplot(data=data, x='Taille de la séquence', y='Score R² sur jeu de validation', dashes=False, markers=True,  style=["o"] * len(models), linewidth=1.5)
ax.get_legend().remove()
plt.xticks(x_val, x_val)
plt.show()
