import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

MODEL_NAME = "models_sans_dropout/sequence_nn_512_%d_epoch=*.pt"
MODEL_NAME = "models_sans_dropout/sequence_nn_512_%d_epoch=*.pt"
MODEL_NAME = "sequence_lstm_nn_1024_%d_epoch=*.pt"

nn = glob(MODEL_NAME)

models_size = [75, 100]
models = [(glob(MODEL_NAME % size), size) for size in models_size]

values_dict = dict()

for data, nn_size in models:

    values_dict[nn_size] = dict()
    values = []

    for fn in data:
        epoch, train_loss, train_r2, val_loss, val_r2 = [eleme.split("_")[0] for eleme in fn.split("=")][1:]
        val_r2 = val_r2.split(".pt")[0]
        epoch, train_loss, train_r2, val_loss, val_r2 = int(epoch), float(train_loss), float(train_r2), float(val_loss), float(val_r2)
        values.append((epoch, train_loss, train_r2, val_loss, val_r2))

    values.sort(key=lambda x: x[0])
    values = np.array(values)
    values_dict[nn_size]["epoch"] = values[:, 0]
    values_dict[nn_size]["train_loss"] = values[:, 1]
    values_dict[nn_size]["train_r2"] = values[:, 2]
    values_dict[nn_size]["val_loss"] = values[:, 3]
    values_dict[nn_size]["val_r2"] = values[:, 4]

    print(nn_size, "max r2 training", np.max(values_dict[nn_size]["train_r2"]))
    print(nn_size, "max r2 validation", np.max(values_dict[nn_size]["val_r2"]))

fig, axs = plt.subplots(1, 2)

data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["train_loss"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
sns.lineplot(ax=axs[0], data=data, dashes=False, linewidth=1.5)
axs[0].set(xlabel='Epoch', ylabel='Mean Squared Error sur jeu d\'entraînement')

data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["train_r2"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
sns.lineplot(ax=axs[1], data=data, dashes=False, linewidth=1.5)
axs[1].set(xlabel='Epoch', ylabel='R² sur jeu d\'entraînement')


plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2)

data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["val_loss"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
sns.lineplot(ax=axs[0], data=data, dashes=False, linewidth=1.5)
axs[0].set(xlabel='Epoch', ylabel='Mean Squared Error sur jeu de validation')


data = []
for _, nn_size in models:
    data.append(values_dict[nn_size]["val_r2"])
data = pd.DataFrame(np.array(data).T, columns=list(map(lambda x: "Séquence : " + str(x[1]), models)))
sns.lineplot(ax=axs[1], data=data, dashes=False, linewidth=1.5)
axs[1].set(xlabel='Epoch', ylabel='R² sur jeu de validation')

plt.tight_layout()
plt.show()
# fig.suptitle(MODEL_NAME, fontsize=16)








exit()

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