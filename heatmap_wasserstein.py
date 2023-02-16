"""
Permet de générer le graphiques à partir des fichiers de matrice de wasserstein en csv
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATA_TYPE = "IOVC" # ou "IOVC" 
FILE_NAME = f"TVSum-{DATA_TYPE}-Wasserstein.csv"

# Chargement du csv
a = np.genfromtxt(FILE_NAME, delimiter=";", skip_header=True)
a = a[:, 1:].astype(float)
a = np.nan_to_num(a)

# Lecture du nom des vidéos
first_line = []
with open(FILE_NAME, "r") as csv:
    first_line = csv.readline().split(";")[1:]

# Génération du graphique
fig, ax = plt.subplots(1,1)

ax.imshow(a, cmap='jet', interpolation='nearest')

ax.set_xticks(np.arange(0, len(a)))
ax.set_yticks(np.arange(0, len(a)))

ax.set_xticklabels(first_line, rotation=90, fontsize=4)
ax.set_yticklabels(first_line, fontsize=4)

# Enregistrement / affichage du graphique 
plt.title(f"TVSum {DATA_TYPE} Wasserstein")
plt.savefig(f"TVSum-{DATA_TYPE}-Wasserstein.png", dpi=1000, pad_inches=0)
plt.show()