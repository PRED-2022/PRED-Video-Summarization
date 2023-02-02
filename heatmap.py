import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATA_TYPE = "IOVC" # ou "IOVC" 
FILE_NAME = f"TVSum-{DATA_TYPE}-Wasserstein.csv"

a = np.genfromtxt(FILE_NAME, delimiter=";", skip_header=True)
a = a[:, 1:].astype(float)
a = np.nan_to_num(a)
print(a)
first_line = []
with open(FILE_NAME, "r") as csv:
    first_line = csv.readline().split(";")[1:]

fig, ax = plt.subplots(1,1)

ax.imshow(a, cmap='jet', interpolation='nearest')


ax.set_xticks(np.arange(0, len(a)))
ax.set_yticks(np.arange(0, len(a)))

ax.set_xticklabels(first_line, rotation=90, fontsize=6)
ax.set_yticklabels(first_line, fontsize=6)


plt.title(f"TVSum {DATA_TYPE} Wasserstein")
plt.savefig(f"TVSum-{DATA_TYPE}-Wasserstein.png", dpi=1000, pad_inches=0)
plt.show()