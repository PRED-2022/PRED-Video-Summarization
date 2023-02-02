import numpy
import matplotlib.pyplot as plt

DATA_TYPE = "Memorability" # ou "IOVC" 

a = numpy.genfromtxt('TVSum-{DATA_TYPE}-Wasserstein.csv', delimiter=";", skip_header=True)
a = a[:, 1:].astype(float)
a = numpy.nan_to_num(a)
print(a)

plt.imshow(a, cmap='jet', interpolation='nearest')
plt.title("TVSum {DATA_TYPE} Wasserstein")
plt.savefig("TVSum-{DATA_TYPE}-Wasserstein.png", dpi=500, pad_inches=0)
plt.show()