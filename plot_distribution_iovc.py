import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

JSON_FILE = "./VSUMM-OpenVideo-iovc.json"

with open(JSON_FILE) as f:
    DB = json.load(f)

    for db_key in DB.keys():
        db_data = np.around(np.asarray(DB[db_key]), decimals=3)

        w = Counter(db_data)

        plt.bar(w.keys(), w.values())
        plt.title(db_key)
        plt.show()