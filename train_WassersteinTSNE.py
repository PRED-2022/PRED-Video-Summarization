import WassersteinTSNE as WT
import pandas as pd
import json
import numpy as np

with open("./TVsum-iovc.json") as json_file:
    gros_tab = json.load(json_file)

    all_values = []
    index = []
    b_index = []

    for key in gros_tab.keys():
        arr = gros_tab[key]

        index += np.arange(0, len(arr)).tolist()
        b_index += np.repeat(key, len(arr)).tolist()

        all_values += arr

    arrays = [
        np.array(b_index),
        np.array(index)
    ]

    s = pd.DataFrame(np.array(all_values), index=arrays)

    print(s)

    D = WT.WassersteinDistanceMatrix(s)
