import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
import glob
import numpy as np
import json
from rich.progress import Progress

VIDEO_FOLDER = './tvsum/video/'

FACE_INTENSITY_FILEPATH = './tv-sum-face-intensity.json'

CMAP = cm.get_cmap('Spectral')

if __name__ == "__main__":
    with open(FACE_INTENSITY_FILEPATH, 'r', newline='') as json_file:
        video_dict = json.loads(json_file.read())
        
        for video_name in video_dict.keys():
            nbr_face, size, max_proba = zip(*[ [len(x), (max([0] + [ f["emo_proba"] for f in x ])*5)**4, max([0] + [ f["emo_proba"] for f in x ]) ] for x in video_dict[video_name] if x is not None ])
            print(max_proba)

            # plt.plot(nbr_face, '-')
            plt.scatter(range(len(nbr_face)), nbr_face, c=max_proba) #  s=size,
            plt.ylabel('Nombre de visage détecté')
            plt.colorbar()
            plt.show()