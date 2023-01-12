import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import glob
import numpy as np
import json
from rich.progress import Progress

VIDEO_FOLDER = './tvsum/video/'

FACE_INTENSITY_FILEPATH = './tv-sum-face-intensity.json'

if __name__ == "__main__":
    with open(FACE_INTENSITY_FILEPATH, 'r', newline='') as json_file:
        video_dict = json.loads(json_file.read())
        
        for video_name in video_dict.keys():
            nbr_face, max_proba = zip(*[ [len(x), (max([0] + [ f["emo_proba"] for f in x ])*5)**4 ] for x in video_dict[video_name] if x is not None ])
            print(max_proba)

            # plt.plot(nbr_face, '-')
            plt.scatter(range(len(nbr_face)), nbr_face,s=max_proba)
            plt.ylabel('Nombre de visage détecté')
            plt.show()