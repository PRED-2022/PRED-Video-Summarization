import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), './paperwithcode/ResidualMaskingNetwork'))

from rmn import RMN

import torch
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import glob
import numpy as np
import csv
from tqdm import tqdm
from rich.progress import Progress

VIDEO_FOLDER = './tvsum/video/'

FACE_INTENSITY_FILEPATH = './tv-sum-mem-score.csv'

DISPLAY_VIDEO = True

if __name__ == "__main__":
    m = RMN()
  
    video_names = glob.glob1(VIDEO_FOLDER, '*.mp4')
    
    with open(FACE_INTENSITY_FILEPATH, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['video_name', 'data'])

        with Progress() as progress:
            video_task = progress.add_task("[red]Boucle video...", total=len(video_names))
            for video_name in video_names:
                progress.advance(video_task)

                cap = cv2.VideoCapture(VIDEO_FOLDER + video_name)

                i = 0
                video_analyse_task = progress.add_task("[blue]Analyse video : " + video_name, total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
                    ret, frame = cap.read()
                    i = i + 1
                    progress.advance(video_analyse_task)
                    
                    if not ret:
                        break
                
                    results = m.detect_emotion_for_single_frame(frame)
                    frame = m.draw(frame, results)
                    
                    if DISPLAY_VIDEO:
                        print(results)
                        cv2.imshow('Frame', frame)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break