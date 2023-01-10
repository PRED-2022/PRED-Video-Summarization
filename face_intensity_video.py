import cv2
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), './paperwithcode/ResidualMaskingNetwork'))

from rmn import RMN

import torch
from PIL import Image
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
from rich.progress import Progress

VIDEO_FOLDER = './tvsum/video/'

FACE_INTENSITY_FILEPATH = './tv-sum-face-intensity.csv'

DISPLAY_VIDEO = False

if __name__ == "__main__":
    m = RMN()
  
    video_names = glob.glob1(VIDEO_FOLDER, '*.mp4')

    video_data = dict() 
    
    with open(FACE_INTENSITY_FILEPATH, 'w', newline='') as output_file:

        with Progress() as progress:
            video_task = progress.add_task("[red]Boucle video...", total=len(video_names))
            for video_name in video_names:
                progress.advance(video_task)

                cap = cv2.VideoCapture(VIDEO_FOLDER + video_name)

                video_data[video_name] = list()

                i = 0
                video_analyse_task = progress.add_task("[blue]Analyse video : " + video_name, total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
                    ret, frame = cap.read()
                    i = i + 1
                    progress.advance(video_analyse_task)
                    
                    if not ret:
                        video_data[video_name].append(None)
                        break
                
                    results = m.detect_emotion_for_single_frame(frame)

                    video_data[video_name].append(results)

                    frame = m.draw(frame, results)
                    
                    if DISPLAY_VIDEO:
                        cv2.imshow('Frame', frame)
                        # Press Q on keyboard to  exit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            
        output_file.write(json.dumps(video_data))
        print("Fichier sauvegardé") 