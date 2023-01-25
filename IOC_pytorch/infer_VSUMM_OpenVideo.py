import os
import torch
from torchvision.transforms import ToTensor, Compose, Normalize
from datasets_ioc import ResizeAndPad
from model_IOC import IOCNet
from PIL import Image

import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from glob import glob1
import numpy as np
import json
from rich.progress import Progress

VIDEO_SIZE = (300, 400)
VIDEO_FOLDER = 'M:/VSUMM-OpenVideo/'
IOVC_FILEPATH = './VSUMM-OpenVideo-iovc.json'
IOVC_WEIGHTS = "./IOC_pytorch/weights/model_weights.pth"

LENGTH_BATCH_OF_IMAGES = 150

if __name__ == "__main__":

    # Set the gpu device if available, cpu if not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    preprocess = Compose([
        ToTensor(),
    ])

    model = IOCNet(37, 50)
    model.load_state_dict(torch.load(IOVC_WEIGHTS))
    model.to(device).eval()
   
    with torch.no_grad():

        video_names = glob1(VIDEO_FOLDER, '*.mpg')

        assert len(video_names) > 0, "No video were found to predict"

        video_data = dict()

        with Progress() as progress:
            video_task = progress.add_task("[red]Boucle video...", total=len(video_names))
            for video_name in video_names:
                progress.advance(video_task)

                cap = cv2.VideoCapture(VIDEO_FOLDER + video_name)

                video_data[video_name] = list()

                i = 0
                past_frames = []
                video_analyse_task = progress.add_task("[blue]Analyse video : " + video_name, total=cap.get(cv2.CAP_PROP_FRAME_COUNT) // LENGTH_BATCH_OF_IMAGES + 1)
               
                video_is_over = False
                while cap.isOpened() and video_is_over is False:
                    ret, frame = cap.read()
                    i = i + 1
                    
                    if not ret:
                        video_is_over = True
                    
                    else:
                        frame = cv2.resize(frame, VIDEO_SIZE[::-1])
                        frame = Image.fromarray(frame)
                        frame = preprocess(frame)
                        frame = frame.to(device)

                        mean, std = frame.mean([1,2]), frame.std([1,2])

                        if 0 in std:
                            std[0] = std[1] = std[2] = 1

                        frame = Normalize(mean, std)(frame).unsqueeze(0)

                        past_frames.append(frame)

                    if len(past_frames) > 0 and (len(past_frames) % LENGTH_BATCH_OF_IMAGES == 0 or video_is_over is True):
                        batch = torch.cat(past_frames, 0)
                        past_frames = []

                        predicted_ioc = model(batch)
                        predicted_ioc = predicted_ioc.squeeze().cpu().tolist()
                        video_data[video_name] += predicted_ioc

                        progress.advance(video_analyse_task)

                with open(IOVC_FILEPATH, 'w', newline='') as output_file:
                    json_obj = json.dumps(video_data)
                    output_file.write(json_obj)
                    print("Fichier sauvegardé") 