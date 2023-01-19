import os
import torch
from torchvision.transforms import ToTensor, Compose
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
VIDEO_FOLDER = './tvsum/video/'
IOVC_FILEPATH = './tv-sum-iovc.json'
IOVC_WEIGHTS = "./IOC_pytorch/weights/model_weights.pth"

if __name__ == "__main__":

    # Set the gpu device if available, cpu if not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    preprocess = Compose([
        # Resize(VIDEO_SIZE),
        ToTensor(),
    ])

    model = IOCNet(37, 50)
    model.load_state_dict(torch.load(IOVC_WEIGHTS))
    model.to(device).eval()
   
    with torch.no_grad():

        video_names = glob1(VIDEO_FOLDER, '*.mp4')

        assert len(video_names) > 0, "No video were found to predict"

        video_data = dict()
        
        with open(IOVC_FILEPATH, 'w', newline='') as output_file:

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
                    
                        frame= cv2.resize(frame, VIDEO_SIZE[::-1])
                        frame = preprocess(Image.fromarray(frame)).to(device).unsqueeze(0)
                        predicted_ioc = model(frame)
                        print(predicted_ioc.cpu().squeeze())
                        predicted_ioc = predicted_ioc.squeeze().detach().cpu().numpy()

                        video_data[video_name].append(predicted_ioc)

            output_file.write(json.dumps(video_data))
            print("Fichier sauvegardé") 