"""
Prédire la mémorabilité d'une base de vidéos
et l'écrire dans un fichier CSV
"""
import cv2
import torch
from resmem import ResMem, transformer
from PIL import Image
import glob
import csv
from rich.progress import Progress


# Dossier de la base
VIDEO_FOLDER = './SumMe/videos/'
# Extension des vidéos de la base à inférer
VIDEO_EXTENSION = ".webm"
# Ecriture de la mémorabilité dans le CSV
MEM_SCORE_FILEPATH = './summe-mem-score.csv'

if __name__ == "__main__":
    video_names = glob.glob1(VIDEO_FOLDER, '*' + VIDEO_EXTENSION)
    
    model = ResMem(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Inférence utilisant", device)

    with open(MEM_SCORE_FILEPATH, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['video_name', 'memorability_scores'])


        with Progress() as progress:
            video_task = progress.add_task("[red]Boucle video...", total=len(video_names))
            for video_name in video_names:
                progress.advance(video_task)

                cap = cv2.VideoCapture(VIDEO_FOLDER + video_name)

                memorability_scores = []
                
                video_analyse_task = progress.add_task("[blue]Analyse video : " + video_name, total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    progress.advance(video_analyse_task)

                    image_x = transformer(Image.fromarray(frame))
                    image_x = image_x.to(device)
                    prediction = model(image_x.view(-1, 3, 227, 227))
                    
                    prediction = prediction.detach().cpu().numpy()[0][0]
                    memorability_scores.append(prediction)

                writer.writerow([video_name, ",".join(map(str, memorability_scores))])


