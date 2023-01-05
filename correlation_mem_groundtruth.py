import cv2
import torch
from read_tvsum import get_frames_importance
from resmem import ResMem, transformer
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import glob
import numpy as np
import csv
from tqdm import tqdm
from rich.progress import Progress

VIDEO_FOLDER = './tvsum/video/'

MEM_SCORE_FILEPATH = './tv-sum-mem-score.csv'

DISPLAY_VIDEO = False

if __name__ == "__main__":
    pearson_correlations = []
    spearman_correlations = []

    video_names = glob.glob1(VIDEO_FOLDER, '*.mp4')
    
    model = ResMem(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print("Inférence utilisant", device)

    with open(MEM_SCORE_FILEPATH, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        writer.writerow(['video_name', 'pearson', 'spearman', 'memorability_scores'])


        with Progress() as progress:
            video_task = progress.add_task("[red]Boucle video...", total=len(video_names))
            for video_name in video_names:
                progress.advance(video_task)

                cap = cv2.VideoCapture(VIDEO_FOLDER + video_name)

                memorability_scores = []

                worst_image = None
                best_image = None
                
                i = 0
                video_analyse_task = progress.add_task("[blue]Analyse video...", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
                while cap.isOpened():
                    ret, frame = cap.read()
                    i = i + 1
                    if not ret:
                        break

                    progress.advance(video_analyse_task)

                    if i % 1 == 0:
                        # Display the resulting frame
                        image_x = transformer(Image.fromarray(frame))
                        image_x = image_x.to(device)
                        prediction = model(image_x.view(-1, 3, 227, 227))
                        
                        prediction = prediction.detach().cpu().numpy()[0][0]
                        memorability_scores.append(prediction)

                        if worst_image is None or prediction < worst_image[0]:
                            worst_image = (prediction, frame)

                        if best_image is None or prediction > best_image[0]:
                            best_image = (prediction, frame)

                        if DISPLAY_VIDEO:
                            cv2.imshow('Frame', frame)
                            # Press Q on keyboard to  exit
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break


                video_id = video_name.split('.')[0]
                ground_truth = get_frames_importance(video_id)

                pearson_correlation = stats.pearsonr(memorability_scores, ground_truth).statistic
                pearson_correlations.append(pearson_correlation)

                spearman_correlation = stats.spearmanr(memorability_scores, ground_truth).correlation
                spearman_correlations.append(spearman_correlation)

                writer.writerow([video_name, pearson_correlation, spearman_correlation, ",".join(map(str, memorability_scores))])


    print("--------------------------------")
    print("Corrélation de Pearson : ")
    print("- Moyenne : {} ".format(np.average(pearson_correlations)))
    print("- Ecart-type : {} ".format(np.std(pearson_correlations)))
    print("- Min : {} - Max : {}".format(np.min(pearson_correlations), np.max(pearson_correlations)))
    print("--------------------------------")
    print("Corrélation de Spearman : ")
    print("- Moyenne : {} ".format(np.average(spearman_correlations)))
    print("- Ecart-type : {} ".format(np.std(spearman_correlations)))
    print("- Min : {} - Max : {}".format(np.min(spearman_correlations), np.max(spearman_correlations)))
    print("--------------------------------")

    # Affichage du graphique
    plt.plot(spearman_correlations, '-r')
    plt.plot(pearson_correlations, '-b')
    plt.ylabel('Corrélation')
    plt.xlabel('Numéro des vidéos')
    plt.show()


