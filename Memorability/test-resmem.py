"""
Test de l'inférence de la mémorabilité avec ResMem
sur une vidéo
"""
import cv2
import torch
from resmem import ResMem, transformer
from PIL import Image
import matplotlib.pyplot as plt
from rich.progress import Progress

VIDEO_FOLDER = "./video/"
VIDEO_NAME = "ferrari-f12.mp4"

DISPLAY_VIDEO = False

if __name__ == "__main__":

    cap = cv2.VideoCapture(VIDEO_FOLDER + VIDEO_NAME)
   
    model = ResMem(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    list_item = []

    worst_image = None
    best_image = None

    i = 0
    with Progress() as progress:
        task = progress.add_task("[blue]Analysing...", total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened():
            ret, frame = cap.read()
            i = i + 1

            progress.advance(task)

            if not ret:
                break

            if i % 25 == 0:

                # Display the resulting frame
                image_x = transformer(Image.fromarray(frame))
                image_x = image_x.to(device)
                prediction = model(image_x.view(-1, 3, 227, 227))
                
                prediction = prediction.detach().cpu().numpy()[0][0]
                list_item.append(prediction)

                if worst_image is None or prediction < worst_image[0]:
                    worst_image = (prediction, frame)

                if best_image is None or prediction > best_image[0]:
                    best_image = (prediction, frame)

                if DISPLAY_VIDEO:
                    cv2.imshow('Frame', frame)
                    # Press Q on keyboard to  exit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break


    # Affichage des pires et des meilleures image
    cv2.imshow('Worst Memorability : ' + str(worst_image[0]), worst_image[1])
    cv2.imshow('Best Memorability : ' + str(best_image[0]), best_image[1])

    # Affichage du graphique
    plt.plot(list_item, '-r')
    plt.ylabel('Memorability of frames')
    plt.xlabel(VIDEO_NAME)
    plt.ylim([0.0, 1.0])
    plt.xlim([0, len(list_item)-1])
    plt.show()
