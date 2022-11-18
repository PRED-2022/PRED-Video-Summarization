import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from timm.models import create_model
import resmem
import numpy

class TestModel:
    def __init__(self, model=None, transformer=None, frame_size=None, name=""):
        self.model = model
        self.transformer = transformer
        self.frame_size = frame_size
        self.name = name

VIDEO_FOLDER = "./"
VIDEO_NAME = "me-at-the-zoo.webm"

DISPLAY_VIDEO = False


if __name__ == "__main__":

    """
    ResMem from
    @inproceedings{ResMem2021,
        title = {Embracing New Techniques in Deep Learning for Predicting Image Memorability},
        author = {Needell, Coen D. and Bainbridge, Wilma A.},
        booktitle = {Proceedings of the Vision Sciences Society, 2021},
        year = {2021},
        publisher = {ARVO Journals},
        url = {https://www.coeneedell.com/publication/vss2021/vss2021.pdf}
    }
    """
    resmem_model = TestModel(model=resmem.ResMem(pretrained=True), transformer=resmem.transformer, frame_size=227, name="ResMem")

    humanmem_transformer = transforms.Compose([
        transforms.RandomResizedCrop(224, (1, 1)), # this is equivalent to resize
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    ])


    """
    HumanMem from 
    @inproceedings{han2022machinemem,
        title={What Images are More Memorable to Machines?},
        author={Junlin Han and Huangying Zhan and Jie Hong and Pengfei Fang and Hongdong Li and Lars Petersson and Ian Reid},
        booktitle={arXiv preprint arXiv:2211.07625},
        year={2022}
    }

    From https://github.com/JunlinHan/MachineMem
    Pre-trained model : https://drive.google.com/drive/folders/1tO4ruBAToGSLZZ8VzAJ6v6O_99p6AJKI
    """
    humanmem_model = TestModel(model=create_model("resnet50", drop_rate=0.5, num_classes=1,), transformer=humanmem_transformer, frame_size=224, name="HumanMem")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load("./model/humanmem_predictor.tar", map_location=device)
    humanmem_model.model.load_state_dict(checkpoint['state_dict'])

    models = [resmem_model, humanmem_model]

    barplots = []
    #worst_images = []
    #best_images = []

    for m in models:
        print(device)
        m.model.to(device)
        m.model.eval()
        list_item = []

        worst_image = None
        best_image = None

        cap = cv2.VideoCapture(VIDEO_FOLDER + VIDEO_NAME)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            i = i + 1

            if not ret:
                break

            if i % 1 == 0:
                
                # Display the resulting frame
                image_x = m.transformer(Image.fromarray(frame))
                image_x = image_x.to(device)
                prediction = m.model(image_x.view(-1, 3, m.frame_size, m.frame_size))
                
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

        barplots.append(list_item)

        #worst_images.append(worst_image)
        #best_images.append(best_image)


        # Affichage des pires et des meilleures image
        #cv2.imshow('Worst Memorability : ' + str(worst_image[0]), worst_image[1])
        #cv2.imshow('Best Memorability : ' + str(best_image[0]), best_image[1])

    #cv2.waitKey(10000)
    # Affichage du graphique
    for m in range(len(models)):
        plt.plot(barplots[m], c=numpy.random.rand(3,), label=models[m].name)

    plt.ylabel('Memorability of frames')
    plt.xlabel(VIDEO_NAME)
    plt.ylim([0.0, 1.0])
    plt.xlim([0, len(list_item)-1])
    leg = plt.legend(loc='upper left')
    plt.show()
