import os
import torch
from torchvision.transforms import ToTensor, Compose
from datasets_ioc import ResizeAndPad
from model_IOC import IOCNet
from PIL import Image
import csv

OUTPUT_CSV_PATH = "path/to/output.csv"


def test_model(args, dims):

    # Set the gpu device if available, cpu if not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    imgs_test = [os.path.join(args.path, file) for file in os.listdir(args.path) if
                 os.path.isfile(os.path.join(args.path, file))]

    assert len(imgs_test) > 0, "No images were found to predict"

    preprocess = Compose([
        ResizeAndPad(dims),
        ToTensor(),
    ])

    model = IOCNet(37, 50)
    model.load_state_dict(torch.load(args.weights))
    model.to(device).eval()

    output_data = []
    with torch.no_grad():
        for img_path in imgs_test:
            img = Image.open(img_path).convert('RGB')
            img = preprocess(img)
            img = img.to(device)
            predicted_ioc = model(img)
            predicted_ioc = predicted_ioc.squeeze().detach().cpu().numpy()
            print(predicted_ioc)
            _output_data = [img_path, predicted_ioc]
            output_data.append(_output_data)

    header = ["image_name", "ioc"]
    with open(OUTPUT_CSV_PATH, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(output_data)

    return 0