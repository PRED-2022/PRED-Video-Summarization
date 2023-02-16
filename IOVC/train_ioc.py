import torch
import sys
from torch.utils.data import DataLoader
from datasets_ioc import ImgIOCDataset, ResizeAndPad
from torchvision.transforms import ToTensor, Compose, Normalize
from model_IOC import IOCNet
from utils import load_data_ioc
import torch.nn as nn


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batch = len(dataloader)
    train_loss = 0
    for batch, (img_batch, ioc_batch) in enumerate(dataloader):
        img_batch, ioc_batch = img_batch.to(device), ioc_batch.to(device)
        ioc_batch = ioc_batch.type(torch.float32)
        ioc_batch = ioc_batch.unsqueeze(1)

        pred_ioc = model(img_batch)
        loss = loss_fn(pred_ioc, ioc_batch)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(img_batch)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= num_batch
    print(f"Average train loss: {train_loss:>8f} \n")
    return train_loss


def valid_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    counter = 0
    with torch.no_grad():
        for img, ioc in dataloader:
            img, ioc = img.to(device), ioc.to(device)
            ioc = ioc.type(torch.float32)
            ioc = ioc.unsqueeze(1)
            pred = model(img)
            test_loss += loss_fn(pred, ioc).item()
            counter += 1

    test_loss /= num_batches
    print(f"Test Error: \n Average loss: {test_loss:>8f} \n")
    return test_loss


def train_model(args, dims, params):

    # Set the gpu device if available, cpu if not
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # Get the paths for training and validation images and saliency maps
    img_train, img_valid, ioc_train, ioc_valid = load_data_ioc(args.path, split_ratio=0.8)

    kwargs = {"batch_size": params["batch_size"],
              "shuffle": params["shuffle"]}
    if torch.cuda.is_available():
        cuda_kwargs = {
            "num_workers": 1,
            "pin_memory": True
        }
        kwargs.update(cuda_kwargs)

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])

    train_dataloader = DataLoader(
        ImgIOCDataset(img_train, ioc_train, Compose([
            ResizeAndPad(dims),
            ToTensor(),
        ]), normalize),
        **kwargs)
    valid_dataloader = DataLoader(
        ImgIOCDataset(img_valid, ioc_valid, Compose([
            ResizeAndPad(dims),
            ToTensor(),
        ]), normalize),
        **kwargs)

    model = IOCNet(37, 50).to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), params["learning_rate"])

    # Loads pretrained weights, if exitst:
    if params["use_pretrain_weights"]:
        model.load_state_dict(torch.load(args.weights))

    current_val_loss = sys.maxsize
    for epoch in range(params["n_epochs"]):
        print(f"Epoch {epoch+1}\n------------------------------------")
        train_loss = train_loop(train_dataloader, model, loss, optimizer, device)
        val_loss = valid_loop(valid_dataloader, model, loss, device)
        print("Val loss: {} | Current best Val Loss {}".format(val_loss, current_val_loss))
        if val_loss <= current_val_loss:
            print(f"Saving best weights....\n")
            torch.save(model.state_dict(), './weights/model_weights.pth')
            current_val_loss = val_loss

    return 0