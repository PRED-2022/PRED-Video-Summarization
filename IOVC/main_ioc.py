import config_ioc
import argparse
import os
from train_ioc import train_model
from test_ioc import test_model
import torch


phases_list = ["train", "test"]

current_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='PyTorch MSINet')
parser.add_argument("phase", metavar="PHASE", choices=phases_list,
                    help="sets the network phase (allowed: train or test)")

parser.add_argument("-p", "--path",
                    help="specify the path where training data will be \
                          downloaded to or test data is stored")

parser.add_argument("-w", "--weights", metavar="WEIGHTS",
                    help="define where to find the weights of the model to use")

args = parser.parse_args()


if __name__ == "__main__":
    dims = (300, 400)
    print(dims)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    if args.phase == "train":
        train_model(args, dims, config_ioc.PARAMS)
    if args.phase == "test":
        test_model(args, dims)
