import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights


class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class VGGEncoder(nn.Module):
    def __init__(self, feat_shape_h, feat_shape_w):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.upsample_2 = nn.UpsamplingBilinear2d(size=(feat_shape_h, feat_shape_w))
        self.upsample_3 = nn.UpsamplingBilinear2d(size=(feat_shape_h, feat_shape_w))


    def forward(self, x):
        x1 = self.features[:19](x)
        x2 = self.features[:28](x)
        x3 = self.features(x)
        x2 = self.upsample_2(x2)
        x3 = self.upsample_2(x3)

        return torch.cat((x1,x2,x3), dim=1)


class ConcatConvNet(nn.Module):
    def __init__(self):
        super(ConcatConvNet, self).__init__()
        self.conv1 = BasicConv2D(1280, 320, 3, 1, padding=1)
        self.conv2 = BasicConv2D(320, 64, 3, 1, padding=1)
        self.conv3 = BasicConv2D(64, 1, 3, 1, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.flatten = nn.Flatten()
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout2(x)
        return x


class RegressorNet(nn.Module):
    def __init__(self):
        super(RegressorNet, self).__init__()
        self.dense_1 = nn.Linear(1850, 1024)
        self.dense_2 = nn.Linear(1024, 256)
        self.dense_3 = nn.Linear(256, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.relu(x)
        x = self.dense_2(x)
        x = self.relu(x)
        x = self.dense_3(x)
        x = self.relu(x)

        return x


class IOCNet(nn.Module):
    def __init__(self, feat_shape_h, feat_shape_w):
        super(IOCNet, self).__init__()
        self.vggnet = VGGEncoder(feat_shape_h, feat_shape_w)
        for param in self.vggnet.parameters():
            param.requires_grad = False
        self.iocnet_model = nn.Sequential(
            self.vggnet,
            ConcatConvNet(),
            RegressorNet()
        )

    def forward(self, x):
        x = self.iocnet_model(x)
        return x
