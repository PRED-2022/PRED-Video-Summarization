from PIL import Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset


class ResizeAndPad:
    def __init__(self, output_dims):
        self.out_h = output_dims[0]
        self.out_w = output_dims[1]

    def __call__(self, image):
        w, h = image.size
        h_rate = h / self.out_h
        w_rate = w / self.out_w

        if h_rate > w_rate:
            new_w = (w * self.out_h) // h
            out_img = F.resize(image, [self.out_h, new_w])
            p_right_left = (self.out_w - new_w) // 2
            padding = [p_right_left, 0, p_right_left, 0]
        else:
            new_h = (h * self.out_w) // w
            out_img = F.resize(image, [new_h, self.out_w])
            p_top_bottom = (self.out_h - new_h) // 2
            padding = [0, p_top_bottom, 0, p_top_bottom]
        return F.pad(out_img, padding, 0)


class ImgIOCDataset(Dataset):
    def __init__(self, img_list, ioc_list, transform, normalize):
        self.img_list = img_list
        self.ioc_list = ioc_list
        self.transform = transform
        self.normalize = normalize

        assert len(self.img_list) == len(self.ioc_list), \
            "Number of files do not match: Images: {}, IOC values: {}".format(len(self.img_list), len(self.ioc_list))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        ioc_score = self.ioc_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
            img = self.normalize(img)
        return img, ioc_score


class IOC_TestDataset(Dataset):
    def __init__(self, img_list, transform):
        self.img_list = img_list
        self.transform = transform

    def len(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img

