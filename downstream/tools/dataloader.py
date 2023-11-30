from torchvision import transforms
from downstream.parser_args2 import args
from torch.utils.data import Dataset
import csv,os,torch
from PIL import Image

transformI = transforms.Compose([
    transforms.Resize(args.crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-10, 10)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3], std=[0.22])])


class Data(Dataset):
    def __init__(self,syu_labels,syu_data_dir):
        self.listImagePaths = []
        self.listImageLabels = []
        self.label = []
        self.transform = transformI
        self.conver = args.channels
        self.syu_labels = syu_labels
        self.syu_data_dir = syu_data_dir

        with open(self.syu_labels, "r") as f:
            cr = csv.reader(f)
            row = [r for r in cr][1:]
            self.listImagePaths.extend([os.path.join(self.syu_data_dir, r[0]) for r in row])
            self.listImageLabels.extend([[int(r[1])] for r in row])
            self.label.extend([int(r[1]) for r in row])

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert(self.conver)
        img1 = self.transform(imageData)
        img = torch.FloatTensor(img1)
        imageLabel = torch.LongTensor(self.listImageLabels[index])
        return img, imageLabel

    def __len__(self):
        return len(self.listImageLabels)

    def label(self):
        return self.label
