from torch.utils.data import Dataset
import torch
import os
from pretrain.main import args
from torchvision import transforms
from PIL import Image
import numpy as np
import random
# 数据处理
transform = transforms.Compose([
    transforms.Resize((235, 235)),
    transforms.CenterCrop(args.crop_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])

transformI = transforms.Compose([
    transforms.Resize((235, 235)),
    transforms.CenterCrop(args.crop_size),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])

transformII = transforms.Compose([
    transforms.Resize((235, 235)),
    transforms.CenterCrop(args.crop_size),
    transforms.RandomRotation((-10, 10)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])


class LoadDatasets(Dataset):
    def __init__(self, ids):
        self.fileroot = args.pretrain_data_dir
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.transform1 = transformI
        self.transform2 = transformII
        self.d = ids
        self.choose_neg = args.choose_neg

        for i in ids:
            for j in ids[i]:
                for k in range(ids[i][j]):
                    self.listImagePaths.extend([args.pretrain_data_dir + str(i) + '_' + str(j) + '_' + str(int(k) + 1) + '.jpg'])

                    self.listImageLabels.append(i)

    def sample(self, i, f1, f2, b1, b2):

        fore1 = self.listImagePaths[f1]
        fore2 = self.listImagePaths[f2]
        behind1 = self.listImagePaths[b1]
        behind2 = self.listImagePaths[b2]

        img_fore1 = torch.FloatTensor(self.transform(Image.open(fore1).convert('L')))
        img_fore2 = torch.FloatTensor(self.transform(Image.open(fore2).convert('L')))
        img_meanfore = (img_fore1 + img_fore2) / 2
        img_mf = transforms.functional.to_pil_image(img_meanfore)
        img_meanfore_1 = torch.FloatTensor(self.transform1(img_mf))
        img_meanfore_2 = torch.FloatTensor(self.transform2(img_mf))

        img_behind1 = torch.FloatTensor(self.transform(Image.open(behind1).convert('L')))
        img_behind2 = torch.FloatTensor(self.transform(Image.open(behind2).convert('L')))
        img_meanbehind = (img_behind1 + img_behind2) / 2
        img_mb = transforms.functional.to_pil_image(img_meanbehind)
        img_meanbehind_1 = torch.FloatTensor(self.transform1(img_mb))
        img_meanbehind_2 = torch.FloatTensor(self.transform2(img_mb))

        Mean = (img_meanfore + img_meanbehind) / 2
        M = transforms.functional.ToPILImage(Mean)
        Mean_1 = torch.FloatTensor(self.transform1(M))
        Mean_2 = torch.FloatTensor(self.transform2(M))

        list01 = [img_meanfore, img_meanfore_1, img_meanfore_2, img_meanbehind, img_meanbehind_1, img_meanbehind_2,
                  Mean, Mean_1, Mean_2]
        for item in list01:
            i.append(item)

        return i

    def __getitem__(self, index):
        i = []
        imagePath = self.listImagePaths[index]

        image = np.asarray(Image.open(imagePath).convert('L'))

        img = self.transform(image)
        img1 = self.transform1(image)
        img2 = self.transform2(image)
        img = torch.FloatTensor(img)
        img1 = torch.FloatTensor(img1)
        img2 = torch.FloatTensor(img2)
        i.append(img)
        i.append(img1)
        i.append(img2)

        imageLabel = self.listImageLabels[index]

        none, patient, video, shot = imagePath.split('_')
        patient = os.path.split(patient)[1]
        shot = os.path.splitext(shot)[0]

        sumshot = self.d[int(patient)][int(video)]

        if sumshot == 1:
            i = self.sample(i, index, index, index, index)
        else:
            if int(shot) - 2 <= 0:
                if int(shot) - 2 == 0:
                    if int(shot) + 1 == sumshot:
                        i = self.sample(i, index + 1, index + 1, index - 1, index - 1)
                    elif int(shot) + 1 > sumshot:
                        i = self.sample(i, index, index, index - 1, index - 1)
                    else:
                        i = self.sample(i, index + 1, index + 2, index - 1, index - 1)

                else:
                    if int(shot) + 1 == sumshot:
                        i = self.sample(i, index + 1, index + 1, index, index)
                    elif int(shot) + 1 > sumshot:
                        i = self.sample(i, index, index, index, index)
                    else:
                        i = self.sample(i, index + 1, index + 2, index, index)


            elif int(shot) + 1 >= sumshot:
                if int(shot) + 1 == sumshot:
                    if int(shot) - 2 == 0:
                        i = self.sample(i, index + 1, index + 1, index - 1, index - 1)
                    elif int(shot) - 2 < 0:
                        i = self.sample(i, index + 1, index + 1, index, index)
                    else:
                        i = self.sample(i, index + 1, index + 1, index - 1, index - 2)
                else:
                    if int(shot) - 2 == 0:
                        i = self.sample(i, index, index, index - 1, index - 1)
                    elif int(shot) - 2 < 0:
                        i = self.sample(i, index, index, index, index)
                    else:
                        i = self.sample(i, index, index, index - 1, index - 2)

            else:
                i = self.sample(i, index + 1, index + 2, index - 1, index - 2)

        key = []
        value = []

        n = len(self.d.keys())
        N = np.arange(n)
        a = list(set(N) - set(np.array([int(patient)])))

        if len(self.d[int(patient)]) != 1:
            for k, v in self.d[int(patient)].items():
                if int(k) != int(video):
                    key.append(int(k))
                    value.append(int(v))
            cho = int(self.choose_neg / 11) + 5
            for j in range(cho):
                ch_k = random.choice(key)
                ch_v = random.randint(1, value[key.index(ch_k)])
                ne = self.fileroot + str(patient) + '_' + str(ch_k) + '_' + str(
                    ch_v) + '.jpg'
                ne1 = Image.open(ne).convert('L')
                ne1 = self.transform2(ne1)
                ne1 = torch.FloatTensor(ne1)
                i.append(ne1)

            negative_labels = random.sample(a, int((self.choose_neg / 11) * 10))

            for label in negative_labels:
                mid = np.random.choice(len(self.d[label])) + 1
                last = np.random.choice(self.d[label][mid]) + 1
                neg1 = self.fileroot + str(label) + '_' + str(mid) + '_' + str(
                    last) + '.jpg'
                neg1 = Image.open(neg1).convert('L')
                neg1 = self.transform2(neg1)
                neg1 = torch.FloatTensor(neg1)
                i.append(neg1)
            imgdata = torch.stack(i, dim=0)

        else:
            for n in range(5):
                nm = Image.open(imagePath).convert('L')
                nm = self.transform2(nm)
                nm = torch.FloatTensor(nm)
                i.append(nm)

            negative_labels = random.sample(a, self.choose_neg)

            for label in negative_labels:
                mid = np.random.choice(len(self.d[label])) + 1
                last = np.random.choice(self.d[label][mid]) + 1
                neg1 = self.fileroot + str(label) + '_' + str(mid) + '_' + str(
                    last) + '.jpg'
                neg1 = Image.open(neg1).convert('L')
                neg1 = self.transform2(neg1)
                neg1 = torch.FloatTensor(neg1)
                i.append(neg1)
            imgdata = torch.stack(i, dim=0)

        return imgdata, imageLabel

    def __len__(self):
        return len(self.listImageLabels)

