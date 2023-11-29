from torch.utils.data import Dataset
import torch
import os
import SimpleITK as sitk
import re
from DEMO.main import args
from torchvision import transforms
'''
    数据增强:
    1、totensor可以作用在数组和PIL格式数据上，转化成tensor，会把HWC数据转化为CHW；但如果 PIL Image 属于 (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) 中的一种图像类型，
       或者 numpy.ndarray 格式数据类型是 np.uint8 ，则将 [0, 255] 的数据转为 [0.0, 1.0] ，也就是说将所有数据除以 255 进行归一化。
    2、Centercrop啥的操作必须作用在tensor上，不能作用在数组上
    3、如果数据是numpy格式的则需要先转为tensor，使用ToTensor转，再接transforms变换；如果数据是PIL格式的则无先后顺序
'''
transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop((224), scale = (0.5,1.0)),
    transforms.RandomHorizontalFlip(),
])

def totensor(numpy):
    return torch.FloatTensor(numpy)

def LoadDatasets(args):
    if args.mode == 'train':
        with open(args.train_data_txt,'r') as f:
            tmp = f.read()
            train_,eval_ = tmp.split(' ')
            train = train_.split(',')
            eval = eval_.split(',')

        train_datasets = dataloader(args.train_data_dir, train)
        eval_datasets = dataloader(args.eval_data_dir, eval)

        return train_datasets, eval_datasets
    elif args.mode == 'test':
        with open(args.test_data_txt, 'r') as f:
            tmp = f.read()
            test_, _ = tmp.split(' ')
            test = test_.split(',')

        test_datasets = dataloader(args.test_data_dir, test)

        return test_datasets, _


class dataloader(Dataset):
    def __init__(self, root, datalist):
        self.root = root
        self.list = datalist

    def dcm2tensor(self, path):
        '''医学影像读取方法'''
        txt = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(txt)
        tensor = totensor(arr / 1000)
        return tensor

    def __getitem__(self, index):
        p, noi, n = self.list[index].split('_')

        noise_path = os.path.join(self.root, p, noi, n + '.dcm')
        noise = self.dcm2tensor(noise_path)

        dirs = ' '.join(os.listdir(os.path.join(self.root, str(p))))
        normal = re.findall(r'2\.886 x 600 \S*\s\S*', dirs)[0]
        label = self.dcm2tensor(os.path.join(self.root, p, normal, n + '.dcm'))

        return noise, label

    def __len__(self):
        return len(self.list)
