import os
import re
import glob
import copy
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from albumentations import Compose, RandomBrightnessContrast,HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur,Resize
from albumentations.pytorch.functional import img_to_tensor
import cv2
import logging
class ReadDataset():
    """
        initialize once and can be reused as train/test/validation set
    """

    def __init__(self,dataset_name):

        oversample=True
        self.data = {
            "train": [],
            "val":[],
            "test": [],
        }
        self.labels = copy.deepcopy(self.data)

        # compression_version = ''

        # if 'FF++' in dataset_name:
        #     compression_version = dataset_name.split('_')[1]
        #     dataset_name=dataset_name.split('_')[0]


        self.path=f'datasets/{dataset_name}'

        # if oversample:
        #     if 'FF++' in dataset_name:
        #         dataset_path=f'{self.path}/{compression_version}_oversample_dataset.npz'
        #     else:
        #         dataset_path = f'{self.path}/oversample_dataset.npz'
        # else:
        #     if 'FF++' in dataset_name:
        #         dataset_path=f'{self.path}/{compression_version}_dataset.npz'
        #     else:
        #         dataset_path = f'{self.path}/dataset.npz'

        # if os.path.exists(dataset_path):
        #     file = np.load(dataset_path, allow_pickle=True, mmap_mode='r')
        #     self.data = file['data'].tolist()
        #     self.labels = file['labels'].tolist()
        #
        # else:
        #     self.read_txt(oversample=oversample,compression_version=compression_version)
        #     np.savez(dataset_path, data=self.data, labels=self.labels)

        self.read_txt(oversample=oversample)
        logging.info(f"fake data: {sum(self.labels['train'])}, real data: {len(self.labels['train'])-sum(self.labels['train'])}")

    def read_txt(self,oversample=False):

        dataset_files=[os.path.join(self.path,'test_fake.txt'),os.path.join(self.path,'test_real.txt'),
                        os.path.join(self.path,'val_fake.txt'),os.path.join(self.path,'val_real.txt'),
                        os.path.join(self.path,'train_fake.txt'),os.path.join(self.path,'train_real.txt')]
        # dataset_files= glob.glob(f"{self.path}")
        if 'FF++' in self.path or 'DFDC-Preview' in self.path:
            balance_ratio=4
        elif 'Celeb-DF-v2' in self.path:
            balance_ratio = 6
        elif 'DFDC' in self.path:
            balance_ratio = 5
        else:
            # unbalance fake and real
            balance_ratio = 1

        for file in dataset_files:
            with open(file, "r") as f:
                lines = f.readlines()
            if '/test_' in file:
                key='test'
            elif '/val_' in file:
                key='val'
            elif '/train_' in file:
                key = 'train'
            for i in range(len(lines)):
                # start clearing duplicates
                raw = re.sub("\s", "", lines[i]).split(",")
                paths = os.listdir(raw[1]) #video name
                for row in paths:
                    path_dir = os.path.join(raw[1], row)
                    if os.path.isfile(path_dir) and '.png' in path_dir:
                        if oversample and 'train_real' in file:
                            for i in range(balance_ratio):
                                self.data[key].append(path_dir)
                                self.labels[key].append(int(raw[0]))
                        else:
                            self.data[key].append(path_dir)
                            self.labels[key].append(int(raw[0]))

    def get_dataset(self, mode):
        return self.data[mode], self.labels[mode]

class MyDataset(Dataset):

    def __init__(self,
                 data,
                 label,
                 size=224,
                 normalize={"mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]},
                 test=False):
        super().__init__()
        self.size=size
        self.data=data
        self.label = label
        self.normalize = normalize
        self.aug=self.create_train_aug()
        self.transform=self.transform_all()
        self.test=test

    def create_train_aug(self):
        return Compose([
            ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
            GaussNoise(p=0.1),
            GaussianBlur(blur_limit=3, p=0.05),
            HorizontalFlip(),
            PadIfNeeded(min_height=self.size, min_width=self.size, border_mode=cv2.BORDER_CONSTANT),
            OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
            ToGray(p=0.2),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        ]
        )
    def transform_all(self):
        return Resize(p=1, height=self.size, width=self.size)
    def __getitem__(self, idx):

        # img = Image.open(self.data[idx])
        img = cv2.imread(self.data[idx], cv2.IMREAD_COLOR)
        data = self.transform(image=img)
        img = data["image"]
        if not self.test:
            data = self.aug(image=img)
            img = data["image"]

        img=img_to_tensor(img,self.normalize)
        return img,self.label[idx]

    def __len__(self):
        return len(self.data)

def mixup_data(x, y, alpha=0.5, use_cuda=False):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    a=criterion(pred, y_a)
    b=criterion(pred, y_b)
    losses=[]
    try:
        for i in range(len(a)):
            losses.append(lam * a[i]  + (1 - lam) * b[i])
    except:
        return lam * a  + (1 - lam) * b
    return losses
