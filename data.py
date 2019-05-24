import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
class MyDataset(Dataset):
    """Super resolution dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: dir of unlabel data.
            transform: costumized transform.
        """
        self.root_dir = root_dir
        self.transform = transform
        li_HR = []
        li_LR = []
        self.HR_path = self.root_dir+"/HR/"
        self.LR_path = self.root_dir+"/LR/"
        self.transforms_HR_12 = transforms.Compose([transforms.ToTensor()])
        #self.file_name_list = []
        for filename_HR in os.listdir(self.HR_path):
            li_HR.append(filename_HR) #e.g: file_name = 云G22222.jpg
        for filename_LR in os.listdir(self.LR_path):
            li_LR.append(filename_LR)
        self.file_name_list_HR = li_HR
        self.file_name_list_LR = li_LR
    def __len__(self):
        return len(self.file_name_list_LR)

    def __getitem__(self, idx):
        img_name_LR = os.path.join(self.LR_path,
                                self.file_name_list_LR[idx])
        #print(img_name)
        img_name_HR = os.path.join(self.HR_path,
                                self.file_name_list_HR[idx])
        input_LR = cv2.imread(img_name_LR)
        input_HR = cv2.imread(img_name_HR)
        #print("HR:",type(input_HR))
        #print("LR:",type(input_LR))
        input_LR = cv2.resize(input_LR,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
        input_HR_12 = cv2.resize(input_HR,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_NEAREST)
        #normalize
        '''input_LR = cv2.normalize(input_LR, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        input_HR =  cv2.normalize(input_HR, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        input_HR_12 = cv2.normalize(input_HR_12, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)'''
        sample = {'LR':input_LR, 'HR': input_HR, 'HR/2':input_HR_12}
        if self.transform:
            sample['LR'] = self.transform(sample['LR'])
            sample['HR'] = self.transform(sample['HR'])
        sample['HR/2'] = self.transforms_HR_12(sample['HR/2'])#donot normalize the label(ground truth)
        return sample['LR'],sample['HR'],sample['HR/2']
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        in_img, out_img, half_HR = sample['LR'], sample['HR'], sample['HR/2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        in_img = in_img.transpose((2, 0, 1))
        out_img = out_img.transpose((2,0,1))
        half_HR = half_HR.transpose((2, 0, 1))
        return {'LR': torch.from_numpy(in_img),
                'HR': torch.from_numpy(out_img),'HR/2':torch.from_numpy(half_HR)}
