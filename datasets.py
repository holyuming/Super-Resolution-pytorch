import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import glob
import os
import cv2
import numpy as np
import random
import time


class datasetSR(Dataset):
    def __init__(self, lq_paths, gt_paths, transform=None, valid=False) -> None:
        super().__init__()
        random.seed(time.time())
        self.lq_paths   = lq_paths
        self.gt_paths   = gt_paths
        self.transform  = transform
        self.valid      = valid

        self.imagelist1 = []
        self.imagelist2 = []
        for lq_path in lq_paths:
            self.imagelist1 = self.imagelist1 + glob.glob(os.path.join(lq_path, '*'))
        for gt_path in gt_paths:
            self.imagelist2 = self.imagelist2 + glob.glob(os.path.join(gt_path, '*'))

        # self.imagelist1 = sorted(self.imagelist1)
        # self.imagelist2 = sorted(self.imagelist2)
        assert (len(self.imagelist1) == len(self.imagelist2))
        for idx in range(len(self.imagelist1)):
            path1 = os.path.basename(self.imagelist1[idx]).replace('x3', '')
            path2 = os.path.basename(self.imagelist2[idx])        
            assert path1 == path2, f'{path1} not match {path2}.'

        print(f'Number of LR imgs (patches): {str(len(self.imagelist1)):6s}, Number of HR imgs (patches): {str(len(self.imagelist2)):6s}')


    def __getitem__(self, index):
        img_lq = cv2.imread(self.imagelist1[index], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_gt = cv2.imread(self.imagelist2[index], cv2.IMREAD_COLOR).astype(np.float32) / 255.

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float()

        img_gt = np.transpose(img_gt if img_gt.shape[2] == 1 else img_gt[:, :, [2, 1, 0]], (2, 0, 1))   # HWC-BGR to CHW-RGB
        img_gt = torch.from_numpy(img_gt).float()

        if self.valid:
            # assert os.path.basename(self.imagelist1[index]) == os.path.basename(self.imagelist2[index]) # can be only used when Set5 & Set14, cuz they share the same img name
            img_name = os.path.basename(self.imagelist1[index])
            return img_name, img_lq, img_gt

        # random horizontal flip
        if random.randint(0, 1) == 0:
            img_lq = torch.flip(img_lq, dims=[2])
            img_gt = torch.flip(img_gt, dims=[2])

        # random vertical flip
        if random.randint(0, 1) == 0:
            img_lq = torch.flip(img_lq, dims=[1])
            img_gt = torch.flip(img_gt, dims=[1])

        # random permute H W
        if random.randint(0, 1) == 0:
            img_lq = img_lq.permute(0, 2, 1)
            img_gt = img_gt.permute(0, 2, 1)

        return img_lq, img_gt


    def __len__(self):
        return len(self.imagelist1)
        


if __name__ == '__main__':

    batch_size = 64
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    lq_paths = ['/data/SR/div2k/LRbicx3', '/data/SR/flickr2k/LRbicx3', '/data/SR/div2k/LRunkx3', '/data/SR/flickr2k/LRunkx3']
    gt_paths= ['/data/SR/div2k/original', '/data/SR/flickr2k/original', '/data/SR/div2k/original', '/data/SR/flickr2k/original']
    ds = datasetSR(lq_paths=lq_paths, gt_paths=gt_paths)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=16)

    iteration = 0
    t1 = time.time()
    for lq, gt in dl:
        print(lq.shape, gt.shape, iteration)
        iteration += 1
        # assert lq.shape == gt.shape
    t2 = time.time()
    print("time:", t2 - t1)