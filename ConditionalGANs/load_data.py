import os
import os.path as osp
import sys
import numpy as np
import pickle
from PIL import Image
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import time

import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import scipy.io as io
import scipy.misc as misc
import glob
import csv
from skimage import color
from transform import ReLabel, ToLabel, ToSP, Scale

import cv2
import os
import glob
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SC2_Dataset(data.Dataset):
    def __init__(self, root, shuffle=False, small=False, mode='test', transform=None, target_transform=None, types='', show_ab=False, large=False, loader=pil_loader):

        tic = time.time()
        self.root = root
        self.loader = loader
        self.image_transform = transform
        if large:
            self.size = 480
            self.imgpath = glob.glob(root + 'img_480/*.png')
        else:
            self.size = 224
            self.imgpath = glob.glob(root + 'img/*.jpg')
        self.types = types

        # read split
        self.train_file = set()
        with open(self.root + 'train_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.train_file.add(str(row[0]).zfill(6))


        self.test_file = set()
        with open(self.root + 'test_split.csv', 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.test_file.add(str(row[0]).zfill(6))

        self.path = []
        if mode == 'train':
            print(list(self.train_file)[:10])
            for item in self.imgpath:
                if item.split('/')[-1][6:6+6] in self.train_file:
                    self.path.append(item)
            print("len train:", len(self.path))

        elif mode == 'test':
            print(list(self.test_file)[:10])
            for item in self.imgpath:
                if item.split('/')[-1][6:6+6] in self.test_file:
                    self.path.append(item)
            print("len test:", len(self.path))

        self.path = sorted(self.path)


        np.random.seed(0)
        if shuffle:
            perm = np.random.permutation(len(self.path))
            self.path = [self.path[i] for i in perm]

        print('Load %d images, used %fs' % (self.path.__len__(), time.time()-tic))

    def __getitem__(self, index):
        mypath = self.path[index]
        img = self.loader(mypath) # PIL Image


        cv2img = cv2.imread(mypath)
        blurred = cv2.GaussianBlur(cv2img, (3, 3), 0)
        auto = auto_canny(blurred)
        edges = 255 - auto
        edges = edges[:, :, np.newaxis]
        edges = np.concatenate((edges, edges, edges), axis=2)

        edges = np.array(edges)
        if (edges.shape[0] != self.size) or (edges.shape[1] != self.size):
            edges = misc.imresize(edges, (self.size, self.size))

        cv2_img_lab = color.rgb2lab(np.array(edges))

        cv2_img = torch.FloatTensor(np.transpose(edges, (2, 0, 1)))
        cv2_img_lab = torch.FloatTensor(np.transpose(cv2_img_lab, (2, 0, 1)))
        cv2_img_l = torch.unsqueeze(cv2_img_lab[0], 0) / 100.



        img = np.array(img)
        if (img.shape[0] != self.size) or (img.shape[1] != self.size):
            img = misc.imresize(img, (self.size, self.size))

        img_lab = color.rgb2lab(np.array(img)) # np array


        img = (img - 127.5) / 127.5  # -1 to 1
        img = torch.FloatTensor(np.transpose(img, (2,0,1)))
        img_lab = torch.FloatTensor(np.transpose(img_lab, (2,0,1)))

        img_l = torch.unsqueeze(img_lab[0],0) / 100. # L channel 0-100
        img_ab = (img_lab[1::] + 0) / 110. # ab channel -110 - 110



        if self.types == 'raw':
            return img_l, img
        else:
            return img_l, img_ab

    def __len__(self):
        return len(self.path)


# if __name__ == '__main__':
#     data_root = '/home/users/u5612799/DATA/SCReplay/'
#     # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#     #                                  std=[0.229, 0.224, 0.225])
#
#     image_transform = transforms.Compose([
#                               transforms.CenterCrop(224),
#                               transforms.ToTensor(),
#                           ])
#
#     lfw = SC2_Dataset(data_root, mode='train',
#                       transform=image_transform, large=True, types='raw')
#
#     data_loader = data.DataLoader(lfw,
#                                   batch_size=1,
#                                   shuffle=False,
#                                   num_workers=4)
#
#     for i, (data, target) in enumerate(data_loader):
#         print(i, len(lfw))
