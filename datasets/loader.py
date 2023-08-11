import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from torchvision import transforms

from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
    H, W, _ = imgs[0].shape
    resize_img = imgs
    if H < 256 or W < 256:
        for i in range(len(imgs)):
            resize_img[i] = cv2.resize(imgs[i], (320, 320))

        imgs = resize_img
        H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]
    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    # horizontal flip
    if random.randint(0, 1) == 1:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1)

    if not only_h_flip:
        # bad data augmentations for outdoor
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def test_mode(imgs=[], size=256):
    transposed_image = []
    H, W, _ = imgs[0].shape
    if H > W:
        # 交换宽度和高度
        for i in range(len(imgs)):
            transposed_image.append(np.transpose(imgs[i], (1, 0, 2)))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    resize_img = imgs
    if H < 256 or W < 256:
        for i in range(len(imgs)):
            resize_img[i] = cv2.resize(imgs[i], (320, 320))

        imgs = resize_img
        H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


class PairLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False, data_type='dehaze'):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip
        self.data_type = data_type

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        img_name = self.img_names[idx]
        # read image, and scale [0, 1] to [-1, 1]
        degraded_img_path = self.root_dir + '/' + self.data_type + '/' + img_name
        gt_img_path = self.root_dir + '/GT/' + img_name
        source_img = read_img(degraded_img_path) * 2 - 1
        target_img = read_img(gt_img_path) * 2 - 1
        if source_img is None:
            print('No such image: {}'.format(degraded_img_path))
        if target_img is None:
            print('No such image: {}'.format(gt_img_path))

        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.only_h_flip)

        if self.mode == 'valid':
            [source_img, target_img] = align([source_img, target_img], self.size)

        if self.mode == 'test':
            [source_img, target_img] = test_mode([source_img, target_img], self.size)

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # read image, and scale [0, 1] to [-1, 1]
        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}
