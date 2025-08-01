import os
import cv2
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset

from models.utils import process_resize, frame2tensor
from models.LGutils import resize_image, numpy_image_to_torch

class ImagePairDataset(Dataset):
    def __init__(self, input_dir, pairs):
        self.input_dir = Path(input_dir)
        print('Looking for data in directory \"{}\"'.format(input_dir))
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Get names of the current image pairs
        name0, name1 = self.pairs[idx][:2]
        if len(self.pairs[idx]) >= 5:
            rot0, rot1 = int(self.pairs[idx][2]), int(self.pairs[idx][3])
        else:
            rot0, rot1 = 0, 0
        # Load base images with no modifications
        base_image0 = cv2.imread(str(self.input_dir / name0))
        base_image1 = cv2.imread(str(self.input_dir / name1))
        # Resize images for yolo prediction
        yoloimg0 = cv2.resize(base_image0, (640, 640))
        yoloimg1 = cv2.resize(base_image1, (640, 640))
        # Greyscale transformed images
        image0 = cv2.cvtColor(base_image0, cv2.COLOR_BGR2GRAY)
        image1 = cv2.cvtColor(base_image1, cv2.COLOR_BGR2GRAY)
        w0, h0 = image0.shape[1], image0.shape[0]
        w0_new, h0_new = process_resize(w0, h0, (640, 480))
        w1, h1 = image1.shape[1], image1.shape[0]
        w1_new, h1_new = process_resize(w1, h1, (640, 480))
        scales0 = (float(w0) / float(w0_new), float(h0) / float(h0_new))
        scales1 = (float(w1) / float(w1_new), float(h1) / float(h1_new))
        image0 = cv2.resize(image0, (w0_new, h0_new)).astype('float32')
        image1 = cv2.resize(image1, (w1_new, h1_new)).astype('float32')
        if rot0 != 0:
            image0 = np.rot90(image0, k=rot0)
            if rot0 % 2:
                scales0 = scales0[::-1]
        if rot1 != 0:
            image1 = np.rot90(image1, k=rot1)
            if rot1 % 2:
                scales1 = scales1[::-1]
        # greyscale tensors
        inp0 = frame2tensor(image0, "cpu")
        inp1 = frame2tensor(image1, "cpu")
        # rgb tensors
        rgb0, _ = resize_image(base_image0[..., ::-1], (640, 480))
        rgb1, _ = resize_image(base_image1[..., ::-1], (640, 480))
        rgb0 = numpy_image_to_torch(rgb0).to("cpu")
        rgb1 = numpy_image_to_torch(rgb1).to("cpu")

        # Need to convert inp0-1 and rgb0-1 to gpu tensors after batch is pulled

        return self.pairs[idx], (image0, image1), (inp0, inp1), (scales0, scales1), (rgb0, rgb1), (yoloimg0, yoloimg1)
