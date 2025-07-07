import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import math
import pickle
from . import detectors
from loguru import logger

class CELEBV_HQ(Dataset):
    def __init__(
        self, 
        seq_path, 
        iscrop=True, 
        crop_size=224,
        scale=1.25, 
        face_detector='fan',
        is_train=True,
        useVislmk=False
    ):
        self.celebvhq_seq_path = seq_path
        self.imagepath_list = sorted([
            os.path.join(seq_path, imgname) for imgname in os.listdir(seq_path)
            if os.path.isfile(os.path.join(seq_path, imgname)) and os.path.join(seq_path, imgname).endswith('jpg')
        ])

        self.num_frames = len(self.imagepath_list)
        pid = self.imagepath_list[0].split('/')[-2]
        logger.info(f"Found CelebV_HQ-{pid}, Total {self.num_frames} frames")

        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        self.face_detector_type = face_detector
        self.useVislmk = useVislmk
        if face_detector == 'fan':
            self.face_detector = detectors.FAN()
        else:
            print(f'please check the detector: {face_detector}')
            exit()
        self.is_train = is_train


    def __len__(self):
        return self.num_frames


    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        else:
            raise NotImplementedError
        return old_size, center


    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]

        multiview_imagepath = []
        multiview_imagename = []
        multiview_image = []
        
        multiview_dst_image = []
        multiview_ori_image = []
        multiview_kpt = []

        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))

        multiview_imagepath.append(imagepath)
        multiview_imagename.append(imagename)
        multiview_image.append(image)

        for imagepath, imagename, image in zip(multiview_imagepath, multiview_imagename, multiview_image):
            if len(image.shape) == 2:
                image = image[:, :, None].repeat(1, 1, 3)
            if len(image.shape) == 3 and image.shape[2] > 3:
                image = image[:, :, :3]

            h, w, _ = image.shape
            if self.iscrop:
                kpt_npypath = os.path.splitext(imagepath)[0] + '_kpt.npy'
                bbox_txtpath = os.path.splitext(imagepath)[0] + '_bbox.txt'
                # if there is no annotation, run face detector to find bounding box of face
                if os.path.exists(kpt_npypath) and os.path.exists(bbox_txtpath):
                    kpt = np.load(kpt_npypath)  # ndarray [68, 3]
                    bbox = np.loadtxt(bbox_txtpath)  # list
                    left, top, right, bottom = bbox
                    old_size, center = self.bbox2point(left, right, top, bottom, type='kpt68')
                else:
                    bbox, kpt, bbox_type = self.face_detector.run(image)
                    if len(bbox) < 4:
                        print('no face detected! run original image')
                        left, right, top, bottom = 0, h - 1, 0, w - 1
                    else:
                        left, top, right, bottom = bbox
                    old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                    # save landmark for faster loading later
                    np.save(kpt_npypath, kpt)
                    np.savetxt(bbox_txtpath, [bbox], fmt='%.1f')
                size = int(old_size * self.scale)
                src_pts = np.array([
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2]
                ])
            else:
                src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

            DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            image = image / 255.

            dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
            multiview_ori_image.append(image)

            dst_image = dst_image.transpose(2, 0, 1)
            multiview_dst_image.append(dst_image)

            # kpt
            cropped_kpt = np.dot(
                tform.params,
                np.hstack([kpt[:, :2], np.ones([kpt.shape[0], 1])]).T
            ).T
            # normalized kpt
            cropped_kpt[:, :2] = cropped_kpt[:, :2] / self.resolution_inp * 2 - 1

            if self.useVislmk:
                # add visibility score
                cropped_kpt[:, 2] = kpt[:, 2]

            multiview_kpt.append(cropped_kpt)

        multiview_dst_image = np.asarray(multiview_dst_image)
        multiview_ori_image = np.asarray(multiview_ori_image)
        return_dict = {
            'image': torch.tensor(multiview_dst_image).float(),
            'imagename': imagename,
            'tform': torch.tensor(tform.params).float(),
            'landmark': torch.from_numpy(np.array(multiview_kpt)).type(dtype=torch.float32)
        }
        if not self.is_train:
            return_dict['original_image'] = torch.tensor(multiview_ori_image).float()
        return return_dict
