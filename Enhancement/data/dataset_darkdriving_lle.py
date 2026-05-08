import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import torch.nn.functional as F
import random
import cv2
import numpy as np
import glob
import os
import functools

def read_img(env, path, size=None):

    if env is None:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        print(img)
        if size is not None:
            img = cv2.resize(img, (size[0], size[1]))

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_img_seq(path, size=None):

    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    img_l = [read_img(None, v, size) for v in img_path_l]

    imgs = np.stack(img_l, axis=0)
    try:
        imgs = imgs[:, :, :, [2, 1, 0]]
    except Exception:
        import ipdb; ipdb.set_trace()
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs


def cmp(x, y):
    x_index = x.split('/')[-1]
    y_index = y.split('/')[-1]
    x_index = int(x_index)
    y_index = int(y_index)
    if x_index > y_index:
        return 1
    else:
        return -1


class VideoSameSizeDataset(data.Dataset):
    def __init__(self, opt):
        super(VideoSameSizeDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.half_N_frames = opt['N_frames'] // 2
        self.root = opt['dataroot']

        self.data_type = self.opt['data_type']
        self.img_size = self.opt['train_size']
        print(f'training size: {self.img_size}')
        self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}
        if self.data_type == 'lmdb':
            raise ValueError('No need to use LMDB during validation/test.')

        self.imgs_LQ, self.imgs_GT = {}, {}
        subfolders_LQ = []
        subfolders_GT = []
        subfolders_NA = []
        day_folder = osp.join(self.root, 'day')
        night_folder = osp.join(self.root, 'night')

        if not osp.exists(day_folder) or not osp.exists(night_folder):
            raise ValueError('day or night folder not found in dataroot')

        day_files = sorted([f for f in util.glob_file_list(day_folder) if f.endswith('.jpg')])
        night_files = sorted([f for f in util.glob_file_list(night_folder) if f.endswith('.jpg')])

        if len(day_files) != len(night_files):
            raise ValueError('Number of day and night files do not match')


        for day_file, night_file in zip(day_files, night_files):
            base_name = osp.basename(day_file)
            if osp.basename(night_file) != base_name:
                print(f'Warning: filenames do not match: {day_file} vs {night_file}')

            subfolders_LQ.append(night_file)
            subfolders_GT.append(day_file)
            subfolders_NA.append(base_name)

        count = 0
        for subfolder_LQ, subfolder_GT, subfolder_name in zip(subfolders_LQ, subfolders_GT, subfolders_NA):
            img_paths_LQ = [subfolder_LQ]
            img_paths_GT = [subfolder_GT]

            max_idx = len(img_paths_LQ)
            self.data_info['path_LQ'].extend(img_paths_LQ)
            self.data_info['path_GT'].extend(img_paths_GT)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            self.data_info['idx'].append('{}/{}'.format(count, len(subfolders_LQ)))
            count += 1
            if self.cache_data:

                self.imgs_LQ[subfolder_name] = img_paths_LQ
                self.imgs_GT[subfolder_name] = img_paths_GT
        print('total images: {}'.format(len(self.imgs_LQ)))

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        img_LQ_path = self.imgs_LQ[folder][0]
        img_GT_path = self.imgs_GT[folder][0]
        img_LQ_path = [img_LQ_path]
        img_GT_path = [img_GT_path]

        if self.opt['phase'] == 'train':
            img_LQ = util.read_img_seq(img_LQ_path, [self.img_size[0], self.img_size[1]])
            img_GT = util.read_img_seq(img_GT_path, [self.img_size[0], self.img_size[1]])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

            img_LQ_l = [img_LQ]
            img_LQ_l.append(img_GT)
            rlt = util.augment_torch(img_LQ_l, self.opt['use_flip'], self.opt['use_rot'])
            img_LQ = rlt[0]
            img_GT = rlt[1]

            save_img = img_LQ.permute(1, 2, 0).cpu().numpy()
            save_img = (save_img * 255.0).round().astype(np.uint8)
            cv2.imwrite('img_LQ.png', cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

            save_img = img_GT.permute(1, 2, 0).cpu().numpy()
            save_img = (save_img * 255.0).round().astype(np.uint8)
            cv2.imwrite('img_GT.png', cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR))

        elif self.opt['phase'] == 'test':
            img_LQ = util.read_img_seq(img_LQ_path, [self.img_size[0], self.img_size[1]])
            img_GT = util.read_img_seq(img_GT_path, [self.img_size[0], self.img_size[1]])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

        else:
            img_LQ = util.read_img_seq(img_LQ_path, [self.img_size[0], self.img_size[1]])
            img_GT = util.read_img_seq(img_GT_path, [self.img_size[0], self.img_size[1]])
            img_LQ = img_LQ[0]
            img_GT = img_GT[0]

        img_nf = img_LQ.permute(1, 2, 0).numpy() * 255.0
        img_nf = cv2.blur(img_nf, (5, 5))
        img_nf = img_nf * 1.0 / 255.0
        img_nf = torch.Tensor(img_nf).float().permute(2, 0, 1)

        return {
            'LQs': img_LQ,
            'GT': img_GT,
            'nf': img_nf,
            'folder': folder,
            'idx': self.data_info['idx'][index],
            'border': 0,
            'LQ_path': img_LQ_path[0],
        }

    def __len__(self):
        return len(self.data_info['path_LQ'])
