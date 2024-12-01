import os
import os.path as osp
import numpy as np
import random
import cv2
from collections import defaultdict

from .base_dataset import BaseDataset

class GFSSegTrain(BaseDataset):
    num_classes = 20
    def __init__(self, root, list_path, fold, shot=1, mode='train', crop_size=(512, 512),
             ignore_label=255, base_size=(2048,512), resize_label=False, filter=False, seed=123):
        super(GFSSegTrain, self).__init__(mode, crop_size, ignore_label, base_size=base_size)
        assert mode in ['train', 'val_supp']
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.shot = shot
        self.mode = mode
        self.resize_label = resize_label
        self.img_dir = 'JPEGImages'
        self.lbl_dir = 'SegmentationClassAug'

        if fold == -1:
            # training with all classes
            self.base_classes = set(range(1, self.num_classes+1))
            self.novel_classes = set()
            with open(os.path.join(self.list_path), 'r') as f:
                self.data_list = f.read().splitlines()
        else:
            interval = self.num_classes // 4
            # base classes = all classes - novel classes
            self.base_classes = set(range(1, self.num_classes + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
            # novel classes
            self.novel_classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))

            filter_flag = True if (self.mode == 'train' and filter) else False
            list_dir = os.path.dirname(self.list_path)
            list_dir = list_dir + '/fold%s'%fold
            if filter_flag:
                list_dir = list_dir + '_filter'
            list_saved = os.path.exists(os.path.join(list_dir, 'train_fold%s.txt'%fold))
            if list_saved:
                print('id files exist...')
                self.novel_cls_to_ids = defaultdict(list)
                with open(os.path.join(list_dir, 'train_fold%s.txt'%fold), 'r') as f:
                    self.data_list = f.read().splitlines()
                for cls in self.novel_classes:
                    with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'r') as f:
                        self.novel_cls_to_ids[cls] = f.read().splitlines()
                with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s.txt'%(fold, shot, seed)), 'r') as f:
                    self.novel_id_list = f.read().splitlines()
            else:
                '''
                fold0/train_fold0.txt: training images containing base classes (novel classes will be ignored during training)
                fold0/train_novel_class[0-4].txt: training images containing novel class [0-4] (to provide support images for validation)
                '''
                with open(os.path.join(self.list_path), 'r') as f:
                    self.ids = f.read().splitlines()
                print('checking ids...')
                
                self.data_list, self.novel_cls_to_ids = self._filter_and_map_ids(filter_intersection=filter_flag)
                if not os.path.exists(list_dir):
                    os.makedirs(list_dir)
                with open(os.path.join(list_dir, 'train_fold%s.txt'%fold), 'w') as f:
                    for id in self.data_list:
                        f.write(id+"\n")
                for cls in self.novel_classes:
                    with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'w') as f:
                        for id in self.novel_cls_to_ids[cls]:
                            f.write(id+"\n")

    def __len__(self):
        if self.mode == 'val_supp':
            return len(self.novel_classes)
        else:
            return len(self.data_list)

    def _convert_label(self, label):
        new_label = label.copy()
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        assert len(label_class) > 0
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)
        for c in label_class:
            if c in base_list:
                new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
            elif c in novel_list:
                if self.mode == 'train':
                    new_label[label == c] = 0
                else:
                    new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
                    # print('label {}'.format(novel_list.index(c) + len(base_list) + 1))
        return new_label

    def __getitem__(self, index):
        if self.mode == 'val_supp':
            return self._get_val_support(index)
        else:
            return self._get_train_sample(index)

    def _get_train_sample(self, index):
        id = self.data_list[index]
        image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)

        label = self._convert_label(label)
        # date augmentation & preprocess
        image, label = self.resize(image, label, random_scale=True)
        image, label = self.random_flip(image, label)
        image, label = self.crop(image, label)
        image = self.normalize(image)
        image, label = self.pad(self.crop_size, image, label)
        image, label = self.totensor(image, label)

        return image, label, id

    def _get_val_support(self, index):
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)

        target_cls = novel_list[index]
        novel_cls_id = index + len(base_list) + 1

        id_s_list, image_s_list, label_s_list = [], [], []

        for k in range(self.shot):
            id_s = self.novel_id_list[index*self.shot+k]              
            image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id_s), cv2.IMREAD_COLOR)
            label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_s), cv2.IMREAD_GRAYSCALE)

            label = self._convert_label(label)
            # date augmentation & preprocess
            image, label = self.resize(image, label)
            image = self.normalize(image)
            image, label = self.pad(self.crop_size, image, label)
            image, label = self.totensor(image, label)
            id_s_list.append(id_s)
            image_s_list.append(image)
            label_s_list.append(label)
        # print(target_cls)
        # print(id_s_list)
        return image_s_list, label_s_list, id_s_list, novel_cls_id

    def _filter_and_map_ids(self, filter_intersection=False):
        image_label_list = []
        novel_cls_to_ids = defaultdict(list)
        for i in range(len(self.ids)):
            mask = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%self.ids[i]), cv2.IMREAD_GRAYSCALE)
            label_class = np.unique(mask).tolist()
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)
            valid_base_classes = set(np.unique(mask)) & self.base_classes
            valid_novel_classes = set(np.unique(mask)) & self.novel_classes

            if valid_base_classes:
                if filter_intersection:
                    if set(label_class).issubset(self.base_classes):
                        image_label_list.append(self.ids[i])
                else:
                    image_label_list.append(self.ids[i])

            if valid_novel_classes:
            # remove images whose valid objects are all small (according to PFENet)
                new_label_class = []
                for cls in valid_novel_classes:
                    if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                        new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        novel_cls_to_ids[cls].append(self.ids[i])

        return image_label_list, novel_cls_to_ids

class GFSSegVal(BaseDataset):
    num_classes = 20
    def __init__(self, root, list_path, fold, crop_size=(512, 512),
             ignore_label=255, base_size=(2048,512), resize_label=False, use_novel=True, use_base=True):
        super(GFSSegVal, self).__init__('val', crop_size, ignore_label, base_size=base_size)
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.resize_label = resize_label
        self.use_novel = use_novel
        self.use_base = use_base
        self.img_dir = 'JPEGImages'
        self.lbl_dir = 'SegmentationClassAug'

        if fold == -1:
            self.base_classes = set(range(1, self.num_classes+1))
            self.novel_classes = set()
        else:
            interval = self.num_classes // 4
            # base classes = all classes - novel classes
            self.base_classes = set(range(1, self.num_classes + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
            # novel classes
            self.novel_classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))

        with open(os.path.join(self.list_path), 'r') as f:
            self.ids = f.read().splitlines()
#         self.ids = ['2007_005273', '2011_003019']
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)
        label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)

        new_label = label.copy()
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)

        for c in label_class:
            if c in base_list:
                if self.use_base:
                    new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
                else:
                    new_label[label == c] = 0
            elif c in novel_list:
                if self.use_novel:
                    if self.use_base:
                        new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
                    else:
                        new_label[label == c] = (novel_list.index(c) + 1)
                else:
                    new_label[label == c] = 0

        label = new_label.copy()
        # date augmentation & preprocess
        if self.resize_label:
            image, label = self.resize(image, label)
            image = self.normalize(image)
            image, label = self.pad(self.base_size, image, label)
        else:
            image = self.resize(image)
            image = self.normalize(image)
            image = self.pad(self.base_size, image)
        image, label = self.totensor(image, label)

        return image, label, id


class GFSSegTrain_agg(BaseDataset):
    num_classes = 20
    def __init__(self, root, list_path, fold, shot=1, mode='train', crop_size=(512, 512),
             ignore_label=255, base_size=(2048,512), resize_label=False, filter=False, seed=123):
        super(GFSSegTrain_agg, self).__init__(mode, crop_size, ignore_label, base_size=base_size)
        assert mode in ['train', 'val_supp']
        self.root = root
        self.list_path = list_path
        self.fold = fold
        self.shot = shot
        self.mode = mode
        self.resize_label = resize_label
        self.img_dir = 'JPEGImages'
        self.lbl_dir = 'SegmentationClassAug'

        if fold == -1:
            # training with all classes
            self.base_classes = set(range(1, self.num_classes+1))
            self.novel_classes = set()
            with open(os.path.join(self.list_path), 'r') as f:
                self.data_list = f.read().splitlines()
        else:
            interval = self.num_classes // 4
            # base classes = all classes - novel classes
            self.base_classes = set(range(1, self.num_classes + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
            # novel classes
            self.novel_classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))

            filter_flag = True if (self.mode == 'train' and filter) else False
            list_dir = os.path.dirname(self.list_path)
            list_dir = list_dir + '/fold%s'%fold
            if filter_flag:
                list_dir = list_dir + '_filter'
            list_saved = os.path.exists(os.path.join(list_dir, 'train_fold%s.txt'%fold))
            if list_saved:
                print('id files exist...')
                self.novel_cls_to_ids = defaultdict(list)
                with open(os.path.join(list_dir, 'train_fold%s.txt'%fold), 'r') as f:
                    self.data_list = f.read().splitlines()
                for cls in self.novel_classes:
                    with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'r') as f:
                        self.novel_cls_to_ids[cls] = f.read().splitlines()
                with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s.txt'%(fold, shot, seed)), 'r') as f:
                    self.novel_id_list = f.read().splitlines()
            else:
                '''
                fold0/train_fold0.txt: training images containing base classes (novel classes will be ignored during training)
                fold0/train_novel_class[0-4].txt: training images containing novel class [0-4] (to provide support images for validation)
                '''
                with open(os.path.join(self.list_path), 'r') as f:
                    self.ids = f.read().splitlines()
                print('checking ids...')
                
                self.data_list, self.novel_cls_to_ids = self._filter_and_map_ids(filter_intersection=filter_flag)
                if not os.path.exists(list_dir):
                    os.makedirs(list_dir)
                with open(os.path.join(list_dir, 'train_fold%s.txt'%fold), 'w') as f:
                    for id in self.data_list:
                        f.write(id+"\n")
                for cls in self.novel_classes:
                    with open(os.path.join(list_dir, 'train_novel_class%s.txt'%cls), 'w') as f:
                        for id in self.novel_cls_to_ids[cls]:
                            f.write(id+"\n")

    def __len__(self):
        if self.mode == 'val_supp':
            return len(self.novel_classes)
        else:
            return len(self.data_list)

    def _convert_label(self, label):
        new_label = label.copy()
        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        assert len(label_class) > 0
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)
        for c in label_class:
            if c in base_list:
                new_label[label == c] = (base_list.index(c) + 1)    # 0 as background
            elif c in novel_list:
                if self.mode == 'train':
                    new_label[label == c] = 0
                else:
                    new_label[label == c] = (novel_list.index(c) + len(base_list) + 1)
                    # print('label {}'.format(novel_list.index(c) + len(base_list) + 1))
        return new_label

    def __getitem__(self, index):
        if self.mode == 'val_supp':
            return self._get_val_support(index)
        else:
            return self._get_train_sample(index)

    def _get_train_sample(self, index):
        id = self.data_list[index]
        image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id), cv2.IMREAD_COLOR)

    
        label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id), cv2.IMREAD_GRAYSCALE)

        label = self._convert_label(label)
        # date augmentation & preprocess
        image, label = self.resize(image, label, random_scale=True)
        image, label = self.random_flip(image, label)
        image, label = self.crop(image, label)
        image = self.normalize(image)
        image, label = self.pad(self.crop_size, image, label)
        image, label = self.totensor(image, label)

        image_denomalized = image.detach().numpy()
        # canny
        # segtruth = auto_canny(image_denomalized.astype(np.uint8)) / 256.

        # SIFT
        _, segtruth  = extract_Sift_feature(image_denomalized.astype(np.uint8)) 
        segtruth = segtruth / 256.

        return image, label, id, segtruth

    def _get_val_support(self, index):
        # with open(os.path.join(list_dir, 'fold%s_%sshot_seed%s.txt'%(fold, shot, seed)), 'r') as f:
                #     self.novel_id_list = f.read().splitlines()
        base_list = list(self.base_classes)
        novel_list = list(self.novel_classes)

        target_cls = novel_list[index]
        novel_cls_id = index + len(base_list) + 1

        id_s_list, image_s_list, label_s_list = [], [], []

        for k in range(self.shot):
            id_s = self.novel_id_list[index*self.shot+k]              
            image = cv2.imread(osp.join(self.root, self.img_dir, '%s.jpg'%id_s), cv2.IMREAD_COLOR)
            label = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%id_s), cv2.IMREAD_GRAYSCALE)

            label = self._convert_label(label)
            # date augmentation & preprocess
            image, label = self.resize(image, label)
            image = self.normalize(image)
            image, label = self.pad(self.crop_size, image, label)
            image, label = self.totensor(image, label)
            id_s_list.append(id_s)
            image_s_list.append(image)
            label_s_list.append(label)
        # print(target_cls)
        # print(id_s_list)
        return image_s_list, label_s_list, id_s_list, novel_cls_id

    def _filter_and_map_ids(self, filter_intersection=False):
        image_label_list = []
        novel_cls_to_ids = defaultdict(list)
        for i in range(len(self.ids)):
            mask = cv2.imread(osp.join(self.root, self.lbl_dir, '%s.png'%self.ids[i]), cv2.IMREAD_GRAYSCALE)
            label_class = np.unique(mask).tolist()
            if 0 in label_class:
                label_class.remove(0)
            if 255 in label_class:
                label_class.remove(255)
            valid_base_classes = set(np.unique(mask)) & self.base_classes
            valid_novel_classes = set(np.unique(mask)) & self.novel_classes

            if valid_base_classes:
                if filter_intersection:
                    if set(label_class).issubset(self.base_classes):
                        image_label_list.append(self.ids[i])
                else:
                    image_label_list.append(self.ids[i])

            if valid_novel_classes:
            # remove images whose valid objects are all small (according to PFENet)
                new_label_class = []
                for cls in valid_novel_classes:
                    if np.sum(np.array(mask) == cls) >= 16 * 32 * 32:
                        new_label_class.append(cls)

                if len(new_label_class) > 0:
                    # map each valid class to a list of image ids
                    for cls in new_label_class:
                        novel_cls_to_ids[cls].append(self.ids[i])

        return image_label_list, novel_cls_to_ids
    


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged




def draw_gaussian(heatmap, center_x, center_y, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x = center_x
    y = center_y

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def extract_Sift_feature(img, gaussian_radius= 5, show_case=False):
    '''

    :param image_path: input image
    :param show_case: plt.imshow(img) or not
    :return: sift keypoint binary mask and gaussian heatmap
    '''

    # img = cv2.imread(image_path)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img, None)
    # generate binary mask (0 for background, 1 for sift keypoint)
    binary_mask = np.zeros(shape=(img.shape[0], img.shape[1]))
    gaussian_mask = np.zeros(shape=(img.shape[0], img.shape[1]))
    for k_p in kp1:
        binary_mask[int(k_p.pt[1]), int(k_p.pt[0])] = 1
        draw_gaussian(gaussian_mask, int(k_p.pt[0]), int(k_p.pt[1]), radius=gaussian_radius)
    '''
    print(np.count_nonzero(binary_mask))
    print(np.count_nonzero(gaussian_mask), np.unique(gaussian_mask))
    629
    59674 [0.00000000e+00 5.88451217e-04 2.24472210e-03 6.35920926e-03
     8.56277826e-03 1.33792593e-02 2.09049649e-02 2.42580135e-02
     5.10368881e-02 6.87219964e-02 7.97446503e-02 9.25352812e-02
     1.44585493e-01 2.25913453e-01 2.62148806e-01 3.04196123e-01
     4.75303537e-01 5.51539774e-01 7.42657239e-01 8.61775631e-01
     1.00000000e+00]
    '''

    return binary_mask* 255. , gaussian_mask* 255.

