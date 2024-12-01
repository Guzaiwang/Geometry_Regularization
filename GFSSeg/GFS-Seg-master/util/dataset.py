import os
import os.path
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

# manual_seed=123
# torch.manual_seed(manual_seed)
# np.random.seed(manual_seed)
# torch.cuda.manual_seed(manual_seed)
# torch.cuda.manual_seed_all(manual_seed)
# random.seed(manual_seed)
# os.environ['PYTHONHASHSEED'] = str(manual_seed) 


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None, data_split=0, shot=10, seed=123, \
    sub_list=None, sub_val_list=None):
    assert split in ['train', 'val', 'val_supp']
    manual_seed = seed
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    torch.cuda.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    random.seed(manual_seed)    
    data_split=data_split

    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    sub_class_file_list = {}
    for sub_c in sub_val_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])              
        item = (image_name, label_name)

        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        if len(label_class) == 0:
            continue

        flag = 'keep'  
        for c in label_class:
            if c in sub_val_list:
                sub_class_file_list[c].append((image_name, label_name))
                flag = 'drop'
                break  

        if flag == 'keep' and split != 'val_supp':
            item = (image_name, label_name)
            image_label_list.append(item)
    print("Checking Pretrain image&label pair {} list {} done!".format(split, data_split))
    print("All {} pairs in base classes.".format(len(image_label_list)))

    supp_image_label_list = []
    if (split == 'train' or split == 'val_supp'):
        shot = shot
        for c in sub_val_list:
            sub_class_file = sub_class_file_list[c]
            num_file = len(sub_class_file)
            output_data_list = []
            select_list = []
            num_trial = 0
            while(len(select_list) < shot):
                num_trial += 1
                if num_trial >= num_file:
                    print('class {} skip with {} shots'.format(c, len(select_list)))
                    raw_select_list = select_list.copy()
                    for re in range(shot - len(select_list)):
                        rand_select_idx = raw_select_list[random.randint(0,len(raw_select_list)-1)]
                        select_list.append(rand_select_idx)
                        supp_image_label_list.append(sub_class_file[rand_select_idx])
                        output_data_list.append(sub_class_file[rand_select_idx][1].split('/')[-1])                            
                    break
                rand_idx = random.randint(0,num_file-1)
                if rand_idx in select_list:
                    continue
                else:              
                    label = cv2.imread(sub_class_file[rand_idx][1], cv2.IMREAD_GRAYSCALE)
                    label_class = np.unique(label).tolist()
                    label_class.remove(c) 
                    if 0 in label_class:
                        label_class.remove(0)
                    if 255 in label_class:
                        label_class.remove(255)       

                    skip_flag = 0
                    for new_c in label_class:
                        if new_c in sub_val_list:
                            skip_flag = 1
                            break
                    if skip_flag:
                        continue
                    tmp_label = np.zeros_like(label)
                    target_pix = np.where(label == c)
                    if target_pix[0].shape[0] >= 16 * 32 * 32 and target_pix[1].shape[0] >= 16 * 32 * 32:                       
                        select_list.append(rand_idx)
                        supp_image_label_list.append(sub_class_file[rand_idx])
                        output_data_list.append(sub_class_file[rand_idx][1].split('/')[-1])
                    else:
                        continue
    else:
        ### for 'val' mode that evaluates all images
        for c in sub_val_list:
            sub_class_file = sub_class_file_list[c]
            for idx in range(len(sub_class_file)):
                supp_image_label_list.append(sub_class_file[idx])           

    image_label_list = supp_image_label_list + image_label_list
    print("Checking image&label pair {} list done!".format(split))
    print("All {} pairs in novel classes.".format(len(supp_image_label_list)))
    print("All {} pairs in base + novel classes.".format(len(image_label_list)))
    return image_label_list, supp_image_label_list


class SemData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, \
        transform=None, data_split=0, shot=10, seed=123, \
        use_coco=False, val_shot=10, saved_path='./saved_npy_bs6/'):

        self.saved_path = saved_path

        if use_coco:
            print('INFO: using COCO')
            self.class_list = list(range(1, 81))
            if data_split == 3:
                self.sub_val_list = list(range(4, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
            elif data_split == 2:
                self.sub_val_list = list(range(3, 80, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif data_split == 1:
                self.sub_val_list = list(range(2, 79, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif data_split == 0:
                self.sub_val_list = list(range(1, 78, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))  
            elif data_split == 11:
                self.sub_list = list(range(41, 81)) 
                self.sub_val_list = list(range(1, 41))                  
            elif data_split == 10:
                self.sub_list = list(range(1, 41)) 
                self.sub_val_list = list(range(41, 81))                 
            self.base_class_num = len(self.sub_list) + 1
            self.novel_class_num = len(self.sub_val_list)
             
        else:
            # use PASCAL VOC | 0-20 + 255
            print('INFO: using PASCAL VOC')
            if data_split == 3:  
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = [16,17,18,19,20]
            elif data_split == 2:
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = [11,12,13,14,15]
            elif data_split == 1:
                self.sub_list = [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [6,7,8,9,10]
            elif data_split == 0:
                self.sub_list = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [1,2,3,4,5]                
            self.base_class_num = len(self.sub_list) + 1
            self.novel_class_num = len(self.sub_val_list)

        print('sub_list: ',self.sub_list)
        print('sub_val_list: ',self.sub_val_list)
        print('Base_num: {} (including class 0), Novel_num: {}'.format(self.base_class_num, self.novel_class_num))
        save_dir = self.saved_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if use_coco:
            path_np_data_list = str(split) + '_split_' + str(data_split) + '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_data_list_coco.npy'
            path_np_supp_list = str(split) + '_split_' + str(data_split) + '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_supp_list_coco.npy'
        else:
            path_np_data_list = str(split) + '_split_' + str(data_split)+ '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_data_list.npy'
            path_np_supp_list = str(split) + '_split_' + str(data_split)+ '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_supp_list.npy'
        path_np_data_list = os.path.join(save_dir, path_np_data_list)
        path_np_supp_list = os.path.join(save_dir, path_np_supp_list)
        if not os.path.exists(path_np_data_list):
            print('[{}] Creating new lists and will save to **{}** and **{}**'.format(split, path_np_data_list, path_np_supp_list))
            self.data_list, self.supp_image_label_list = make_dataset(split, data_root, data_list, data_split=data_split, shot=shot, seed=seed, \
                                            sub_list=self.sub_list, sub_val_list=self.sub_val_list)  
            np_data_list = np.array(self.data_list)
            np_supp_list = np.array(self.supp_image_label_list)
            np.save(path_np_data_list, np_data_list)
            np.save(path_np_supp_list, np_supp_list)
        else:
            print('[{}] Loading saved lists from **{}** and **{}**'.format(split, path_np_data_list, path_np_supp_list))
            self.data_list = list(np.load(path_np_data_list))
            self.supp_image_label_list = list(np.load(path_np_supp_list))

        print('Processing data list {} with {} shots.'.format(data_split, shot))
        self.data_split=data_split
        self.transform = transform
        self.split = split
        self.use_coco = use_coco

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]                   
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('{} does not exist.'.format(image_path))

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
            
        raw_label = label.copy()
        for c in label_class:
            x,y = np.where(raw_label == c)
  
            if c in self.sub_list:
                label[x[:], y[:]] = (self.sub_list.index(c) + 1)    # ignore the background in sublist, + 1
            elif c in self.sub_val_list:
                label[x[:], y[:]] = (self.sub_val_list.index(c) + self.base_class_num)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        raw_size = torch.Tensor(label.shape[:])
        raw_label = label.copy()
        raw_label_mask = np.zeros((1024, 1024))
        raw_label_mask[:raw_label.shape[0], :raw_label.shape[1]] = raw_label.copy()
        raw_label_mask = torch.Tensor(raw_label_mask)
        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, raw_size, raw_label_mask


class SemData_agg(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, \
        transform=None, data_split=0, shot=10, seed=123, \
        use_coco=False, val_shot=10, saved_path='./saved_npy_bs6/'):

        self.saved_path = saved_path

        if use_coco:
            print('INFO: using COCO')
            self.class_list = list(range(1, 81))
            if data_split == 3:
                self.sub_val_list = list(range(4, 81, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))                    
            elif data_split == 2:
                self.sub_val_list = list(range(3, 80, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif data_split == 1:
                self.sub_val_list = list(range(2, 79, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))    
            elif data_split == 0:
                self.sub_val_list = list(range(1, 78, 4))
                self.sub_list = list(set(self.class_list) - set(self.sub_val_list))  
            elif data_split == 11:
                self.sub_list = list(range(41, 81)) 
                self.sub_val_list = list(range(1, 41))                  
            elif data_split == 10:
                self.sub_list = list(range(1, 41)) 
                self.sub_val_list = list(range(41, 81))                 
            self.base_class_num = len(self.sub_list) + 1
            self.novel_class_num = len(self.sub_val_list)
             
        else:
            # use PASCAL VOC | 0-20 + 255
            print('INFO: using PASCAL VOC')
            if data_split == 3:  
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = [16,17,18,19,20]
            elif data_split == 2:
                self.sub_list = [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = [11,12,13,14,15]
            elif data_split == 1:
                self.sub_list = [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [6,7,8,9,10]
            elif data_split == 0:
                self.sub_list = [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = [1,2,3,4,5]                
            self.base_class_num = len(self.sub_list) + 1
            self.novel_class_num = len(self.sub_val_list)

        print('sub_list: ',self.sub_list)
        print('sub_val_list: ',self.sub_val_list)
        print('Base_num: {} (including class 0), Novel_num: {}'.format(self.base_class_num, self.novel_class_num))
        save_dir = self.saved_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if use_coco:
            path_np_data_list = str(split) + '_split_' + str(data_split) + '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_data_list_coco.npy'
            path_np_supp_list = str(split) + '_split_' + str(data_split) + '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_supp_list_coco.npy'
        else:
            path_np_data_list = str(split) + '_split_' + str(data_split)+ '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_data_list.npy'
            path_np_supp_list = str(split) + '_split_' + str(data_split)+ '_seed_' + str(seed) + '_shot_' + str(val_shot) + '_np_supp_list.npy'
        path_np_data_list = os.path.join(save_dir, path_np_data_list)
        path_np_supp_list = os.path.join(save_dir, path_np_supp_list)
        if not os.path.exists(path_np_data_list):
            print('[{}] Creating new lists and will save to **{}** and **{}**'.format(split, path_np_data_list, path_np_supp_list))
            self.data_list, self.supp_image_label_list = make_dataset(split, data_root, data_list, data_split=data_split, shot=shot, seed=seed, \
                                            sub_list=self.sub_list, sub_val_list=self.sub_val_list)  
            np_data_list = np.array(self.data_list)
            np_supp_list = np.array(self.supp_image_label_list)
            np.save(path_np_data_list, np_data_list)
            np.save(path_np_supp_list, np_supp_list)
        else:
            print('[{}] Loading saved lists from **{}** and **{}**'.format(split, path_np_data_list, path_np_supp_list))
            self.data_list = list(np.load(path_np_data_list))
            self.supp_image_label_list = list(np.load(path_np_supp_list))

        print('Processing data list {} with {} shots.'.format(data_split, shot))
        self.data_split=data_split
        self.transform = transform
        self.split = split
        self.use_coco = use_coco

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path = self.data_list[index]                   
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('{} does not exist.'.format(image_path))

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # GRAY 1 channel ndarray with shape H * W

        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
            
        raw_label = label.copy()
        for c in label_class:
            x,y = np.where(raw_label == c)
  
            if c in self.sub_list:
                label[x[:], y[:]] = (self.sub_list.index(c) + 1)    # ignore the background in sublist, + 1
            elif c in self.sub_val_list:
                label[x[:], y[:]] = (self.sub_val_list.index(c) + self.base_class_num)

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        raw_size = torch.Tensor(label.shape[:])
        raw_label = label.copy()
        raw_label_mask = np.zeros((1024, 1024))
        raw_label_mask[:raw_label.shape[0], :raw_label.shape[1]] = raw_label.copy()
        raw_label_mask = torch.Tensor(raw_label_mask)
        if self.transform is not None:
            image, label = self.transform(image, label)
        
        image_denomalized = image.detach().numpy()
        _, segtruth  = extract_Sift_feature(image_denomalized.astype(np.uint8)) 
        segtruth = segtruth / 256.
        # segtruth = auto_canny(image_denomalized.astype(np.uint8)) / 256.

        return image, label, raw_size, raw_label_mask, segtruth

    # 快速排序
    # 给下面的代码写注释

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

    #提取图像的canny edge
def extract_canny_edge(img):
    # compute the median of the single channel pixel intensities
    # 初始化SIFT检测器
    img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edged = auto_canny(img)
    return edged
    

def auto_sift(image):
    # compute the median of the single channel pixel intensities
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
    return img_with_keypoints


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


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

