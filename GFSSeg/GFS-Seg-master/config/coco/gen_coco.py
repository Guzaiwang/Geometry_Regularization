import os
import tqdm

coco_train_file = './train_data_list.txt'
coco_test_file = './val_data_list.txt'


new_coco_train_file = './coco_train_data_list.txt'
new_coco_test_file = './coco_test_data_list.txt'


new_coco_train_list = open(new_coco_train_file, 'w')
with open(coco_train_file, 'r') as train_files:
    all_training_files = train_files.readlines()
    for training_file in tqdm.tqdm(all_training_files[:]):
        img_path, label_path = training_file.split(' ')[0], training_file.split(' ')[1].strip()
        new_img_path = 'train2014/' + img_path.split('/')[-1]
        new_label_path = 'train_contain_crowd/' + label_path.split('/')[-1]

        new_coco_train_list.writelines(new_img_path+' '+new_label_path+'\n')
new_coco_train_list.close()

new_coco_test_list = open(new_coco_test_file, 'w')
with open(coco_test_file, 'r') as test_files:
    all_test_files = test_files.readlines()
    for test_file in tqdm.tqdm(all_test_files[:]):
        img_path, label_path = test_file.split(' ')[0], test_file.split(' ')[1].strip()
        new_img_path = 'val2014/' + img_path.split('/')[-1]
        new_label_path = 'val_contain_crowd/' + label_path.split('/')[-1]


        new_coco_test_list.writelines(new_img_path+' '+new_label_path+'\n')
new_coco_test_list.close()