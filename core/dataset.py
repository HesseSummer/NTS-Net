import numpy as np
import cv2
import os
from PIL import Image
from torchvision import transforms
import pickle
from config import INPUT_SIZE, img_dir
from transforms import my_transforms1, my_transforms2


def get_allimgnames():
    """获得所有图片名称列表"""
    img_name_list = []
    with open(os.path.join(img_dir, 'images.txt')) as f:
        for line in f:
            img_name_list.append(line[:-1].split(' ')[-1])
    return img_name_list
def get_alllabels():
    """获得所有标签列表"""
    labels = []
    with open(os.path.join(img_dir, 'image_class_labels.txt')) as f:
        for line in f:
            labels.append(line[:-1].split(' '[-1]))
    return labels
def get_split():
    """获得训练集和测试集的划分列表，1代表训练集、0代表验证集"""
    train_test_split = []
    with open(os.path.join(img_dir, 'train_test_split.txt')) as f:
        for line in f:
            train_test_split.append(int(line[:-1].split(' ')[-1]))
    return train_test_split

def get_splitnames():
    """获得训练集和测试集图片列表"""
    save_path = os.path.join(img_dir, 'train_name_list.txt')
    if not os.path.exists(save_path):
        img_name_list = get_allimgnames()
        train_test_split = get_split()
        train_name_list = [img for img, split in zip(img_name_list, train_test_split) if split]
        with open(save_path, 'wb') as f:
            pickle.dump(train_name_list, f)
    else:
        with open(save_path, 'rb') as f:
            train_name_list = pickle.load(f)

    save_path = os.path.join(img_dir, 'test_name_list.pkl')
    if not os.path.exists(save_path):
        img_name_list = get_allimgnames()
        train_test_split = get_split()
        test_name_list = [img for img, split in zip(img_name_list, train_test_split) if not split]
        with open(save_path, 'wb+') as f:
            pickle.dump(test_name_list, f)
    else:
        with open(save_path, 'rb') as f:
            test_name_list = pickle.load(f)

    return train_name_list, test_name_list

def get_splitlabels():
    """获得训练集和测试集label列表"""
    save_path = os.path.join(img_dir, 'train_labels.pkl')
    if not os.path.exists(save_path):
        labels = get_alllabels()
        train_test_spit = get_split()
        train_labels = [label for label, split in zip(labels, train_test_spit) if split]
        with open(save_path, 'wb') as f:
            pickle.dump(train_labels, f)
    else:
        with open(save_path, 'rb') as f:
            train_labels = pickle.load(f)

    save_path = os.path.join(img_dir, 'test_labels.txt')
    if not os.path.exists(save_path):
        labels = get_alllabels()
        train_test_spit = get_split()
        test_labels = [label for label, split in zip(labels, train_test_spit) if split]
        with open(save_path, 'wb+') as f:
            pickle.dump(test_labels, f)
    else:
        with open(save_path, 'rb') as f:
            test_labels = pickle.load(f)

    return train_labels, test_labels

raw_train_transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.RandomCrop(INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

raw_test_transform = transforms.Compose([
    transforms.Resize((600, 600), Image.BILINEAR),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class CUB():
    def __init__(self, root, is_train=True, data_len=None, train_transform=raw_train_transform, test_transform=raw_test_transform):
        self.root = root
        self.is_train = is_train
        train_file_list, test_file_list = get_splitnames()  # 标记为1的是训练集
        train_label, test_label = get_splitlabels()
        self.train_transform = train_transform
        self.test_transform = test_transform
        if self.is_train:
            # 三种分辨率都要
            self.train_img = [Image.open(os.path.join(self.root, 'images', train_file)).convert('RGB') for train_file in
                              train_file_list[:data_len]] + \
                             [Image.open(os.path.join(self.root, 'images_x2', train_file)).convert('RGB') for train_file in
                              train_file_list[:data_len]] + \
                             [Image.open(os.path.join(self.root, 'images_x4', train_file)).convert('RGB') for train_file in
                              train_file_list[:data_len]]  # cv2.imread返回的是numpy.ndarray类型

            self.train_label = train_label[:data_len] * 3
        if not self.is_train:
            self.test_img = [Image.open(os.path.join(self.root, 'images', test_file)).convert('RGB') for test_file in
                             test_file_list[:data_len]]
            self.test_label = test_label[:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            # img = Image.fromarray(img, mode='RGB')
            # img = self.train_transform(img)
            img = my_transforms1(img)
        else:
            img, target = self.test_img[index], self.test_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            # img = Image.fromarray(img, mode='RGB')
            img = my_transforms2(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


if __name__ == '__main__':
    dataset = CUB(root='./CUB_200_2011')
    print(len(dataset.train_img))
    print(len(dataset.train_label))
    for data in dataset:
        print(data[0].size(), data[1])
    dataset = CUB(root='./CUB_200_2011', is_train=False)
    print(len(dataset.test_img))
    print(len(dataset.test_label))
    for data in dataset:
        print(data[0].size(), data[1])
