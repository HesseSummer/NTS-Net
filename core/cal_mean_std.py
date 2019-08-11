# coding: utf-8

import numpy as np
import cv2
import random
import os
from config import img_dir, INPUT_SIZE
import pickle
from dataset import get_splitnames

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""


# train_txt_path = os.path.join("..", "..", "Data/train.txt")

CNum = 2000     # 挑选多少图片进行计算
img_h, img_w = INPUT_SIZE



def get_res(name_list):
    means, stdevs = [], []
    imgs = np.zeros((img_w, img_h, 3, 1))
    # # 使用全部图片
    # for name in name_list:
    #     img = cv2.imread(os.path.join(img_dir, 'images', name))
    # 随机挑选部分图片

    random.shuffle(name_list)
    for i in range(CNum):
        img = cv2.imread(os.path.join(img_dir, 'images', name_list[i]))
        # cv2.imshow('', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]

        imgs = np.concatenate((imgs, img), axis=3)
        print(i)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:,:,i,:].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse() # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))

    return means, stdevs

if __name__ == '__main__':
    train_name_list, test_name_list = get_splitnames()
    means, stdevs = get_res(test_name_list)
    with open('ForNormalization.pkl', 'wb+') as f:
        pickle.dump((means, stdevs), f)
