import numpy as np
from os import listdir
import os.path as osp
import cv2
import timeit

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3


def cal_dir_stat(root):
    cls_dirs = [d for d in listdir(root) ]
    pixel_num = 0  # store all pixel number in the dataset
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)

    for img in cls_dirs:
        print(img)
        if not img.endswith('.JPG'):
            continue
        im = cv2.imread(osp.join(root, img))  # image in M*N*CHANNEL_NUM shape, channel in BGR order
        im = im / 255.0
        pixel_num += (im.size / CHANNEL_NUM)
        channel_sum += np.sum(im, axis=(0, 1))
        channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))

    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]

    return rgb_mean, rgb_std


# Train_root must contain the directory with images to compute mean
train_root = "/Users/vxgupta/data/gr_ds/ShelfImages/test"
start = timeit.default_timer()
mean, std = cal_dir_stat(train_root)
end = timeit.default_timer()
print("elapsed time: {}".format(end - start))
print("mean:{}\nstd:{}".format(mean, std))