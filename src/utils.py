# imports
from config import IMG_SIZE
from dim_reduction import truncated_svd
from dim_reduction import pca

import logging
from six.moves import cPickle

import cv2


# functions
def get_dim_reduction_technique(string):
    if string == 'truncated_svd':
        return truncated_svd
    elif string == 'pca':
        return pca


def get_img(p, img_size=IMG_SIZE):
    img = cv2.imread(p)

    return cv2.resize(img, img_size)


def load_pkl_file(file_path):
    logging.info('loading file at {}'.format(file_path))
    with open(file_path, 'rb') as pkl_file:
        return cPickle.load(pkl_file)


def save_pkl_file(data, file_path):
    logging.info('saving file at {}'.format(file_path))
    with open(file_path, 'wb') as pkl_file:
        cPickle.dump(data, pkl_file)
