# imports
from config import DIM_REDUCTION_TECHNIQUE
from config import DATA_PATH
from config import REDUCTER_PATH
from config import VECTORS_PATH
from config import N_DIMENSIONS
from config import TEST_PERCENTAGE
from config import MODEL_PATH

import sys
sys.path.append('src')
from utils import get_img
from utils import get_dim_reduction_technique
from utils import save_pkl_file
from utils import load_pkl_file
from vectorizer import Vectorizer

import os
import logging
import json

import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# setup
logging.basicConfig(level=logging.INFO)

# flags
flags = tf.app.flags
flags.DEFINE_string('data_path', DATA_PATH,
                    'Path to JSON file with paths to data.')
flags.DEFINE_string('reducter_path', REDUCTER_PATH,
                    'Path to reducter file.')
flags.DEFINE_string('vectors_path', VECTORS_PATH,
                    'Path to pickle file with vectors.')
flags.DEFINE_string('dim_reduction_technique', DIM_REDUCTION_TECHNIQUE,
                    'Either "truncated_svd" or "pca".')
flags.DEFINE_string('model_path', MODEL_PATH,
                    'Path to model.')
flags.DEFINE_integer('n_dimensions', N_DIMENSIONS,
                     'Number of dimensions to reduce to.')
flags.DEFINE_float('test_percentage', TEST_PERCENTAGE,
                   'Percentage of data to test on.')
FLAGS = flags.FLAGS


# functions
def load_data(data_file):
    logging.info('loading JSON file at {}'.format(data_file))
    with open(data_file, 'r') as json_file:
        return json.load(json_file)


def save_vectors_file():
    data = load_data(FLAGS.data_path)

    vectorizer = Vectorizer()

    logging.info('getting vectors')
    img_vectors = []
    genders = []
    for img_path, gender_id in tqdm(data.items()):
        try:
            img_array = get_img(img_path)

            vector = vectorizer.get_vector(img_array)

            img_vectors.append(vector)
            genders.append(gender_id)
        except Exception as e:
            logging.warning('exception: {}'.format(e))

    vectorizer.close()

    dim_reduction_technique = get_dim_reduction_technique(
        FLAGS.dim_reduction_technique)

    reduced, model = dim_reduction_technique(
        img_vectors, FLAGS.n_dimensions)

    save_pkl_file(model, FLAGS.reducter_path)
    save_pkl_file((reduced, genders), FLAGS.vectors_path)


def main():
    if not os.path.exists(FLAGS.vectors_path):
        save_vectors_file()

    X, y = load_pkl_file(FLAGS.vectors_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FLAGS.test_percentage)

    logging.info('training')
    model = LogisticRegression()
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    print('Results on test data:\n')
    print(classification_report(y_test, predicted))

    save_pkl_file(model, FLAGS.model_path)


if __name__ == '__main__':
    main()
