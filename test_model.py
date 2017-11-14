# imports
from config import REDUCTER_PATH
from config import MODEL_PATH

import sys
sys.path.append('src')
from utils import get_img
from utils import load_pkl_file
from vectorizer import Vectorizer

import logging

import tensorflow as tf

# setup
logging.basicConfig(level=logging.INFO)

# flags
flags = tf.app.flags
flags.DEFINE_string('img_path', 'image.jpg',
                    'Path to image to test on.')
flags.DEFINE_string('reducter_path', REDUCTER_PATH,
                    'Path to reducter file.')
flags.DEFINE_string('model_path', MODEL_PATH,
                    'Path to model.')
FLAGS = flags.FLAGS


# functions
def main():
    img = get_img(FLAGS.img_path)

    vectorizer = Vectorizer()
    vector = vectorizer.get_vector(img)
    vectorizer.close()

    reducter = load_pkl_file(FLAGS.reducter_path)
    reduced = reducter.transform([vector])

    model = load_pkl_file(FLAGS.model_path)

    output = model.predict(reduced)[0]

    print('result: {}'.format(output))


if __name__ == '__main__':
    main()
