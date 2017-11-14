# imports
import logging

import numpy as np
from keras.models import Model
from keras import backend as K
from keras_vggface.vggface import VGGFace


# classes
class Vectorizer:
    def __init__(self, layer='fc6'):
        logging.info('loading VGG face')
        self.layer = layer

        vgg_face = VGGFace()

        self.model = Model(inputs=vgg_face.layers[0].input,
                           outputs=vgg_face.get_layer(self.layer).output)

        session = K.get_session()
        K.set_session(session)

    def get_vector(self, img_input):
        img_input = np.array([img_input])
        out = self.model.predict(img_input)

        return out[0]

    def close(self):
        K.clear_session()
