import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers

"""
    0   actinic keratosis
    1   basal cell carcinoma
    2   dermatofibroma
    3   melanoma
    4   nevus
    5   pigmented benign keratosis
    6   seborrheic keratosis
    7   squamous cell carcinoma
    8   vascular lesion

    Consider using SpareCategoricalCrossentropy
"""

def load_training():
    paths = pd.read_csv('training_data.csv')['filepath']
    labels = pd.read_csv('training_data.csv')['label']
    train_X = tf.data.Dataset.from_tensor_slices((paths,labels))
    return train_X


train_X = load_training()
