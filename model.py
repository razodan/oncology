import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from typing import Dict

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

def encode_labels():
    labels = pd.read_csv('training_data.csv')['label']
    labels = {i: value for i, value in enumerate(set(labels))}
    return labels

def load_training(labels: Dict[int,str]=None): # This isn't scalable or robust---consider another approach
    paths = pd.read_csv('training_data.csv')['filepath']
    train_X = tf.data.Dataset.from_tensor_slices((paths,labels))
    return train_X

encoded_labels = encode_labels()
train_X = load_training(encoded_labels)

loss = tf.keras.losses.SparseCategoricalCrossentropy() # SCCE = -1 * ( \sum_{i=1}^{n} \log(\hat{y}_{i, y_i}) )

train = loss(y_true=None,
             y_pred=None)