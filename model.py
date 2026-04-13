import numpy as np
import pandas as pd

from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History
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

X_train: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    "data/Train",
    labels="inferred",
    label_mode="int",
    image_size=(600,450),
    batch_size=32
)

X_test: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
    "data/Test",
    labels="inferred",
    label_mode="int",
    image_size=(600,450),
    batch_size=32,
    shuffle=False
)

X_train_shape = None
for images,_ in X_train.take(1):
    X_train_shape = images.shape

model = tf.keras.Sequential([
    layers.Input(shape=X_train_shape[1:]),
    layers.Rescaling(1./255),

    layers.Conv2D(32,3,activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Conv2D(64,3,activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(9,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # SCCE = -1 * ( \sum_{i=1}^{n} \log(\hat{y}_{i, y_i}) )
    metrics=['accuracy']
)

history: History = model.fit(
    X_train,
    epochs=2,
    batch_size=32
)
