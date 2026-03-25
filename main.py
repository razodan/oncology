import numpy as np
import pandas as pd

from glob import glob


def prep_training_data():
    training_data = glob('data/Train/*/*.jpg')
    training_images = [path.replace('\\','/') for path in training_data]
    df_training = pd.DataFrame({'filepath': training_images})
    df_training['label'] = df_training['filepath'].str.split('/',expand=True)[2]
    df_training.to_csv('training_data.csv')

def prep_testing_data():
    testing_data = glob('data/Test/*/*.jpg')
    testing_images = [path.replace('\\','/') for path in testing_data]
    df_testing = pd.DataFrame({'filepath': testing_images})
    df_testing['label'] = df_testing['filepath'].str.split('/',expand=True)[2]
    df_testing.to_csv('testing_data.csv')

# prep_training_data()
# prep_testing_data()

