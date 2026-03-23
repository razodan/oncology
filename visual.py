import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd

def save_training_image():
    df = pd.read_csv('training_data.csv')
    x = df['label'].value_counts()
    plt.title('Skin Cancer Training Data')
    plt.pie(x.values,
            labels=x.index,
            autopct='%1.1f%%')
    plt.savefig('training_data.png')

def save_testing_image():
    df = pd.read_csv('testing_data.csv')
    x = df['label'].value_counts()
    plt.title('Skin Cancer Testing Data')
    plt.pie(x.values,
            labels=x.index,
            autopct='%1.1f%%')
    plt.savefig('testing_data.png')

# save_training_image()
save_testing_image()