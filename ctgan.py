import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D, Activation, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D, Input
from keras.models import Model
from keras.optimizers import RMSprop
import random

def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

from google.colab import drive
drive.mount('/content/drive')

finaltrain = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_train.csv')
finaltest = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/final15kAll_test.csv')

from ctgan import CTGAN
from ctgan import load_demo

real_data = load_demo()

# Names of the columns that are discrete
discrete_columns = ['y']

ctgan = CTGAN(epochs=10)
ctgan.fit(finaltrain, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(760)
synthetic_data.to_csv("/content/drive/MyDrive/Colab Notebooks/Urep/gan_data/CTgan760.csv",index=False)
