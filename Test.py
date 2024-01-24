import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras import layers, models
#from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt
import random

model_path = r"D:\proj\model_separated"
loaded_model = load_model(model_path)