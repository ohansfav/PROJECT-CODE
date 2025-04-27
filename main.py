# Import necessary libraries
import cv2
import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = './train'
VALIDATION_DIR = './validation'

# Data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255  # Normalize pixel values to [0, 1]
)

# Load and preprocess training data
train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'  # Use 'binary' for two classes or 'categorical' for multiple classes
)

# Load and preprocess validation data
validation_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)