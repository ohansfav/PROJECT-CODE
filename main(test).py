# This is for my final year project // for xray using ai to scan
import cv2
import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


# Load an X-ray image
img = cv2.imread('xray_image.jpg')

# Resize the image
img = cv2.resize(img, (224, 224))

# Normalize pixel values to [0, 1]
img = img / 255.0

# Alternatively, use ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use flow_from_directory to generate augmented data
train_generator = datagen.flow_from_directory(
    'path/to/train/directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)


