import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.tensorflow.keras.callbacks import ReduceLROnPlateau

# Function to load and preprocess the image
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("Image not found.")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return img

# Create data generator for data augmentation
def create_datagen():
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen

# Function to create CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model

# Main function to run the analysis
def analyze_xray(image_path):
    # Load image
    img = load_image(image_path)
    img = np.expand_dims(img, axis=0)
    
    # Create and compile model
    model = create_model()
    
    # Data generator
    datagen = create_datagen()
    train_dir = 'data/train'
    batch_size = 32
    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=10,
        verbose=1,
        callbacks=[ReduceLROnPlateaupatience=5])
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    
    # Display results
    plt.imshow(cv2.imread(image_path))
    plt.title(f'Prediction: {["Normal", "Pneumonia"][predicted_class[0]]}')
    plt.show()

# Run the analysis
analyze_xray('xray_image.jpg')