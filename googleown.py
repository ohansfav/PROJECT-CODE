import os
import cv2
import numpy as np
from tensorflow import keras
from google.cloud import vision_v1 as vision
from PIL import Image

# Function to process and analyze X-ray images
def analyze_xray(image_path):
    """
    Analyze the X-ray image and return the diagnosis.
    """
    # Step 1: Upload the image to Google Cloud Vision API
    image = vision.Image(content=open(image_path, 'rb').read())
    client = vision.ImageAnnotatorClient()
    response = client.label_detection(image=image)
    labels = response.label_annotations

    # Step 2: Extract relevant features from the image
    features = []
    for label in labels:
        features.append(label.description)
    
    # Step 3: Preprocess the image for AI model input
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Assuming model expects 224x224 input
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Step 4: Load the AI model
    model = keras.models.load_model('xray_diagnosis_model.h5')

    # Step 5: Run the prediction
    prediction = model.predict(img)
    diagnosis = np.argmax(prediction, axis=1)[0]

    return diagnosis

# Example usage
if __name__ == "__main__":
    image_path = "path_to_xray_image.jpg"
    diagnosis = analyze_xray(image_path)
    print(f"Diagnosis: {diagnosis}")    