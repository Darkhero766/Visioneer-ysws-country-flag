import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime
import json

import os

if not os.path.exists('flag_classifier.h5'):
    print("Error: Model file 'flag_classifier.h5' not found!")
    print("Please ensure you have trained the model and saved it as 'flag_classifier.h5'")
    exit(1)


model = load_model('flag_classifier.h5')


country_info = {
    "India": "India is the world's largest democracy and the seventh-largest country by land area.",
    "Japan": "Japan is an island country in East Asia, known for its technology and traditions.",
    
    "Brazil": "Brazil is the largest country in South America, famous for its Amazon rainforest and football.",
    
    "France": "France is known for its rich culture, cuisine, and being home to the Eiffel Tower.",
    
    "Germany": "Germany is the largest economy in Europe, known for its engineering and automotive industry.",
    "USA": "The United States is the world's largest economy and a global superpower.",
}




def rotate_image(image, angle):
    height, width = image.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (width, height))

def save_prediction_history(prediction_data):
    try:
        with open('prediction_history.json', 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    
    history.append(prediction_data)
    
    with open('prediction_history.json', 'w') as f:
        json.dump(history, f, indent=2)

def predict_flag(image_path):
    original_img = cv2.imread(image_path)
    best_confidence = 0
    best_prediction = None
    best_angle = 0
    
    
    for angle in [0, 90, 180, 270]:
        rotated_img = rotate_image(original_img.copy(), angle)
        processed_img = cv2.resize(rotated_img, (128, 128))
        processed_img = processed_img / 255.0
        processed_img = np.expand_dims(processed_img, axis=0)

        prediction = model.predict(processed_img, verbose=0)
        confidence = np.max(prediction)
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_prediction = prediction
            best_angle = angle

    predicted_index = np.argmax(best_prediction)
    predicted_country = list(country_info.keys())[predicted_index]
    confidence_percentage = best_confidence * 100
    
    print(f"Flag detected: {predicted_country}")
    
    print(f"Confidence: {confidence_percentage:.2f}%")
    
    print(f"Flag rotation: {best_angle}°")
    print("About the country:", country_info[predicted_country])
    
  
    prediction_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "country": predicted_country,
        "confidence": float(confidence_percentage),
        "rotation": best_angle,
        "image_path": image_path
    }
    save_prediction_history(prediction_data)

def show_prediction_history():
    try:
        with open('prediction_history.json', 'r') as f:
            history = json.load(f)
            print("\nPrediction History:")
            for entry in history[-5:]:  
                print(f"\nTime: {entry['timestamp']}")
                print(f"Country: {entry['country']}")
                print(f"Confidence: {entry['confidence']:.2f}%")
                
                print(f"Rotation: {entry['rotation']}°")
    except FileNotFoundError:
        print("\nNo prediction history found.")

predict_flag("download.png")
show_prediction_history()
