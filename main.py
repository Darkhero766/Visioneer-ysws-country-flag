import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


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
    
}


def predict_flag(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
  
    predicted_country = list(country_info.keys())[predicted_index]
    
    print(f"Flag detected: {predicted_country}")
    print("About the country:", country_info[predicted_country])


predict_flag("download.png")
