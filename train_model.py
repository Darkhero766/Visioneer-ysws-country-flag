import random
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
  
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
  
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')
])

countries = ['India', 'Japan', 'Brazil']
)
num_samples = 100
X_train = []
y_train = []

# India flag
for _ in range(num_samples):
    img = np.ones((128, 128, 3))
    img[:42, :] = [1.0, 0.5, 0]  # Orange
    img[43:85, :] = [1.0, 1.0, 1.0]  # White
    img[86:, :] = [0, 0.8, 0]  # Green
    img += np.random.normal(0, 0.1, img.shape)  
    img = np.clip(img, 0, 1)
  
    X_train.append(img)
    y_train.append(0)

# Japan flag
for _ in range(num_samples):
    img = np.ones((128, 128, 3))  # White background
    X_train.append(img)
    y_train.append(1)  

# Brazil flag 
for _ in range(num_samples):
    img = np.zeros((128, 128, 3))
    img[:, :] = [0, 0.8, 0]  # Green
    X_train.append(img)
    y_train.append(2)

X_train = np.array(X_train)
y_train = np.array(y_train)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)


model.save('flag_classifier.h5')
#print("Model saved as flag_classifier.h5")
