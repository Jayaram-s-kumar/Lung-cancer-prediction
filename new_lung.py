import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('model.h5')  # Replace with the actual path
import cv2

img_path = 'lungscc1.jpeg'  # Replace with the actual path

img = cv2.imread(img_path)
img = cv2.resize(img, (256, 256))  # Resize to match the model's input size
#img = img / 255.0  # Normalize pixel values to the range [0, 1]

# Expand dimensions to add a batch dimension (necessary for model input)
img = np.expand_dims(img, axis=0)
predictions = model.predict(img)
#print(predictions)  # Output: Array of probabilities for each class

predicted_class = np.argmax(predictions)  # Get the index of the highest probability
#print(predicted_class)  # Output: Numerical index of the predicted class

# Map the index to the actual class name (if you have the class names available)
#classes = ['lung_aca', 'lung_n', 'lung_scc']  # Replace with your actual class names
classes = ['Adenocarcinomas tissue','Benign tissue','Squamous cell tissue']
predicted_class_name = classes[predicted_class]
print('')
print(predicted_class_name)  # Output: The predicted class name
