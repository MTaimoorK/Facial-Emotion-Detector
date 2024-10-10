import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt


# load the model
model = load_model("best_model.keras")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
while True:
    ret, test_img = capture.read()
    if not ret:
        continue
    
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    
    faces_deteced = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x, y, w, h) in faces_deteced:
        cv2.rectangle(test_img, (x,y),(x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]
        roi_gray = cv2.resize(roi_gray, (224,224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255
        
        predictions = model.predict(img_pixels)
        
        max_index = np.argmax(predictions[0])
        
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotions = emotions[max_index]
        
        cv2.putText(test_img, predicted_emotions, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    resized_image = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial Emotion Detector', resized_image)
    
    if cv2.waitKey(10) == ord('z'): # wait until 'z' is pressed
        break

capture.release()
cv2.destroyAllWindows