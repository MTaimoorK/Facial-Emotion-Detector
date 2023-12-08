# Facial Emotion Detector

**Author:** Muhammad Taimoor Khan

## Overview
This project implements a real-time facial emotion detector using a pre-trained deep learning model. The model is trained on a dataset containing facial expressions and is capable of predicting emotions such as anger, disgust, fear, happiness, sadness, surprise, and neutral.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Keras
- Matplotlib
- Jupyter Notebook (for running emotion_detector.ipynb)
- Pre-trained model file: best_model.h5 (not included in this repository due to size)

## Usage

### Real-Time Emotion Detection
To test the real-time emotion detection, run `videotester.py`. Ensure that your webcam is connected, and the necessary dependencies are installed.

```bash
python videotester.py
Press 'z' to exit the real-time detection.
```

## Training the Model
The model has been trained and saved as ```best_model.h5.``` If you wish to train the model yourself or modify the training parameters, refer to emotion_detector.ipynb for the Jupyter Notebook containing the training code.

## Project Structure
- ```videotester.py:``` Real-time facial emotion detection script.
- ```emotion_detector.ipynb:``` Jupyter Notebook containing the model training code.
- ```best_model.h5:``` Pre-trained model file.
- ```train/:``` Directory containing the training dataset.
- ```image.jpg:``` Sample image for manual testing.

## Results
The ```best_model.h5``` file contains the trained model, achieving high accuracy and performance. The training history and visualizations are available in the Jupyter Notebook.

## Contribution
Feel free to contribute, raise issues, or use the code in your projects. If you find this project helpful, please consider giving it a star!

