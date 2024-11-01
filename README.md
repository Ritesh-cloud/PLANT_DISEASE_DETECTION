# PLANT_DISEASE_DETECTION
This repository contains a Convolutional Neural Network (CNN) model built with TensorFlow and Keras to classify plant diseases from images. The model processes training and validation image datasets, and includes steps for building, training, evaluating, and saving the model.

Key Features
Data Processing: Loads and preprocesses training and validation image datasets.
Model Architecture: Utilizes multiple convolutional, pooling, and dropout layers to effectively learn features of plant diseases.
Training & Evaluation: Compiles and trains the model using categorical cross-entropy loss and accuracy metrics, with visualization of training accuracy.
Performance Metrics: Generates a classification report and visualizes a confusion matrix for further insights into model performance.
Visualizations
Accuracy Visualization: Plots training and validation accuracy over epochs.
Confusion Matrix: Provides a detailed heatmap for error analysis of predicted vs. actual classes.

CODE FILE 2 
This script demonstrates loading and testing a pre-trained CNN model to classify plant diseases from images. The model, built with TensorFlow and saved as trained_model.h5, can predict disease types in plant leaves based on the test set images.

Key Features
Model Loading: Loads the pre-trained model for efficient testing on new data.
Single Image Prediction: Processes a single test image, displaying it along with the model's prediction.
Prediction Visualization: Maps the prediction to its disease name and overlays it on the displayed image for a clear visual result
