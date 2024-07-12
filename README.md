##  Emotion Recognition

This project focuses on developing a speech-based emotion detection model using Convolutional Neural Networks (CNN). The project involves data processing, feature extraction, training a neural network, and evaluating the model. 
##  Scripts
# Data Processing
data_processing.py reads the audio file paths from audio_files.txt and prepares the data for feature extraction.

# Feature Extraction
feature_extraction.py extracts MFCC, chroma, mel, and contrast features from the audio files and saves them as numpy arrays in ../data/processed/.

# Training
train.py trains a CNN model on the extracted features and labels. It saves the trained model and training history in the ../models/ directory.

# Evaluation
evaluate.py evaluates the trained model on the test set and generates a classification report, which is saved in ../results/evaluation_report.txt.


## Running the Pipeline
To run the entire pipeline, navigate to the pipeline directory and execute:

python -m run_pipeline