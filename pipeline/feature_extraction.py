import os
import librosa
import numpy as np


# Function to extract mfcc, chroma, mel, and contrast features from audio files
def extract_features(file_path, sample_rate=22050):
    try:
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features = np.hstack(
            (
                np.mean(mfccs, axis=1),
                np.mean(chroma, axis=1),
                np.mean(mel, axis=1),
                np.mean(contrast, axis=1),
            )
        )
        return features
    except Exception:
        print(f"Error encountered while parsing file: {file_path}")
        return None


# Load audio files from the text file
audio_files = []
with open("../data/processed/audio_files.txt", "r") as file:
    audio_files = [line.strip() for line in file.readlines()]

# Extract features from all audio files
features = []
labels = []
label_map_tess = {
    "OAF_angry": 4,
    "OAF_disgust": 6,
    "OAF_Fear": 5,
    "OAF_happy": 2,
    "OAF_Pleasant_surprise": 7,
    "OAF_Sad": 3,
    "OAF_neutral": 0,
    "YAF_angry": 4,
    "YAF_disgust": 6,
    "YAF_fear": 5,
    "YAF_happy": 2,
    "YAF_pleasant_surprised": 7,
    "YAF_sad": 3,
    "YAF_neutral": 0,
}
label_map_ravdess = {
    "01": 0,
    "02": 1,
    "03": 2,
    "04": 3,
    "05": 4,
    "06": 5,
    "07": 6,
    "08": 7,
}

for idx, file in enumerate(audio_files, start=1):
    feature = extract_features(file)
    if feature is not None:
        features.append(feature)
        if "audio_speech_actors_01-24" in file:
            # Extract label from RAVDESS file name
            label = file.split(os.sep)[-1].split("-")[2]
            labels.append(label_map_ravdess[label])
        else:
            # Extract label from TESS file path
            emotion = file.split(os.sep)[-2]
            if emotion in label_map_tess:
                labels.append(label_map_tess[emotion])
            else:
                print(f"Skipping {file} with unrecognized emotion: {emotion}")
                features.pop()  # Remove the feature if label is not recognized
        print(f"Processing file {idx} of {len(audio_files)}")

print("Feature extraction complete.")

# Save the features and labels
features = np.array(features)
labels = np.array(labels)
np.save("../data/processed/features.npy", features)
np.save("../data/processed/labels.npy", labels)
