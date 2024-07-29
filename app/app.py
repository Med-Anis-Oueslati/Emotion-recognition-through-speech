import gradio as gr
import numpy as np
import librosa
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("../models/model_tess.h5")

# Emotion labels
emotion_labels = {
    0: "Neutral",
    1: "Calm",
    2: "Happy",
    3: "Sad",
    4: "Angry",
    5: "Fearful",
    6: "Disgust",
    7: "Surprised",
}


# Function to extract features from an audio file
def extract_features(audio, sample_rate=22050):
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        features = np.hstack(
            (
                np.mean(mfccs, axis=1),
                np.mean(chroma, axis=1),
                np.mean(mel, axis=1),
                np.mean(contrast, axis=1),
            )
        )
        return features
    except Exception as e:
        print(f"Error encountered while parsing audio: {str(e)}")
        return None


# Function to make predictions on an uploaded audio file and return a bar chart of probabilities
def predict_emotion(audio_file):
    # Load the audio file using librosa
    audio, sr = librosa.load(audio_file, sr=22050)

    # Extract features from the audio file
    features = extract_features(audio, sample_rate=sr)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Reshape for model input
        features = np.expand_dims(features, axis=-1)  # Add channel dimension
        prediction = model.predict(features)[0]  # Get prediction probabilities
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        probabilities = {
            emotion_labels[i]: float(prob) for i, prob in enumerate(prediction)
        }

        # Create a bar chart of the probabilities
        fig, ax = plt.subplots()
        ax.bar(probabilities.keys(), probabilities.values())
        ax.set_xlabel("Emotions")
        ax.set_ylabel("Probability")
        ax.set_title("Emotion Probabilities")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to a file-like object and return it
        return predicted_emotion, fig
    else:
        return "Error: Could not process the audio file.", None


# Define the Gradio interface
interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=["text", "plot"],
    title="Emotion Detection from Speech",
    description="Upload an audio file and the model will predict the emotion expressed in the speech along with the probabilities for each emotion.",
)

if __name__ == "__main__":
    interface.launch()
