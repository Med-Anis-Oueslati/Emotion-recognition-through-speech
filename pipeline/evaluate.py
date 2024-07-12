import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import joblib

# Load the test set
X_test = np.load("../data/processed/X_test.npy")
y_test = np.load("../data/processed/y_test.npy")
# Load the model
model = tf.keras.models.load_model("../models/emotion_recognition_model.h5")
# Load the training history
history = joblib.load("../models/training_history.pkl")

# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {score[1]}")

# Generate classification report and confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

report = classification_report(y_test, y_pred_classes)
print(report)

# Save the report to a file
with open("../results/evaluation_report.txt", "w") as f:
    f.write(report)
