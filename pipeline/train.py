import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Activation, Dropout, MaxPooling1D, Flatten, Dense
from keras.optimizers import RMSprop
import joblib

# Load features and labels
features = np.load("../data/processed/features.npy")
labels = np.load("../data/processed/labels.npy")


print(np.unique(labels))
# Convert labels to integers
y = labels


# Print unique labels to verify
print("Unique labels in y_train:", np.unique(y))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, y, test_size=0.2, random_state=42, stratify=y
)

# Reshape data to fit the model: (num_samples, timesteps, num_features)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Initialize a sequential model
model = Sequential()

# First convolutional layer
model.add(
    Conv1D(64, 5, padding="same", input_shape=(X_train.shape[1], X_train.shape[2]))
)
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=4))

# Second convolutional layer
model.add(Conv1D(128, 5, padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=4))

# Third convolutional layer
model.add(Conv1D(256, 5, padding="same"))
model.add(Activation("relu"))
model.add(Dropout(0.1))

# Flatten the output
model.add(Flatten())

# Dense layer
model.add(Dense(8))
model.add(Activation("softmax"))

# Define the RMSprop optimizer with a lower learning rate
opt = RMSprop(learning_rate=0.001)

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
)

# Print a summary of the model architecture
model.summary()

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# Save the model
model.save("../models/emotion_recognition_model.h5")
# Save the training history
joblib.dump(history.history, "../models/training_history.pkl")
# Save the test set for later use in evaluation
np.save("../data/processed/X_test.npy", X_test)
np.save("../data/processed/y_test.npy", y_test)
