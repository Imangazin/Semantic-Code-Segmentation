import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load Uncentered Data
unc_data_path = "data/unc_data.json"
with open(unc_data_path, "r", encoding="utf-8") as f:
    unc_data = json.load(f)

# Convert data to feature matrix and labels
X_unc = [entry["window"] for entry in unc_data]  # Extract character sequences
y_unc = np.array([entry["label"] for entry in unc_data])  # Extract labels

# Define character vocabulary
unique_chars = set("".join(X_unc))
char_to_index = {char: idx + 1 for idx, char in enumerate(unique_chars)}  # Start indexing from 1
vocab_size = len(unique_chars) + 1  # Extra slot for padding

# Convert characters to integer sequences
MAX_LENGTH = 100  # Fixed input length
X_encoded = np.zeros((len(X_unc), MAX_LENGTH), dtype=int)
y_encoded = np.zeros((len(y_unc), MAX_LENGTH), dtype=int)

for i, text in enumerate(X_unc):
    encoded_text = [char_to_index.get(char, 0) for char in text[:MAX_LENGTH]]  # Truncate/PAD
    X_encoded[i, :len(encoded_text)] = encoded_text  # Assign values
    y_encoded[i, :len(encoded_text)] = y_unc[i]  # Assign labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_unc)

# Define Neural Network Architecture
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=20),
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(150, activation='relu')),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(75, activation='relu')),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))  # Binary output per character
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("uncentered_bilstm_model.keras")
print("Model training complete. Saved as 'uncentered_bilstm_model.h5'.")