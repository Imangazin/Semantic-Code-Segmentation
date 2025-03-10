import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Load Centered Data
cen_data_path = "data/cen_data.json"
with open(cen_data_path, "r", encoding="utf-8") as f:
    cen_data = json.load(f)

# Convert data to feature matrix and labels
MAX_LENGTH = 101  # Centered data uses 101 characters
y_cen = [entry["label"] if isinstance(entry["label"], list) else [entry["label"]] * MAX_LENGTH for entry in cen_data]
X_cen = [entry["window"] for entry in cen_data]  # Extract character sequences
print("Sample y_cen[0]:", y_cen[0])  # Should be a list, not a single integer
print("Type of y_cen[0]:", type(y_cen[0]))
print("Fixed Sample y_cen[0]:", y_cen[0])
print("Type of y_cen[0]:", type(y_cen[0]))
print("Length of y_cen[0]:", len(y_cen[0]))  # Should be 101

# Define character vocabulary
unique_chars = set("".join(X_cen))
char_to_index = {char: idx + 1 for idx, char in enumerate(unique_chars)}  # Start indexing from 1
vocab_size = len(unique_chars) + 1  # Extra slot for padding

# Convert characters to integer sequences
X_encoded = np.zeros((len(X_cen), MAX_LENGTH), dtype=int)
y_encoded = np.zeros((len(y_cen), MAX_LENGTH, 1), dtype=int)  # Ensure correct shape
print("y_cen shape before encoding:", np.array(y_cen).shape)
print("y_encoded shape after encoding:", y_encoded.shape)

for i, text in enumerate(X_cen):
    encoded_text = [char_to_index.get(char, 0) for char in text[:MAX_LENGTH]]  # Truncate/PAD
    X_encoded[i, :len(encoded_text)] = encoded_text  # Assign values
    y_encoded[i, :, 0] = np.array(y_cen[i][:MAX_LENGTH])  # Use per-character labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_cen)

# Define Neural Network Architecture
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=20, input_length=MAX_LENGTH),
    keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True)),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(150, activation='relu')),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(75, activation='relu')),
    keras.layers.Dropout(0.2),
    keras.layers.TimeDistributed(keras.layers.Dense(1, activation='sigmoid'))
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("data/centered_bilstm_model.keras")
print("Model training complete. Saved as 'centered_bilstm_model.keras'.")