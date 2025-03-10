import joblib
import numpy as np
from tensorflow import keras

# Load trained models
log_reg_model = joblib.load("data/logistic_regression_model.pkl")
bilstm_model = keras.models.load_model("data/centered_bilstm_model.keras")
uncentered_bilstm_model = keras.models.load_model("data/uncentered_bilstm_model.keras")

# Define a function to preprocess source code into numerical format
def preprocess_code(code_snippet, char_to_index, max_length=101, model_type="bilstm"):
    """
    Converts source code into numerical representation.
    - For BiLSTM: Uses character embeddings (101 characters).
    - For Logistic Regression: Uses Bag of Characters (50 features).
    """
    if model_type == "logistic":
        # Convert source code to Bag of Characters (BOC) format (50 features)
        boc_vector = np.zeros(50, dtype=int)  # Ensure it matches logistic regression's expected input
        for i, char in enumerate(code_snippet[:50]):  # Limit to 50 characters
            boc_vector[i] = char_to_index.get(char, 0)
        return boc_vector.reshape(1, -1)  # Reshape for model input
    
    else:
        # BiLSTM model uses 101-character sequences
        encoded_text = [char_to_index.get(char, 0) for char in code_snippet[:max_length]]
        padded_sequence = np.zeros((max_length,), dtype=int)
        padded_sequence[:len(encoded_text)] = encoded_text
        return padded_sequence.reshape(1, max_length)  # Reshape for model input

# Define a function to segment source code using the trained models
def segment_code(model, code_snippet, char_to_index, threshold=0.5, model_type="bilstm"):
    """
    Predict segmentation points for source code using different models.
    """
    processed_code = preprocess_code(code_snippet, char_to_index, model_type=model_type)

    if model_type == "logistic":
        # Logistic Regression outputs a single probability, not per-character probabilities
        prediction = model.predict(processed_code)[0]  # Get a single prediction value
        #print(f"Logistic Regression Prediction: {prediction}")  # Debug output

        segmentation_points = [0] if prediction > threshold else []  # If above threshold, segment at position 0
    
    else:
        # BiLSTM models output probabilities per character
        predictions = model.predict(processed_code)[0]  
        print(f"{model_type} Predictions: {predictions}")  # Debug output
        segmentation_points = [i for i, prob in enumerate(predictions) if prob > threshold]

    return segmentation_points

# Define character-to-index mapping (should be same as training data preprocessing)
char_to_index = {char: idx + 1 for idx, char in enumerate("abcdefghijklmnopqrstuvwxyz(){}[];=+-/*<>:")}

# Example source code snippet

source_code = "def add(a, b): return a + b"


# Run segmentation using all three models
log_reg_seg_points = segment_code(log_reg_model, source_code, char_to_index, threshold=0.1, model_type='logistic')
bilstm_seg_points = segment_code(bilstm_model, source_code, char_to_index, threshold=0.19, model_type='bilstm')
uncentered_bilstm_seg_points = segment_code(uncentered_bilstm_model, source_code, char_to_index, threshold=0.1, model_type='bilstm')

# Print segmentation results
print("Segmentation Points (Logistic Regression):", log_reg_seg_points)
print("Segmentation Points (Centered BiLSTM):", bilstm_seg_points)
print("Segmentation Points (Uncentered BiLSTM):", uncentered_bilstm_seg_points)

# Visualize segmentation by adding '|' at segment points
def visualize_segmentation(source_code, seg_points):
    return "".join(["|" + ch if i in seg_points else ch for i, ch in enumerate(source_code)])

print("Segmented Code (Logistic Regression):", visualize_segmentation(source_code, log_reg_seg_points))
print("Segmented Code (Centered BiLSTM):", visualize_segmentation(source_code, bilstm_seg_points))
print("Segmented Code (Uncentered BiLSTM):", visualize_segmentation(source_code, uncentered_bilstm_seg_points))
