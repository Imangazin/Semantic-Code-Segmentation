import json
import re

# Load extracted code snippets
snippets_file_path = "data/1755_snippets.json"

with open(snippets_file_path, "r", encoding="utf-8") as f:
    code_snippets = json.load(f)
    print(f"Loaded {len(code_snippets)} code snippets from {snippets_file_path}")

# # Step 0: Preprocess & Clean the Code Snippets
# def clean_code_snippets(snippets):
#     """Cleans and normalizes code snippets by removing duplicates and unnecessary formatting."""
#     cleaned_snippets = set()  # Use a set to remove duplicates
#     for snippet in snippets:
#         snippet = snippet.strip()  # Remove leading/trailing whitespace
#         snippet = re.sub(r'\s+', ' ', snippet)  # Normalize spaces
#         snippet = re.sub(r'#.*', '', snippet)  # Remove inline comments (Python-style)
#         cleaned_snippets.add(snippet)
#     return list(cleaned_snippets)

# # Run preprocessing
# code_snippets = clean_code_snippets(code_snippets)
# print(f"Cleaned {len(code_snippets)} code snippets.")

# Step 1: Filter out snippets with fewer than 4 lines
def filter_short_snippets(snippets, min_lines=4):
    return [snippet for snippet in snippets if snippet.count("\n") >= (min_lines - 1)]

filtered_snippets = filter_short_snippets(code_snippets)

# Step 2: Concatenate snippets with newline characters
def concatenate_snippets(snippets):
    return "\n\n".join(snippets)  # Double newline for better segmentation

concatenated_code = concatenate_snippets(filtered_snippets)

# Step 3: Identify segmentation points ("dividing newlines")
def mark_dividing_newlines(code):
    lines = code.split("\n")
    segment_labels = [1 if i == 0 or re.match(r'\s*$', lines[i-1]) else 0 for i in range(len(lines))]
    return [{"text": lines[i], "label": segment_labels[i]} for i in range(len(lines))]

segmented_data = mark_dividing_newlines(concatenated_code)

# Step 4: Implement Segmentation Methods
WINDOW_SIZE = 50  # Adjust based on model input size

# Method 1: Bag of Characters (BoC)
def bag_of_characters(lines, window_size=WINDOW_SIZE):
    boc_data = []
    for entry in lines:
        char_vector = [1 if ord(char) < 128 else 0 for char in entry["text"][:window_size]]  # Binary feature vector
        char_vector += [0] * (window_size - len(char_vector))  # Pad to fixed length
        boc_data.append({"vector": char_vector, "label": entry["label"]})
    return boc_data

# Method 2: Uncentered Window
def uncentered_window(lines, window_size=WINDOW_SIZE):
    unc_data = []
    for i in range(len(lines)):
        window_start = max(0, i - window_size)
        window_text = ''.join([lines[j]["text"] for j in range(window_start, i)])
        unc_data.append({"window": window_text, "label": lines[i]["label"]})
    return unc_data

# Method 3: Centered Window
def centered_window(lines, window_size=WINDOW_SIZE):
    cen_data = []
    for i in range(len(lines)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(lines), i + window_size // 2)
        window_text = ''.join([lines[j]["text"] for j in range(window_start, window_end)])
        cen_data.append({"window": window_text, "label": lines[i]["label"]})
    return cen_data

# Apply the segmentation methods
boc_data = bag_of_characters(segmented_data)
unc_data = uncentered_window(segmented_data)
cen_data = centered_window(segmented_data)

# Save the segmented datasets
with open("data/boc_data.json", "w", encoding="utf-8") as f:
    json.dump(boc_data, f, indent=4)

with open("data/unc_data.json", "w", encoding="utf-8") as f:
    json.dump(unc_data, f, indent=4)

with open("data/cen_data.json", "w", encoding="utf-8") as f:
    json.dump(cen_data, f, indent=4)

print("Segmentation methods applied and saved.")

# Reloading and inspecting BOC data to identify issues
boc_data_path = "data/boc_data.json"

try:
    with open(boc_data_path, "r", encoding="utf-8") as f:
        boc_data = json.load(f)

    # Check the structure of the dataset
    boc_sample = boc_data[:5] if isinstance(boc_data, list) else boc_data
    boc_summary = {
        "Total Records": len(boc_data),
        "Sample Data": boc_sample
    }
except Exception as e:
    boc_summary = {"Error": str(e)}

print(boc_summary)