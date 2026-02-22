"""
Utilities for class label management and prediction decoding
"""
import json
import os
import numpy as np

def load_class_indices(json_file='class_indices.json'):
    """
    Load class indices from JSON file created during training.
    
    Returns:
        index_to_label: dict mapping index (int) -> class name (str)
        label_to_index: dict mapping class name (str) -> index (int)
    """
    if not os.path.exists(json_file):
        print(f"Warning: {json_file} not found. Using default mapping.")
        default_mapping = {
            "Healthy": 0,
            "Bacterial Spot": 1,
            "Early Blight": 2,
            "Late Blight": 3,
            "Leaf Mold": 4,
            "Septoria Leaf Spot": 5,
            "Spider Mites": 6,
            "Target Spot": 7,
            "Tomato Yellow Leaf Curl Virus": 8,
            "Tomato Mosaic Virus": 9
        }
        return {v: k for k, v in default_mapping.items()}, default_mapping
    
    try:
        with open(json_file, 'r') as f:
            label_to_index = json.load(f)
        
        # Reverse: index -> label
        index_to_label = {int(v): k for k, v in label_to_index.items()}
        
        print(f"\n✓ Loaded {len(index_to_label)} classes from {json_file}:")
        for idx in sorted(index_to_label.keys()):
            is_healthy = " [HEALTHY]" if index_to_label[idx] == "Healthy" else ""
            print(f"  Index {idx}: {index_to_label[idx]}{is_healthy}")
        print()
        
        return index_to_label, label_to_index
    
    except Exception as e:
        print(f"Error loading class indices: {e}")
        return {}, {}

def validate_class_mapping(index_to_label):
    """
    Validate that Healthy class is at index 0 (as expected).
    Print warning if Healthy is mapped to any other index.
    """
    healthy_idx = None
    for idx, label in index_to_label.items():
        if label == "Healthy":
            healthy_idx = idx
            break
    
    if healthy_idx is None:
        print("⚠️  WARNING: Healthy class not found in mapping!")
        return False
    
    if healthy_idx != 0:
        print(f"⚠️  WARNING: Healthy class is at index {healthy_idx}, expected 0!")
        print("   This may cause healthy leaves to be misclassified as diseased.")
        return False
    
    print("✓ Healthy class correctly mapped to index 0")
    return True

def get_class_name(index, index_to_label):
    """
    Get class name from prediction index.
    Always returns the real disease name, never returns "Class_X" placeholders.
    """
    if index in index_to_label:
        return index_to_label[index]
    else:
        # fallback: if index not in mapping, still try to use a reasonable default
        # rather than "Class_X" which is confusing
        print(f"Warning: index {index} not found in mapping, using 'Unknown'")
        return "Unknown"

def decode_prediction(prediction, index_to_label, unknown_threshold=0.40):
    """
    Decode model prediction array to class name and confidence.
    
    Args:
        prediction: numpy array of shape (num_classes,) with probabilities [0..1]
        index_to_label: dict mapping index -> class name
        unknown_threshold: if top confidence < this, return "Unknown" (default 0.40 = 40%)
    
    Returns:
        (class_name, confidence, all_probabilities)
    """
    if not isinstance(prediction, np.ndarray):
        prediction = np.array(prediction)
    
    # Get predicted class (highest probability)
    predicted_idx = int(np.argmax(prediction))
    confidence = float(prediction[predicted_idx])
    
    # Only mark as Unknown if confidence is very low (< 40%)
    if confidence < unknown_threshold:
        predicted_class = "Unknown"
    else:
        predicted_class = get_class_name(predicted_idx, index_to_label)
    
    # Get all class probabilities sorted by confidence
    all_probs = {}
    for idx in range(len(prediction)):
        class_name = get_class_name(idx, index_to_label)
        all_probs[class_name] = float(prediction[idx]) * 100
    
    # Sort by confidence descending
    all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
    
    return predicted_class, confidence, all_probs

def is_healthy_prediction(predicted_class):
    """
    Check if prediction is Healthy class.
    Healthy = no disease = clean green leaves.
    """
    return predicted_class == "Healthy"
