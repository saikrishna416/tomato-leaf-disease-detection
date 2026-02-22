"""
Tomato Leaf Disease Classification
10 classes mapped to indices (0-9)
"""

CLASS_NAMES = [
    'Healthy',
    'Bacterial Spot',
    'Early Blight',
    'Late Blight',
    'Leaf Mold',
    'Septoria Leaf Spot',
    'Spider Mites',
    'Target Spot',
    'Tomato Yellow Leaf Curl Virus',
    'Tomato Mosaic Virus'
]

CLASS_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

def get_class_name(index):
    """Get class name from index"""
    return INDEX_CLASS.get(index, 'Unknown')

def get_class_index(name):
    """Get class index from name"""
    return CLASS_INDEX.get(name, -1)
