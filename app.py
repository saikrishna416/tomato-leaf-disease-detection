from flask import Flask, render_template, request, redirect, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from class_mapping import CLASS_NAMES, get_class_name
from remedies import get_remedies
import numpy as np
import os
import csv
from datetime import datetime
from utils import load_class_indices, decode_prediction, validate_class_mapping, is_healthy_prediction

# Optional OpenCV for basic leaf detection/segmentation
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    cv2 = None
    HAVE_CV2 = False

from PIL import Image as PILImage
import colorsys

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Load model at startup
try:
    model = load_model('Training/tomato_leaf_disease_model.h5')
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"ERROR: Failed to load model.h5: {e}")
    model = None

# Load and validate class indices
INDEX_TO_LABEL, LABEL_TO_INDEX = load_class_indices('class_indices.json')
if not INDEX_TO_LABEL:
    print("ERROR: class_indices.json not found or empty!")
    print("Expected: class_indices.json with mapping like {\"Healthy\": 0, \"Early Blight\": 1, ...}")
    sys.exit(1)

validate_class_mapping(INDEX_TO_LABEL)

# Human-friendly disease names
friendly_names = {
    'Tomato___Bacterial_spot': 'Bacterial Spot',
    'Tomato___Early_blight': 'Early Blight',
    'Tomato___Late_blight': 'Late Blight',
    'Tomato___Leaf_Mold': 'Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Septoria Leaf Spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite': 'Spider Mites',
    'Tomato___Target_Spot': 'Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Healthy'
}

# Convert model class name to clean name
def clean_name(raw_name):
    if raw_name in friendly_names:
        return friendly_names[raw_name]

    if raw_name.startswith("Tomato___"):
        raw_name = raw_name.replace("Tomato___", "")

    raw_name = raw_name.replace("_", " ")
    return raw_name.title()

# Basic treatment/advice mapping (general guidance)
treatment_advice = {

    'Tomato___Bacterial_spot': [
        'Spray copper-based bactericides at early stage.',
        'Remove and destroy infected leaves immediately.',
        'Avoid overhead watering to reduce spread.',
        'Use certified disease-free seeds.'
    ],

    'Tomato___Early_blight': [
        'Apply fungicides like chlorothalonil or mancozeb.',
        'Remove infected lower leaves regularly.',
        'Practice crop rotation every season.',
        'Ensure proper plant spacing for airflow.'
    ],

    'Tomato___Late_blight': [
        'Apply systemic fungicides such as metalaxyl.',
        'Remove severely infected plants immediately.',
        'Avoid water staying on leaves for long periods.',
        'Monitor plants during cool and humid weather.'
    ],

    'Tomato___Leaf_Mold': [
        'Improve greenhouse ventilation.',
        'Reduce humidity levels around plants.',
        'Apply recommended fungicides if infection spreads.',
        'Remove affected leaves carefully.'
    ],

    'Tomato___Septoria_leaf_spot': [
        'Remove infected leaves and plant debris.',
        'Apply protective fungicides when needed.',
        'Avoid overhead irrigation.',
        'Practice crop rotation.'
    ],

    'Tomato___Spider_mites_Two-spotted_spider_mite': [
        'Spray insecticidal soap or neem oil.',
        'Use miticides if infestation is severe.',
        'Increase humidity around plants.',
        'Introduce biological predators if possible.'
    ],

    'Tomato___Target_Spot': [
        'Apply recommended fungicides.',
        'Remove infected plant parts.',
        'Maintain good plant spacing.',
        'Avoid water splash between plants.'
    ],

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        'Remove infected plants immediately.',
        'Control whiteflies using insecticides or sticky traps.',
        'Use virus-resistant tomato varieties.',
        'Keep field free from weeds.'
    ],

    'Tomato___Tomato_mosaic_virus': [
        'Remove infected plants immediately.',
        'Disinfect tools after each use.',
        'Avoid handling plants after tobacco use.',
        'Use certified virus-free seeds.'
    ],

    'Tomato___healthy': [
        'No disease detected.',
        'Maintain proper irrigation and fertilization.',
        'Inspect plants regularly for early disease signs.'
    ]
}

# Expanded disease information (causes, symptoms, spread, prevention)
# Detailed Disease Information
disease_info = {

    'Tomato___Bacterial_spot': {
        'causes': 'Caused by Xanthomonas bacteria.',
        'symptoms': 'Small dark brown spots with yellow halos on leaves.',
        'spread': 'Spreads through rain splash, infected tools, and contaminated seeds.',
        'prevention': 'Use certified seeds, avoid overhead irrigation, sanitize tools.'
    },

    'Tomato___Early_blight': {
        'causes': 'Caused by the fungus Alternaria solani.',
        'symptoms': 'Concentric ring patterns on older leaves, yellowing, leaf drop.',
        'spread': 'Spreads through wind and rain splash from infected plant debris.',
        'prevention': 'Crop rotation, remove infected leaves, apply fungicides.'
    },

    'Tomato___Late_blight': {
        'causes': 'Caused by Phytophthora infestans.',
        'symptoms': 'Large dark water-soaked spots, rapid plant collapse.',
        'spread': 'Spreads quickly in cool and wet weather through wind and water.',
        'prevention': 'Avoid wet leaves, remove infected plants immediately.'
    },

    'Tomato___Leaf_Mold': {
        'causes': 'Caused by the fungus Passalora fulva.',
        'symptoms': 'Yellow patches on upper leaf surface, fuzzy mold underneath.',
        'spread': 'Spreads in high humidity conditions.',
        'prevention': 'Improve ventilation and reduce humidity.'
    },

    'Tomato___Septoria_leaf_spot': {
        'causes': 'Caused by Septoria lycopersici fungus.',
        'symptoms': 'Small circular spots with dark borders.',
        'spread': 'Spreads through rain splash and infected debris.',
        'prevention': 'Remove debris and avoid overhead watering.'
    },

    'Tomato___Spider_mites_Two-spotted_spider_mite': {
        'causes': 'Caused by spider mites infestation.',
        'symptoms': 'Speckled yellow leaves with fine webbing.',
        'spread': 'Spreads by wind and plant contact.',
        'prevention': 'Increase humidity and use appropriate miticides.'
    },

    'Tomato___Target_Spot': {
        'causes': 'Caused by fungal pathogens.',
        'symptoms': 'Target-like circular spots on leaves and fruit.',
        'spread': 'Spreads in warm and humid conditions.',
        'prevention': 'Apply fungicides and maintain plant spacing.'
    },

    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'causes': 'Caused by Tomato Yellow Leaf Curl Virus (TYLCV).',
        'symptoms': 'Leaf curling, yellowing, stunted growth.',
        'spread': 'Transmitted by whiteflies.',
        'prevention': 'Control whiteflies and use resistant varieties.'
    },

    'Tomato___Tomato_mosaic_virus': {
        'causes': 'Caused by Tomato Mosaic Virus.',
        'symptoms': 'Mosaic light and dark green patches on leaves.',
        'spread': 'Spreads through contaminated tools and hands.',
        'prevention': 'Sanitize tools and use virus-free seeds.'
    },

    'Tomato___healthy': {
        'causes': '🌿 Plant appears healthy. Continue proper irrigation, fertilization, and regular monitoring.',
        'symptoms': 'Leaf appears uniformly green and healthy.',
        'spread': 'Not applicable.',
        'prevention': 'Maintain proper watering and nutrient balance.'
    }
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Make disease_info available in templates
app.jinja_env.globals['disease_info'] = disease_info

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Simple green-area heuristic and optional leaf detection
def is_full_green(pil_image, green_thresh=0.30, spot_thresh=0.15):
    """Return True if image is mostly green and contains few dark/brown spots.
    pil_image: PIL.Image in RGB mode (any size)
    green_thresh: fraction of pixels that must be classified as green
    spot_thresh: max allowed fraction of dark/brown spot pixels
    """
    img = np.array(pil_image.convert('RGB'))
    # convert to HSV for robust green detection
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) if HAVE_CV2 else None
    if hsv is None:
        # fallback naive RGB check: green pixel where G much greater than R and B
        g = img[:,:,1].astype('int')
        r = img[:,:,0].astype('int')
        b = img[:,:,2].astype('int')
        green_mask = (g > r + 20) & (g > b + 20) & (g > 60)
        green_ratio = green_mask.mean()
        # dark/brown spot detection: pixels with low V (all channels low) and red bias
        dark_mask = (img.mean(axis=2) < 70)
        spot_ratio = dark_mask.mean()
    else:
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        # Hue for green roughly 25-95 (OpenCV scale 0-179)
        green_mask = ((h >= 25) & (h <= 95) & (s >= 40) & (v >= 40))
        green_ratio = green_mask.mean()
        # spots: low V and moderate S (brown/black spots)
        spot_mask = (v < 80) & (s > 30)
        spot_ratio = spot_mask.mean()

    return (green_ratio >= green_thresh) and (spot_ratio <= spot_thresh), float(green_ratio), float(spot_ratio)


def detect_leaf_and_crop(input_path):
    """If OpenCV available, attempt to find the largest green contour (leaf) and return cropped PIL.Image.
    If detection fails or cv2 not present, returns original PIL.Image."""
    img_pil = Image.open(input_path).convert('RGB')
    if not HAVE_CV2:
        return img_pil

    img = np.array(img_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # green mask
    mask = cv2.inRange(hsv, (25, 40, 40), (95, 255, 255))
    # clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_pil
    # largest contour
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    # pad a bit
    pad = int(0.05 * max(w,h))
    x0 = max(0, x-pad)
    y0 = max(0, y-pad)
    x1 = min(img.shape[1], x+w+pad)
    y1 = min(img.shape[0], y+h+pad)
    crop = img_pil.crop((x0,y0,x1,y1))
    return crop

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # Collect available sample images from static/samples
    samples_dir = os.path.join('static', 'samples')
    samples = []
    if os.path.exists(samples_dir):
        samples = sorted(os.listdir(samples_dir))
    # Prepare slider images: prefer static/images if present, otherwise pick a few leaf images from static
    images_dir = os.path.join('static', 'images')
    slider_images = []
    if os.path.exists(images_dir):
        slider_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('jpg','jpeg','png'))])
        # convert to paths relative to static
        slider_images = [os.path.join('images', f) for f in slider_images[:6]]
    else:
        # fallback: look in static root for obvious leaf images
        candidates = [f for f in os.listdir('static') if f.lower().endswith(('jpg','jpeg','png'))]
        # prefer names containing 'leaf' or 'tomato'
        prioritized = [f for f in candidates if ('leaf' in f.lower() or 'tomato' in f.lower())]
        use = prioritized + [f for f in candidates if f not in prioritized]
        slider_images = [f for f in use[:6]]

    return render_template('index.html', samples=samples, slider_images=slider_images)

@app.route('/details')
def details():
    return render_template('details.html', treatment_advice=treatment_advice, friendly_names=friendly_names, disease_info=disease_info)

def save_prediction_record(disease, confidence, image_filename=''):
    """Save prediction record to CSV with image filename"""
    history_file = 'prediction_history.csv'
    file_exists = os.path.exists(history_file)
    
    try:
        with open(history_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['date', 'disease', 'confidence', 'image_path'])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'disease': disease,
                'confidence': f'{confidence:.4f}',
                'image_path': image_filename
            })
        print(f"✓ Record saved: {disease} - {image_filename}")
    except Exception as e:
        print(f"Error saving record: {e}")

# helper: compute green pixel ratio from a PIL image
def compute_green_ratio_from_pil(pil_img, downscale=128):
    """Compute fraction of green pixels using HSV"""
    import colorsys
    import numpy as np
    
    img = pil_img.copy().convert('RGB')
    img.thumbnail((downscale, downscale))
    arr = np.array(img) / 255.0
    flat = arr.reshape(-1, 3)
    total = flat.shape[0]
    if total == 0:
        return 0.0
    
    green_count = 0
    for r, g, b in flat:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if 0.18 <= h <= 0.50 and s >= 0.20 and v >= 0.15:
            green_count += 1
    return green_count / total

def compute_spot_ratio_from_pil(pil_img, downscale=128):
    """Compute fraction of pixels that look like spots/lesions"""
    import colorsys
    import numpy as np
    
    img = pil_img.copy().convert('RGB')
    img.thumbnail((downscale, downscale))
    arr = np.array(img) / 255.0
    flat = arr.reshape(-1, 3)
    total = flat.shape[0]
    if total == 0:
        return 0.0
    
    spot_count = 0
    for r, g, b in flat:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # skip strong greens
        if 0.18 <= h <= 0.50 and s >= 0.20 and v >= 0.15:
            continue
        # yellow/orange (chlorosis)
        if 0.05 <= h <= 0.18 and s >= 0.15 and v >= 0.15:
            spot_count += 1
            continue
        # brown/dark (necrosis)
        if 0.02 <= h <= 0.12 and s >= 0.15 and 0.03 <= v <= 0.6:
            spot_count += 1
            continue
        # very dark/black
        if v <= 0.10 and s >= 0.10:
            spot_count += 1
    return spot_count / total
@app.route('/predict', methods=['POST'])
def predict():
    """Predict disease from uploaded image"""
    
    if not model:
        flash('Model not loaded. Please restart the app.', 'error')
        return redirect('/')

    # Get both possible inputs
    gallery_file = request.files.get('gallery_image')
    camera_file = request.files.get('camera_image')

    file = None

    if gallery_file and gallery_file.filename != '':
        file = gallery_file
    elif camera_file and camera_file.filename != '':
        file = camera_file

    if file is None:
        flash('No image selected', 'error')
        return redirect('/')

    try:
        from PIL import Image
        import numpy as np

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{file.filename}")
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Open image
        pil_img = Image.open(filepath).convert('RGB')

        # Preprocess for model
        img_for_model = pil_img.resize((224, 224))
        img_array = np.array(img_for_model) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array, verbose=0)
        pred_probs = predictions[0]

        predicted_class, confidence, all_probs = decode_prediction(
           
            pred_probs,
            INDEX_TO_LABEL,
            unknown_threshold=0.25
        )
        display_name = clean_name(predicted_class)

        # Green override logic
        green_ratio = compute_green_ratio_from_pil(pil_img, downscale=128)
        spot_ratio = compute_spot_ratio_from_pil(pil_img, downscale=128)

        if green_ratio >= 0.85 and confidence < 0.90 and spot_ratio < 0.02:
            predicted_class = 'Healthy'
            confidence = max(confidence, green_ratio)

        # Build probability dictionary
        all_probs = {}
        for idx in range(len(pred_probs)):
            raw_name = INDEX_TO_LABEL.get(idx, f'Unknown_{idx}')
            clean_display = clean_name(raw_name)
            all_probs[clean_display] = float(pred_probs[idx]) * 100

        all_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
        # Clean display name
        display_name = clean_name(predicted_class)

        # Get disease info safely
        info = disease_info.get(predicted_class, {})

        remedies = {
            "description": info.get("causes", "Information not available."),
            "symptoms": info.get("symptoms", "Information not available."),
            "treatment": treatment_advice.get(predicted_class, []),
            "prevention": [info.get("prevention", "Maintain good agricultural practices.")]
        }
        save_prediction_record(display_name, confidence, filename)
        return render_template(
            'result.html',
            predicted_class=display_name,
            confidence=confidence,
            all_probabilities=all_probs,
            remedies=remedies,
            image_file=filename
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f'Error: {str(e)}', 'error')
        return redirect('/')

@app.route('/history')
def history():
    """Display prediction history"""
    rows = []
    try:
        history_file = 'prediction_history.csv'
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader) if reader else []
                rows.reverse()  # newest first
    except Exception as e:
        print(f"Error loading history: {e}")
    
    return render_template('history.html', rows=rows, cleared=False)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history"""
    try:
        history_file = 'prediction_history.csv'
        if os.path.exists(history_file):
            os.remove(history_file)
        print("✓ History cleared")
    except Exception as e:
        print(f"Error clearing history: {e}")
    
    return render_template('history.html', rows=[], cleared=True)

@app.route('/download_history')
def download_history():
    """Download history as CSV"""
    from flask import send_file
    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        return send_file(history_file, as_attachment=True, download_name='prediction_history.csv')
    return redirect('/history')

import json

@app.route('/metrics')
def metrics():

    with open("real_metrics.json") as f:
        metrics_data = json.load(f)

    return render_template(
        "metrics.html",
        accuracy=metrics_data["accuracy"],
        confusion_matrix=metrics_data["confusion_matrix"],
        class_names=metrics_data["class_names"],
        report=metrics_data["classification_report"]
    )
@app.route('/predict_sample/<path:filename>')
def predict_sample(filename):
    # Predict using a sample image located in static/samples
    filepath = os.path.join('static', 'samples', filename)
    if not os.path.exists(filepath):
        return 'Sample not found', 404

    # Try to detect leaf and crop to the leaf area if possible
    cropped_pil = detect_leaf_and_crop(filepath)
    is_green, green_ratio, spot_ratio = is_full_green(cropped_pil)

    cropped_resized = cropped_pil.resize((224,224))
    img_array = image.img_to_array(cropped_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0]

    # If clear full-green leaf, override to healthy
    if is_green:
        result = 'Tomato___healthy'
        top_prob = float(prediction[class_names.index('Tomato___healthy')])
        friendly = friendly_names.get(result, result)
        advice = treatment_advice.get(result, ['No advice available'])
        low_confidence = False
        top3 = [
            {'name': class_names[i], 'friendly': friendly_names.get(class_names[i], class_names[i]), 'prob': float(prediction[i])}
            for i in list(np.argsort(prediction)[::-1][:3])
        ]
        conf_val = top_prob
    else:
        CONF_THRESHOLD = 0.65
        HEALTHY_BOOST_THRESHOLD = 0.40
        top_idx = int(np.argmax(prediction))
        top_prob = float(prediction[top_idx])
        healthy_idx = class_names.index('Tomato___healthy')
        top_indices = list(np.argsort(prediction)[::-1][:3])
        top3 = [
            {'name': class_names[i], 'friendly': friendly_names.get(class_names[i], class_names[i]), 'prob': float(prediction[i])}
            for i in top_indices
        ]

        if top_prob < CONF_THRESHOLD:
            if prediction[healthy_idx] >= HEALTHY_BOOST_THRESHOLD:
                result = 'Tomato___healthy'
                friendly = friendly_names['Tomato___healthy']
                advice = treatment_advice['Tomato___healthy']
                low_confidence = False
            else:
                result = 'Uncertain'
                friendly = 'Uncertain (low confidence)'
                advice = ['Prediction confidence is low. Please try a clearer image or more lighting.']
                low_confidence = True
        else:
            result = class_names[top_idx]
            friendly = friendly_names.get(result, result)
            advice = treatment_advice.get(result, ['No advice available'])
            low_confidence = False
        conf_val = top_prob

    # Append to history CSV for tracking
    history_path = os.path.join('history.csv')
    header_needed = not os.path.exists(history_path)
    with open(history_path, 'a', encoding='utf-8') as f:
        if header_needed:
            f.write('timestamp,image,disease,confidence,green_ratio,spot_ratio\n')
        f.write(f"{int(time.time())},{os.path.basename(filepath)},{result},{conf_val:.4f},{green_ratio:.4f},{spot_ratio:.4f}\n")

    return render_template('results.html', prediction=result, friendly_name=friendly, advice=advice, image_path=filepath, top3=top3, confidence=conf_val, low_confidence=low_confidence, is_green=is_green)
   
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
