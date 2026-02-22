import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys
import argparse

# --- CONFIG ---
parser = argparse.ArgumentParser(description='Train tomato leaf model')
parser.add_argument('--data-dir', default=None, help='Path to dataset root (contains train/ and val/)')
args = parser.parse_args()

DATA_DIR = None
if args.data_dir:
    DATA_DIR = Path(args.data_dir)
else:
    DATA_DIR = Path(os.environ.get('TRAIN_DATA_DIR', 'data'))

TRAIN_DIR = DATA_DIR / 'train'
VAL_DIR = DATA_DIR / 'val'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 20
MODEL_OUT = Path('model.h5')
CLASS_INDICES_OUT = Path('class_indices.json')
METRICS_IMG_OUT = Path('static/images/metrics.png')
FINE_TUNE = True
FINE_TUNE_AT = 100

os.makedirs(METRICS_IMG_OUT.parent, exist_ok=True)

# --- VALIDATE DATASET FOLDERS ---
def _folder_has_subdirs(p: Path):
    try:
        return any((p / d).is_dir() for d in os.listdir(p))
    except Exception:
        return False

if not TRAIN_DIR.exists():
    print(f"\nERROR: training directory not found: {TRAIN_DIR}")
    print("\nPrepare your dataset with this folder structure:")
    print(f"  {TRAIN_DIR / 'Healthy'}/*.jpg")
    print(f"  {TRAIN_DIR / 'Early Blight'}/*.jpg")
    print(f"  {TRAIN_DIR / 'Late Blight'}/*.jpg")
    print(f"  ... (add more disease classes as needed)")
    print(f"\n  {VAL_DIR / 'Healthy'}/*.jpg")
    print(f"  {VAL_DIR / 'Early Blight'}/*.jpg")
    print(f"  ... (validation set)")
    print(f"\nThen run:")
    print(f"  python train.py --data-dir {DATA_DIR}\n")
    sys.exit(1)

if not VAL_DIR.exists():
    print(f"\nERROR: validation directory not found: {VAL_DIR}")
    print(f"Create validation folders as shown above.\n")
    sys.exit(1)

if not _folder_has_subdirs(TRAIN_DIR):
    print(f"\nERROR: no class subfolders found under: {TRAIN_DIR}")
    print("Each disease class should be a subfolder with images.")
    print("Example: data/train/Healthy/, data/train/Early Blight/, etc.\n")
    sys.exit(1)

if not _folder_has_subdirs(VAL_DIR):
    print(f"\nERROR: no class subfolders found under: {VAL_DIR}")
    print("Each disease class should be a subfolder with images.\n")
    sys.exit(1)

print(f"✓ Using dataset from: {DATA_DIR.resolve()}\n")

# --- Data generators ---
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.1,
    zoom_range=0.12,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode='nearest'
)
val_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    str(TRAIN_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_aug.flow_from_directory(
    str(VAL_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# save class indices to JSON (label -> index)
label_to_index = train_gen.class_indices
with open(CLASS_INDICES_OUT, 'w', encoding='utf-8') as f:
    json.dump(label_to_index, f, indent=2, ensure_ascii=False)
print(f"Saved class indices -> {CLASS_INDICES_OUT}")
print("Class mapping:", label_to_index)

# compute class weights to help under-represented classes (helps Healthy if minority)
classes, counts = np.unique(train_gen.classes, return_counts=True)
total = counts.sum()
num_classes = len(label_to_index)
class_weight = {int(c): float(total / (num_classes * cnt)) for c, cnt in zip(classes, counts)}
print("Class counts:", dict(zip(classes.tolist(), counts.tolist())))
print("Class weights:", class_weight)

# --- Build model ---
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# callbacks
callbacks = [
    ModelCheckpoint(str(MODEL_OUT), monitor='val_accuracy', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

# --- Train head first ---
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=class_weight,
    callbacks=callbacks
)

# Optional fine-tuning
if FINE_TUNE:
    base_model.trainable = True
    # freeze until layer FINE_TUNE_AT
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Fine-tuning model...")
    ft_history = model.fit(
        train_gen,
        epochs=EPOCHS // 2,
        validation_data=val_gen,
        class_weight=class_weight,
        callbacks=callbacks
    )
    # merge histories
    for k in history.history.keys():
        history.history[k] = history.history[k] + ft_history.history.get(k, [])

# ensure best model saved
model.save(str(MODEL_OUT))
print(f"Saved model -> {MODEL_OUT}")

# --- Evaluate on validation set and build confusion matrix ---
# predict on full validation set
val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
y_pred_prob = model.predict(val_gen, steps=val_steps, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_gen.classes  # flow_from_directory preserves order when shuffle=False

# confusion matrix & classification report
cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
report = classification_report(y_true, y_pred, target_names=[k for k, v in sorted(label_to_index.items(), key=lambda x: x[1])], digits=4)
print("\nClassification Report:\n", report)

# --- Plot metrics and confusion matrix ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
# Accuracy
axes[0].plot(history.history.get('accuracy', []), label='train_acc')
axes[0].plot(history.history.get('val_accuracy', []), label='val_acc')
axes[0].set_title('Accuracy')
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history.get('loss', []), label='train_loss')
axes[1].plot(history.history.get('val_loss', []), label='val_loss')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True)

# Confusion matrix
im = axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[2].set_title('Confusion Matrix')
tick_labels = [k for k, v in sorted(label_to_index.items(), key=lambda x: x[1])]
axes[2].set_xticks(np.arange(len(tick_labels)))
axes[2].set_yticks(np.arange(len(tick_labels)))
axes[2].set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
axes[2].set_yticklabels(tick_labels, fontsize=9)
plt.colorbar(im, ax=axes[2])

plt.tight_layout()
plt.savefig(str(METRICS_IMG_OUT), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved metrics image -> {METRICS_IMG_OUT}")
print("Done.")
