"""
Generate a sample metrics plot and save as static/metrics.png
Run this script after training to visualize model accuracy and loss.
Or use it with mock data for demo purposes.
"""
import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Paths
history_path = os.path.join('Training', 'history.json')
out_path = os.path.join('static', 'metrics.png')

os.makedirs('static', exist_ok=True)

# Try to load training history; if not found, create mock data for demo
if os.path.exists(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
else:
    # Mock data for demo (replace with real training history once available)
    epochs_count = 20
    history = {
        'accuracy': list(np.linspace(0.5, 0.92, epochs_count)),
        'val_accuracy': list(np.linspace(0.48, 0.88, epochs_count)),
        'loss': list(np.linspace(1.5, 0.2, epochs_count)),
        'val_loss': list(np.linspace(1.6, 0.25, epochs_count))
    }

acc = history.get('accuracy', [])
val_acc = history.get('val_accuracy', [])
loss = history.get('loss', [])
val_loss = history.get('val_loss', [])

if not acc:
    print('Error: No accuracy data found in history.')
    exit(1)

epochs = range(1, len(acc) + 1)

# Create figure with two subplots
plt.figure(figsize=(12, 5))

# Accuracy subplot
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo-', label='Training Accuracy', linewidth=2, markersize=4)
if val_acc:
    plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy', linewidth=2, markersize=4)
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Loss subplot
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo-', label='Training Loss', linewidth=2, markersize=4)
if val_loss:
    plt.plot(epochs, val_loss, 'ro-', label='Validation Loss', linewidth=2, markersize=4)
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_path, dpi=100, bbox_inches='tight')
print(f'✅ Metrics plot saved to {out_path}')
plt.close()
