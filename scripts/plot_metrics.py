import json
import os
import matplotlib.pyplot as plt

# Expects Training/history.json with keys: accuracy, val_accuracy, loss, val_loss
history_path = os.path.join('Training','history.json')
out_path = os.path.join('static','metrics.png')

if not os.path.exists(history_path):
    print('No Training/history.json found. Save Keras history as JSON first.')
    raise SystemExit(1)

with open(history_path,'r',encoding='utf-8') as f:
    history = json.load(f)

acc = history.get('accuracy')
val_acc = history.get('val_accuracy')
loss = history.get('loss')
val_loss = history.get('val_loss')

epochs = range(1, len(acc)+1)
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(epochs, acc, 'bo-', label='Training acc')
if val_acc:
    plt.plot(epochs, val_acc, 'ro-', label='Validation acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, loss, 'bo-', label='Training loss')
if val_loss:
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.legend()

plt.tight_layout()
plt.savefig(out_path)
print('Saved metrics to', out_path)