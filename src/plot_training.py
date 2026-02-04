import json
import matplotlib.pyplot as plt
import os
from src.config import Config

def plot_history():
    history_path = Config.HISTORY_JSON
    
    # Check if file exists
    if not os.path.exists(history_path):
        print(f"❌ Error: Could not find {history_path}")
        return

    # 1. Load the data
    with open(history_path, 'r') as f:
        history = json.load(f)

    # 2. Parse Data (Handle List vs Dictionary automatically)
    if isinstance(history, list):
        # New SOTA Format (List of Dicts)
        epochs = [x['epoch'] for x in history]
        tr_acc = [x['train_acc'] for x in history]
        val_acc = [x['val_acc'] for x in history]
        
        # Check if loss was saved (it might be missing in some versions)
        if 'train_loss' in history[0]:
            tr_loss = [x['train_loss'] for x in history]
            val_loss = [x['val_loss'] for x in history]
        else:
            tr_loss, val_loss = None, None
            print("⚠️ Note: Loss data was not found in JSON. Check Comet.ml for Loss graphs.")
            
    else:
        # Old Format (Dict of Lists)
        tr_acc = history.get('tr_acc') or history.get('train_acc')
        val_acc = history.get('ts_acc') or history.get('val_acc')
        tr_loss = history.get('tr_loss') or history.get('train_loss')
        val_loss = history.get('ts_loss') or history.get('val_loss')
        epochs = range(1, len(tr_acc) + 1)

    # --- PLOT 1: ACCURACY ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, tr_acc, 'b-o', label='Training Acc')
    plt.plot(epochs, val_acc, 'r-s', label='Validation Acc')
    plt.title(f'SOTA Model Accuracy ({Config.NUM_CLASSES} classes)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path_acc = 'accuracy_curve_sota.png'
    plt.savefig(save_path_acc, dpi=300)
    print(f"✅ Saved Accuracy Plot: {save_path_acc}")
    # plt.show() # Commented out for headless server environments

    # --- PLOT 2: LOSS (Only if available) ---
    if tr_loss and val_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, tr_loss, 'b-o', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-s', label='Validation Loss')
        plt.title('SOTA Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path_loss = 'loss_curve_sota.png'
        plt.savefig(save_path_loss, dpi=300)
        print(f"✅ Saved Loss Plot: {save_path_loss}")

if __name__ == "__main__":
    plot_history()