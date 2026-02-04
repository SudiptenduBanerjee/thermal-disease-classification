import torch
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os

from src.config import Config
from src.dataset import get_loaders
from src.model import SOTA_ThermalModel  # <--- CHANGED: Import the new SOTA model

def plot_confusion_matrix():
    print("--- Generating SOTA Confusion Matrix ---")
    
    # 1. Get Data and Classes
    _, _, test_loader, classes = get_loaders()
    
    # 2. Initialize the SOTA Model
    model = SOTA_ThermalModel(len(classes))
    model = model.to(Config.DEVICE)
    
    # 3. Load the Weights (Robust Loading)
    # We look for the best saved model from your config
    weights_path = os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH)
    
    print(f"Loading weights from: {weights_path}")
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=Config.DEVICE)
        
        # FIX: Check if weights have 'module.' prefix (from DataParallel) and remove it if needed
        if list(state_dict.keys())[0].startswith('module.'):
            print("Detected DataParallel weights. Removing 'module.' prefix...")
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
            
        print("Weights loaded successfully.")
    else:
        print(f"ERROR: Weights not found at {weights_path}. Please check the path.")
        return

    # 4. Run Inference
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Running inference on Test Set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # 5. Calculate & Plot Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'SOTA Confusion Matrix (ConvNeXt-Tiny + Attention)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    save_path = "confusion_matrix_sota.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")
    
    # Calculate Final Accuracy
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Final Calculated Test Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    plot_confusion_matrix()