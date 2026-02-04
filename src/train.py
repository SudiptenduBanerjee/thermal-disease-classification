from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
import time
from tqdm import tqdm

from src.config import Config
from src.dataset import get_loaders
from src.model import SOTA_Thermal_ConvNeXt # Ensure this matches your model file

# --- 1. COMET SETUP ---
experiment = Experiment() 

# MixUp Implementation for SOTA regularization
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(Config.DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_thermal_model():
    train_loader, val_loader, test_loader, classes = get_loaders()
    
    # Initialize SOTA Model
    model = SOTA_Thermal_ConvNeXt(len(classes))
    
    if Config.NUM_GPUS > 1:
        print(f"--- PARALLEL TRAINING ON {Config.NUM_GPUS} GPUs ---")
        train_model = nn.DataParallel(model)
    else:
        train_model = model
    
    train_model = train_model.to(Config.DEVICE)
    
    # SOTA Loss & Optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(train_model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    history, best_acc = [], 0.0

    print(f"Starting SOTA V3 Training: {Config.EPOCHS} Epochs")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        # --- TRAINING LOOP ---
        train_model.train()
        train_loss, train_corrects = 0.0, 0
        
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                # Apply MixUp
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, Config.MIXUP_ALPHA)
                
                optimizer.zero_grad()
                outputs = train_model(inputs)
                
                # SOTA Loss calculation
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                tepoch.set_postfix(loss=loss.item())
        
        # --- VALIDATION LOOP ---
        train_model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = train_model(inputs)
                loss = criterion(outputs, labels) # No mixup in val
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        # Metrics Calculation
        tr_loss = train_loss / len(train_loader.dataset)
        ts_loss = val_loss / len(val_loader.dataset)
        ts_acc = val_corrects.double() / len(val_loader.dataset)
        current_lr = optimizer.param_groups[0]['lr']
        
        # --- LOGGING TO COMET ---
        metrics = {
            "tr_loss": float(tr_loss),
            "ts_loss": float(ts_loss),
            "ts_acc": float(ts_acc),
            "lr": float(current_lr)
        }
        experiment.log_metrics(metrics, step=epoch+1)
        
        # Print Summary
        print(f"✅ Ep {epoch+1} | V-Acc: {ts_acc*100:.2f}% | V-Loss: {ts_loss:.4f} | LR: {current_lr:.6f}")

        # Save History & Model
        history.append({"epoch": epoch + 1, **metrics})
        if ts_acc > best_acc:
            best_acc = ts_acc
            torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH))
            experiment.log_metric("best_val_acc", float(best_acc))

        scheduler.step()

    # --- FINAL TEST & CONFUSION MATRIX ---
    print("\n--- Training Complete. Generating Final Comet Report ---")
    best_path = os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    experiment.log_confusion_matrix(
        y_true=all_labels, 
        y_predicted=all_preds, 
        labels=classes,
        title="SOTA V3 Final Matrix"
    )

    with open(Config.HISTORY_JSON, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    train_thermal_model()