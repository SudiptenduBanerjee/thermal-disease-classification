from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os
import time
from tqdm import tqdm # You need to install this: pip install tqdm

from src.config import Config
from src.dataset import get_loaders
from src.model import MobileNetV2_Attention

# --- 1. COMET SETUP ---
experiment = Experiment() 

# Log Hyperparameters explicitly for the dashboard
params = {
    "model": Config.MODEL_NAME,
    "epochs": Config.EPOCHS,
    "batch_size": Config.BATCH_SIZE,
    "learning_rate": Config.LEARNING_RATE,
    "image_size": Config.IMAGE_SIZE,
    "num_classes": Config.NUM_CLASSES,
    "optimizer": "AdamW",
    "smoothing": Config.LABEL_SMOOTHING
}
experiment.log_parameters(params)

def log_weights_as_histograms(model, epoch):
    """Logs 3D histograms of weights to Comet to visualize learning health"""
    model_to_log = model.module if isinstance(model, nn.DataParallel) else model
    
    for name, param in model_to_log.named_parameters():
        # Only log attention and classifier weights to keep logs clean
        if ('attention' in name or 'classifier' in name) and 'weight' in name:
            experiment.log_histogram_3d(param.detach().cpu().numpy(), name=name, step=epoch)

def train_thermal_model():
    train_loader, val_loader, test_loader, classes = get_loaders()
    
    print(f"Initializing {Config.MODEL_NAME}...")
    model = MobileNetV2_Attention(len(classes))
    
    # --- PARALLELIZATION ---
    if Config.NUM_GPUS > 1:
        print(f"--- PARALLEL TRAINING ON {Config.NUM_GPUS} GPUs ---")
        train_model = nn.DataParallel(model)
    else:
        train_model = model
    
    train_model = train_model.to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(train_model.parameters(), lr=Config.LEARNING_RATE)
    # Cosine Annealing helps reach better accuracy in later epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    history = []
    best_acc = 0.0

    print(f"Starting Training: {Config.EPOCHS} Epochs | Batch: {Config.BATCH_SIZE}")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        # --- TRAINING LOOP ---
        train_model.train()
        train_loss, train_corrects = 0.0, 0
        
        # TQDM Status Bar
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]") as tepoch:
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                optimizer.zero_grad()
                outputs = train_model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                
                loss.backward()
                optimizer.step()
                
                # Update Stats
                loss_val = loss.item()
                train_loss += loss_val * inputs.size(0)
                train_corrects += torch.sum(preds == labels.data)
                
                # Live update of the progress bar
                tepoch.set_postfix(loss=loss_val)
        
        tr_loss = train_loss / len(train_loader.dataset)
        tr_acc = train_corrects.double() / len(train_loader.dataset)

        # --- VALIDATION LOOP ---
        train_model.eval()
        val_loss, val_corrects = 0.0, 0
        
        with tqdm(val_loader, unit="batch", desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]") as vepoch:
            with torch.no_grad():
                for inputs, labels in vepoch:
                    inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                    outputs = train_model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
        
        ts_loss = val_loss / len(val_loader.dataset)
        ts_acc = val_corrects.double() / len(val_loader.dataset)
        
        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        duration = time.time() - start_time

        # --- 1. DATA SAVING (JSON FORMAT REQ) ---
        epoch_data = {
            "epoch": epoch + 1,
            "train_acc": float(tr_acc),
            "val_acc": float(ts_acc),
            "train_loss": float(tr_loss),
            "val_loss": float(ts_loss),
            "lr": float(current_lr)
        }
        history.append(epoch_data)

        # --- 2. TERMINAL STATUS BAR (DETAILED) ---
        print(f"✅ Ep {epoch+1} Summary | T-Acc: {tr_acc*100:.2f}% | V-Acc: {ts_acc*100:.2f}% | "
              f"T-Loss: {tr_loss:.4f} | V-Loss: {ts_loss:.4f} | Time: {duration:.1f}s")
        print("-" * 70)

        # --- 3. COMET LOGGING ---
        experiment.log_metrics(epoch_data, step=epoch+1)
        
        # Log Histograms every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_weights_as_histograms(model, epoch+1)

        # Save Best Model
        if ts_acc > best_acc:
            best_acc = ts_acc
            torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH))
            experiment.log_metric("best_val_acc", best_acc)

        # Save JSON file immediately
        with open(Config.HISTORY_JSON, 'w') as f:
            json.dump(history, f, indent=4)

    # --- FINAL TESTING ---
    print("\n--- Training Complete. Running Final Test Evaluation ---")
    
    # Load Best Weights
    best_model_path = os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(Config.DEVICE)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Final Test"):
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Comet Confusion Matrix
    experiment.log_confusion_matrix(
        y_true=all_labels, 
        y_predicted=all_preds, 
        labels=classes,
        title=f"Final Test Matrix - {Config.MODEL_NAME}"
    )

    final_test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n🏆 FINAL TEST ACCURACY: {final_test_acc*100:.2f}%")
    experiment.log_metric("final_test_acc", final_test_acc)

if __name__ == "__main__":
    train_thermal_model()