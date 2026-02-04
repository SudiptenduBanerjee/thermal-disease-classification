from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import os 
from src.config import Config
from src.dataset import get_loaders
from src.model import SOTA_ThermalModel # <-- CHANGED MODEL IMPORT

# Init Comet

experiment = Experiment() 

def log_weights_as_histograms(model, epoch):
    """Logs weight distributions to the Histograms tab (for SOTA analysis)"""
    # Handle nn.DataParallel wrapper if used
    model_to_log = model.module if isinstance(model, nn.DataParallel) else model
    for name, param in model_to_log.named_parameters():
        if 'weight' in name and param.grad is not None:
            experiment.log_histogram_3d(param.detach().cpu().numpy(), name, step=epoch)

def train_thermal_model():
    train_loader, val_loader, test_loader, classes = get_loaders()
    
    # 1. Initialize Model
    model = SOTA_ThermalModel(len(classes))
    
    # 2. Parallelize for training
    if Config.NUM_GPUS > 1:
        print(f"--- PARALLEL TRAINING ON {Config.NUM_GPUS} GPUs ---")
        train_model = nn.DataParallel(model)
    else:
        train_model = model
    
    train_model = train_model.to(Config.DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = optim.AdamW(train_model.parameters(), lr=Config.LEARNING_RATE)
    
    history = []
    best_acc = 0.0

    for epoch in range(Config.EPOCHS):
        train_model.train()
        train_loss, train_corrects = 0.0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = train_model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
        
        tr_loss = train_loss / len(train_loader.dataset)
        tr_acc = train_corrects.double() / len(train_loader.dataset)

        train_model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = train_model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        
        ts_loss = val_loss / len(val_loader.dataset)
        ts_acc = val_corrects.double() / len(val_loader.dataset)

        # --- FIX 1: Multiply by 100 so it prints 13.43% instead of 0.13% ---
        print(f"Epoch [{epoch+1:02d}/{Config.EPOCHS}] "
              f"T-Loss: {tr_loss:.4f} T-Acc: {tr_acc*100:.2f}% | "
              f"V-Loss: {ts_loss:.4f} V-Acc: {ts_acc*100:.2f}%")
        
        # --- FIX 2: Save the underlying model (model), not the wrapper (train_model) ---
        # This prevents the "module." prefix from being saved in the file!
        if ts_acc > best_acc:
            best_acc = ts_acc
            torch.save(model.state_dict(), os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH))
            print(f"   --> ⭐ New Best Model Saved! ({ts_acc*100:.2f}%)")
            experiment.log_metric("best_val_acc", best_acc)

        experiment.log_metrics({"tr_loss": tr_loss, "tr_acc": tr_acc, "val_loss": ts_loss, "val_acc": ts_acc}, step=epoch+1)
        history.append({"epoch": epoch+1, "train_acc": float(tr_acc), "val_acc": float(ts_acc)})

    # --- FINAL EVALUATION ---
    print("--- Training Done. Evaluating Best Model on Test Set ---")
    
    # --- FIX 3: Load directly into the raw model (no module prefix anymore) ---
    best_model_path = os.path.join(Config.MODEL_DIR, Config.BEST_MODEL_SAVE_PATH)
    model.load_state_dict(torch.load(best_model_path))
    model = model.to(Config.DEVICE)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # --- FIX 4: Confusion Matrix Fix ---
    experiment.log_confusion_matrix(
        y_true=all_labels, 
        y_predicted=all_preds, 
        labels=classes,
        title="SOTA Final Test Confusion Matrix"
    )

    final_test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"FINAL TEST ACCURACY: {final_test_acc*100:.2f}%")
    experiment.log_metric("final_test_acc", final_test_acc)

    with open(Config.HISTORY_JSON, 'w') as f:
        json.dump(history, f, indent=4)


if __name__ == "__main__":
    train_thermal_model()