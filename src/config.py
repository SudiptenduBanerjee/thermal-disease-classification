import torch

class Config:
    
    # --- DATA PATHS ---
    DATA_ROOT = "./dataset/research_experiments/split_70_10_20" 
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR = f"{DATA_ROOT}/val"
    TEST_DIR = f"{DATA_ROOT}/test"
    
    # --- MODEL CONFIG ---
    MODEL_NAME = "SOTA_ConvNeXt_CBAM" # Changed for SOTA
    NUM_CLASSES = 15
    IMAGE_SIZE = (224, 224)
    
    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 64  # Adjusted for ConvNeXt memory usage on 4 GPUs
    EPOCHS = 100     # SOTA requires more epochs with MixUp
    LEARNING_RATE = 4e-5 
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA = 0.2 # Added for SOTA Regularization
    
    # --- LOGGING & SAVING ---
    MODEL_DIR = "./"
    HISTORY_JSON = f"training_history_{MODEL_NAME}.json" 
    BEST_MODEL_SAVE_PATH = f"{MODEL_NAME}_best.pth"
    
    # --- HARDWARE ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()