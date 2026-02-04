import torch

class Config:
    # --- COMET.ML SETUP ---
    # COMET_API_KEY = "YOUR_API_KEY_HERE" 
    # PROJECT_NAME = "thermal-mobilenet-attention"
    # WORKSPACE = None 

    # --- DATA PATHS ---
    # Pointing to your specific dataset structure
    DATA_ROOT = "./dataset/research_experiments/split_70_10_20" 
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR = f"{DATA_ROOT}/val"
    TEST_DIR = f"{DATA_ROOT}/test"
    
    # --- MODEL CONFIG ---
    MODEL_NAME = "MobileNetV2_Attention"
    NUM_CLASSES = 15  # Based on your folder structure (Citrus, Guava, Mango, etc.)
    IMAGE_SIZE = (224, 224)
    
    # --- HYPERPARAMETERS ---
    BATCH_SIZE = 128  # High batch size for 4 GPUs
    EPOCHS = 50       # As requested by your sir
    LEARNING_RATE = 1e-4 
    LABEL_SMOOTHING = 0.1
    
    # --- LOGGING & SAVING ---
    MODEL_DIR = "./"
    # Specific JSON filename
    HISTORY_JSON = f"training_history_{MODEL_NAME}.json" 
    BEST_MODEL_SAVE_PATH = f"{MODEL_NAME}_best.pth"
    
    # --- HARDWARE ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()