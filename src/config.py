import torch

class Config:
    # Comet.ml Setup (Remember to uncomment and fill this in!)
    # COMET_API_KEY = "YOUR_API_KEY_HERE" 
    # PROJECT_NAME = "thermal-sota-upgrade"
    # WORKSPACE = None 

    # Explicit Dataset Paths (Assuming your 15-class structure)
    DATA_ROOT = "./dataset/research_experiments/split_70_10_20" 
    TRAIN_DIR = f"{DATA_ROOT}/train"
    VAL_DIR = f"{DATA_ROOT}/val"
    TEST_DIR = f"{DATA_ROOT}/test"
    
    # Model Configuration
    NUM_CLASSES = 15  # Confirmed for your 15-class dataset
    IMAGE_SIZE = (224, 224)
    
    # Training Hyperparameters - TWEAKED FOR SOTA
    BATCH_SIZE = 128 
    EPOCHS = 30 
    LEARNING_RATE = 5e-5 # Lower LR for fine-tuning the strong backbone
    
    # NEW: Label Smoothing for better generalization
    LABEL_SMOOTHING = 0.1
    OPTIMIZER_TYPE = 'AdamW' # Using AdamW for better regularization
    
    # Logs and Saves
    MODEL_DIR = "./" # Directory to save history/models (root of the project)
    HISTORY_JSON = "training_history_sota.json"
    BEST_MODEL_SAVE_PATH = "sota_best_model.pth"
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()