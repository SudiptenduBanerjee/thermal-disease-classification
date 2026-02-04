import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.config import Config

class ThermalTransform:
    """
    Applies CLAHE (Contrast Limiting) and Inferno Colormap 
    to make disease spots pop out in thermal images.
    """
    def __call__(self, img):
        img_np = np.array(img)
        # Convert to grayscale if not already
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
            
        # Apply CLAHE to sharpen thermal gradients
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Pseudo-color (Inferno is better for ML than Jet)
        enhanced_color = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        
        # Convert back to RGB for the model
        enhanced_rgb = cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2RGB)
        return Image.fromarray(enhanced_rgb)

def get_loaders():
    # Training: Augmentation + Thermal Enhance
    train_transform = transforms.Compose([
        ThermalTransform(),
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), # Slight rotation for robustness
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation/Test: Just Thermal Enhance + Resize
    eval_transform = transforms.Compose([
        ThermalTransform(),
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(Config.VAL_DIR, transform=eval_transform)
    test_ds = datasets.ImageFolder(Config.TEST_DIR, transform=eval_transform)

    # num_workers=8 to feed the 4 GPUs fast enough
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes