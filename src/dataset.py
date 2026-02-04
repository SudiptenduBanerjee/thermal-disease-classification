import cv2
import numpy as np
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from src.config import Config # Note the 'src.' prefix

class ThermalTransform:
    def __call__(self, img):
        img_np = np.array(img)
        if len(img_np.shape) == 3:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_np
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(enhanced_rgb)

def get_loaders():
    # Training Augmentation
    train_transform = transforms.Compose([
        ThermalTransform(),
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Validation/Test Normalization
    eval_transform = transforms.Compose([
        ThermalTransform(),
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Explicitly creating datasets from the respected paths
    train_ds = datasets.ImageFolder(Config.TRAIN_DIR, transform=train_transform)
    val_ds = datasets.ImageFolder(Config.VAL_DIR, transform=eval_transform)
    test_ds = datasets.ImageFolder(Config.TEST_DIR, transform=eval_transform)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes