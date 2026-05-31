import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os

class BRATSDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Charger en niveaux de gris (L)
        image = np.array(Image.open(self.image_paths[idx]), dtype=np.float32) / 255.0
        mask = np.array(Image.open(self.mask_paths[idx]), dtype=np.float32) / 255.0
        
        # Ajout de la dimension canal (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
            # Attention : le masque doit subir les mêmes transformations géométriques
            
        return image, mask