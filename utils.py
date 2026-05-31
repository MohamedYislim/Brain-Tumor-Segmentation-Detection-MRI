import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
from skimage.measure import regionprops, label

class BraTSDataset(Dataset):
    def __init__(self, root_dir, patient_ids, slice_index=75, transform=None):
        """
        Args:
            root_dir: Chemin vers le dossier 'data'
            patient_ids: Liste des noms de dossiers (ex: ['BraTS2021_00001', ...])
            slice_index: L'index de la tranche 2D à extraire (0-154)
        """
        self.root_dir = root_dir
        self.patient_ids = patient_ids
        self.slice_index = slice_index
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def _normalize(self, img):
        """Normalisation Z-score standard"""
        mask = img > 0
        if np.any(mask):
            mean = img[mask].mean()
            std = img[mask].std()
            img[mask] = (img[mask] - mean) / (std + 1e-8)
        return img

    def _get_bbox(self, mask_2d):
        """Génère la boîte englobante [y1, x1, y2, x2] normalisée entre 0 et 1"""
        lbl = label(mask_2d)
        props = regionprops(lbl)
        if len(props) == 0:
            return torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        
        # On prend la plus grande région
        main_prop = max(props, key=lambda x: x.area)
        y1, x1, y2, x2 = main_prop.bbox
        
        # Normalisation par la taille de l'image (240x240 pour BraTS)
        return torch.tensor([y1/240, x1/240, y2/240, x2/240], dtype=torch.float32)

    def __getitem__(self, idx):
        p_id = self.patient_ids[idx]
        p_path = os.path.join(self.root_dir, p_id)

        # 1. Charger FLAIR (Image) et SEG (Masque)
        flair_path = os.path.join(p_path, f"{p_id}_flair.nii.gz")
        seg_path = os.path.join(p_path, f"{p_id}_seg.nii.gz")

        img_vol = nib.load(flair_path).get_fdata()
        mask_vol = nib.load(seg_path).get_fdata()

        # 2. Extraire la tranche 2D
        img_2d = img_vol[:, :, self.slice_index]
        mask_2d = mask_vol[:, :, self.slice_index]

        # 3. Prétraitement
        img_2d = self._normalize(img_2d)
        mask_2d = (mask_2d > 0).astype(np.float32) # Binarisation : Tumeur = 1

        # 4. Générer la Bounding Box pour la détection
        bbox = self._get_bbox(mask_2d)

        # 5. Conversion en Tenseurs PyTorch [C, H, W]
        img_tensor = torch.from_numpy(img_2d).float().unsqueeze(0) 
        mask_tensor = torch.from_numpy(mask_2d).float().unsqueeze(0)

        return img_tensor, mask_tensor, bbox

# --- EXEMPLE D'UTILISATION ---

# Liste de vos dossiers patients
all_patients = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]

# Création du Dataset
dataset = BraTSDataset(root_dir='data', patient_ids=all_patients)

# Création du DataLoader (pour l'entraînement par lots)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test de lecture d'un lot
images, masks, bboxes = next(iter(dataloader))
print(f"Batch images: {images.shape}") # [4, 1, 240, 240]
print(f"Batch masks: {masks.shape}")   # [4, 1, 240, 240]
print(f"Batch bboxes: {bboxes.shape}") # [4, 4]