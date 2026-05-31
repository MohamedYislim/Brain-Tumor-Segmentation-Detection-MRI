# train.py
from dataset.dataset import BRATSDataset
from torch.utils.data import DataLoader

# Chemin vers le dossier contenant tous les patients
images_dir = r"C:\Users\Dell\Desktop\M1\ML\vs code file\Projet BraTS\data\raw\BraTS2021_00001"

# Créer le dataset
dataset = BRATSDataset(images_dir)

# Créer le dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Vérifier
print(f"Nombre de coupes extraites: {len(dataset)}")
images, masks = next(iter(dataloader))
print(images.shape, masks.shape)  # ex: torch.Size([4,1,240,240])