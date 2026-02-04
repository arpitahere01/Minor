# Nepali Handwritten Word Recognition (NHWR) Project

This repo contains preprocessing and DataLoader code for Nepali handwritten word recognition using a Vision Transformer (ViT).

## Folder Structure
NHWR-MINOR/
│
├─ dataset_loader.py # Preprocessing & DataLoader
├─ Split_Dataset/ # Train/Val/Test folders with class subfolders ( INSIDE SHARED DRIVE)
└─ Processed_Dataset/ # Optional: saved tensors
---

## Features of `dataset_loader.py`

- Resizes images to 224×224 (ViT input size)  
- Converts grayscale → RGB automatically  
- Normalizes using ImageNet mean & std  
- Applies augmentation only on training data  
- Creates train, val, test PyTorch DataLoaders  
- Labels assigned automatically from folder names  

---

## Usage
bash
1. **Clone the repo**  
bash
git clone <your-repo-url>
cd Minor

2.Install dependencies

pip install torch torchvision tqdm

3.Run the loader

python dataset_loader.py

//Prints number of classes, samples, and one batch shape.
//DataLoaders are ready for training.
//Training teammates can directly use the DataLoaders; augmentation is included in the training set automatically.
