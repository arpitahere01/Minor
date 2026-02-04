
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ===============================
# Paths
# ===============================
DATASET_PATH = "/content/drive/MyDrive/Nepali Characters/Data/Split_Dataset"

TRAIN_DIR = f"{DATASET_PATH}/train"
VAL_DIR   = f"{DATASET_PATH}/val"
TEST_DIR  = f"{DATASET_PATH}/test"

# ===============================
# ImageNet normalization (ViT standard)
# ===============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ===============================
# Transforms
# ===============================

# Training: resize + augmentation + normalization
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),                 # converts to [0,1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Validation & Test: resize + normalization only
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# ===============================
# Datasets
# ===============================
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=VAL_DIR, transform=val_test_transform)
test_dataset  = datasets.ImageFolder(root=TEST_DIR, transform=val_test_transform)

# ===============================
# DataLoaders
# ===============================
BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# Sanity checks
# ===============================
print("Number of classes:", len(train_dataset.classes))
print("Classes:", train_dataset.classes)
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# Check one batch
images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
