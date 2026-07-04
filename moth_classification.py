import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models
from sklearn.metrics import classification_report


# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
TRAIN_DIR = "data/train"
VAL_DIR   = "data/valid"
TEST_DIR  = "data/test"

BATCH_SIZE             = 64
IMG_SIZE               = 224
NUM_CLASSES            = 50
FEATURE_EXTRACT_EPOCHS = 89
FINE_TUNE_EPOCHS       = 20
CHECKPOINT_FE          = "moth_feature_extract.pth"
CHECKPOINT_FT          = "moth_fine_tuned.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────
# Moths can appear at any orientation (wall, ceiling, flying)
# so both horizontal and vertical flips are justified here,
# unlike scene classification datasets.
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────
# Dataset & DataLoaders
# ─────────────────────────────────────────
# Dataset already has separate train/valid/test folders
# so no manual splitting needed unlike some other datasets.
train_set = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
val_set   = ImageFolder(root=VAL_DIR,   transform=val_transforms)
test_set  = ImageFolder(root=TEST_DIR,  transform=val_transforms)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
print(f"Classes: {len(train_set.classes)}")


# ─────────────────────────────────────────
# Model — ResNet18 pretrained on ImageNet
# ─────────────────────────────────────────
model = models.resnet18(weights='IMAGENET1K_V1')

# Freeze all pretrained layers, replace classifier head
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(512, NUM_CLASSES)
model     = model.to(device)

criterion = nn.CrossEntropyLoss()


# ─────────────────────────────────────────
# Stage 1: Feature Extraction
# Only the new fc layer trains. All ResNet weights stay frozen.
# ─────────────────────────────────────────
print("\n--- Stage 1: Feature extraction (fc only) ---")

optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
best_val_loss = float("inf")

for epoch in range(FEATURE_EXTRACT_EPOCHS):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            val_loss += criterion(model(x_batch), y_batch).item()

    avg_val = val_loss / len(val_loader)
    scheduler.step(avg_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_FE)

    print(f"Epoch {epoch+1:3d}/{FEATURE_EXTRACT_EPOCHS} | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {avg_val:.4f}")


# ─────────────────────────────────────────
# Stage 2: Fine-tuning
# Unfreeze layer4 (last conv block) and train it with fc
# at a lower learning rate to preserve pretrained knowledge.
# ─────────────────────────────────────────
print("\n--- Stage 2: Fine-tuning (layer4 + fc) ---")

model.load_state_dict(torch.load(CHECKPOINT_FE))

for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = optim.Adam([
    {"params": model.layer4.parameters()},
    {"params": model.fc.parameters()},
], lr=0.0001)
scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
best_val_loss = float("inf")

for epoch in range(FINE_TUNE_EPOCHS):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            val_loss += criterion(model(x_batch), y_batch).item()

    avg_val = val_loss / len(val_loader)
    scheduler.step(avg_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CHECKPOINT_FT)

    print(f"Epoch {epoch+1:3d}/{FINE_TUNE_EPOCHS} | "
          f"Train Loss: {running_loss/len(train_loader):.4f} | "
          f"Val Loss: {avg_val:.4f}")


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT_FT))
model.eval()

all_preds, all_labels = [], []
correct = total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch)

        _, predicted = torch.max(preds, 1)
        total   += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=test_set.classes,
))
