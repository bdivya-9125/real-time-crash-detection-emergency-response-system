# ===============================
# 1️⃣ Install packages
# ===============================
!pip install torch torchvision opencv-python-headless kaggle timm

# ===============================
# 2️⃣ Kaggle API setup
# ===============================
from google.colab import files
files.upload()  # upload kaggle.json

!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# ===============================
# 3️⃣ Download dataset
# ===============================
!kaggle datasets download -d ckay16/accident-detection-from-cctv-footage
!unzip -q accident-detection-from-cctv-footage.zip -d dataset

# ===============================
# 4️⃣ Imports
# ===============================
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2
from google.colab.patches import cv2_imshow
import numpy as np, random

# ===============================
# 5️⃣ Reproducibility
# ===============================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ===============================
# 6️⃣ Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# 7️⃣ Transforms
# ===============================
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ===============================
# 8️⃣ Dataset + Loader
# ===============================
train_dataset = datasets.ImageFolder('dataset/data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('dataset/data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

classes = train_dataset.classes
print("Classes:", classes)

# ===============================
# 9️⃣ Model (ResNet50 fine-tuned)
# ===============================
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = True

# ===============================
# 🔟 Loss, Optimizer, Scheduler
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# ===============================
# 11️⃣ Training Loop with AMP
# ===============================
num_epochs = 35
best_acc = 0

scaler = torch.amp.GradScaler("cuda")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2f}%")

    scheduler.step(val_acc)

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_accident_model.pth')

print(f"✅ Best Validation Accuracy: {best_acc:.2f}%")
