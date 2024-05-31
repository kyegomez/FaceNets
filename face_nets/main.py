import os
import gdown
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from timm.models.vision_transformer import vit_base_patch16_224

# Constants
ROOT_DIR = './data'
ZIP_FILE = 'img_align_celeba.zip'
EXTRACTED_DIR = os.path.join(ROOT_DIR, 'img_align_celeba')
ATTR_FILE = os.path.join(ROOT_DIR, 'list_attr_celeba.txt')
PARTITION_FILE = os.path.join(ROOT_DIR, 'list_eval_partition.txt')
GOOGLE_DRIVE_ID = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'

# Download CelebA dataset from Google Drive
def download_celeba():
    os.makedirs(ROOT_DIR, exist_ok=True)
    zip_path = os.path.join(ROOT_DIR, ZIP_FILE)
    
    if not os.path.exists(zip_path):
        print(f"Downloading CelebA dataset to {zip_path}...")
        gdown.download(id=GOOGLE_DRIVE_ID, output=zip_path, quiet=False)
    else:
        print(f"{zip_path} already exists, skipping download.")

    # Extract the dataset
    if not os.path.exists(EXTRACTED_DIR):
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ROOT_DIR)
    else:
        print(f"{EXTRACTED_DIR} already exists, skipping extraction.")

# Dataset Preparation
class CelebADataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.dataset = CelebA(root=root, split=split, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label[20]  # Example: Use 'Smiling' attribute for classification

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Model Definition
class ViTFaceRecognition(nn.Module):
    def __init__(self, num_classes=2):
        super(ViTFaceRecognition, self).__init__()
        self.vit = vit_base_patch16_224(pretrained=True)
        self.fc = nn.Linear(self.vit.head.in_features, num_classes)
        self.vit.head = nn.Identity()  # Remove the original head

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(x)
        return x

# Training Loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        epoch_acc = val_running_corrects.double() / len(val_loader.dataset)

        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = model.state_dict()

    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # Download and preprocess the CelebA dataset
    download_celeba()

    # Paths
    root_dir = ROOT_DIR

    # Datasets and Dataloaders
    train_dataset = CelebADataset(root=root_dir, split='train', transform=transform)
    val_dataset = CelebADataset(root=root_dir, split='valid', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize Model
    model = ViTFaceRecognition(num_classes=2)
    model = model.cuda() if torch.cuda.is_available() else model

    # Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25)

    # Save the model
    torch.save(trained_model.state_dict(), 'vit_face_recognition.pth')
    print("Model training complete and saved as vit_face_recognition.pth")
