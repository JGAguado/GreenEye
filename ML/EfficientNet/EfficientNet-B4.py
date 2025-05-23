import torch
import os
import pandas as pd
import json
from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report

# ------------ DATASET CLASS ------------
class PlantNetDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform
        
        # Load class mapping
        mapping_dir = r"C:\Users\jeppu\Desktop\BTH\ML\Repo\GreenEye\backend\models"
        
        # Create class mapping
        with open(os.path.join(mapping_dir, "class_idx_to_species_id.json"), "r") as f:
            raw_idx2id = json.load(f)
            
        with open(os.path.join(mapping_dir, "plantnet300K_species_id_2_name.json"), "r") as f:
            raw_id2name = json.load(f)
            
        # Build the proper mapping
        self.idx2species = {}
        for idx_str, species_id in raw_idx2id.items():
            idx = int(idx_str)
            name = raw_id2name.get(species_id, "Unknown")
            self.idx2species[idx] = name
            
        # Invert for dataset usage
        self.class_to_idx = {name: idx for idx, name in self.idx2species.items()}
        self.classes = [self.idx2species[i] for i in range(len(self.idx2species))]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["image_path"]
        species = self.data.iloc[idx]["species"]
    
        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Get class index safely
        if species in self.class_to_idx:
            label = self.class_to_idx[species]
        else:
            # Handle unknown species (use a default index or skip)
            print(f"Warning: Unknown species {species}")
            label = 0  # Default to first class
        
        return image, torch.tensor(label)

# ------------ DATA PREPARATION ------------
def prepare_plantnet_data(plantnet_dir=None):
    """Prepare PlantNet-300K data for fine-tuning"""
    data = []
    
    # Set the correct path to your PlantNet-300K dataset
    if plantnet_dir is None:
        plantnet_dir = Path(r"C:\Users\jeppu\Desktop\BTH\ML\Idefics3\Dataset\plantnet_300K")
    
    # Load species ID to name mapping
    with open(plantnet_dir / "plantnet300K_species_id_2_name.json", "r") as f:
        species_id_to_name = json.load(f)
    
    # Process each split (train, val, test)
    for split in ["train", "val", "test"]:
        split_path = plantnet_dir / "images" / split
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist")
            continue
            
        # Process each species folder (named by ID)
        for species_id in os.listdir(split_path):
            species_path = split_path / species_id
            if not species_path.is_dir():
                continue
                
            # Get species name from the ID
            species_name = species_id_to_name.get(species_id, f"Unknown-{species_id}")
            
            # Process each image in the species folder
            for img_file in os.listdir(species_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = str(species_path / img_file)
                    data.append({
                        "image_path": img_path,
                        "species": species_name,
                        "split": split
                    })
    
    print(f"Loaded {len(data)} images from {plantnet_dir}")
    return pd.DataFrame(data)

# ------------ MODEL SETUP ------------
def create_model(num_classes):
    # Load pretrained EfficientNet-B4
    model = models.efficientnet_b4(pretrained=True)
    
    # Replace the classifier with a new one
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# ------------ TRAINING FUNCTION ------------
def train_model(model, train_loader, val_loader, device, num_epochs=15):
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in tqdm(train_loader, desc="Training"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track stats
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track stats
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(
            val_targets, 
            val_preds,
            target_names=[val_loader.dataset.classes[i] for i in range(len(val_loader.dataset.classes))],
            zero_division=0
        ))
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), "efficientnet_b4_plantnet.pth")
            print(f"Model saved with accuracy: {best_accuracy:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    return model

# ------------ INFERENCE FUNCTION ------------
def run_inference(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds,
        target_names=class_names,
        zero_division=0
    ))

# ------------ MAIN EXECUTION ------------
def main():

    # Check if GPU is available
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0))
        print("CUDA current device:", torch.cuda.current_device())

    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.RandomCrop((300, 300)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # Added saturation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print("Loading PlantNet dataset...")
    data_df = prepare_plantnet_data()
    
    # Split into train/val/test
    train_df = data_df[data_df["split"] == "train"]
    val_df = data_df[data_df["split"] == "val"]
    test_df = data_df[data_df["split"] == "test"]
    
    # For testing with small subset
    # Uncomment the following lines to limit the dataset size for quick testing
    #train_size = 100  # Adjust as needed
    #val_size = 20     # Adjust as needed
    #train_df = train_df.head(min(len(train_df), train_size))
    #val_df = val_df.head(min(len(val_df), val_size))
    
    # Create datasets
    train_dataset = PlantNetDataset(train_df, transform=train_transform)
    val_dataset = PlantNetDataset(val_df, transform=val_transform)
    test_dataset = PlantNetDataset(test_df, transform=val_transform)
    
    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training on {len(train_dataset)} images, validating on {len(val_dataset)} images")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Create and train model
    model = create_model(num_classes=len(train_dataset.classes))
    model = model.to(device)

    # Force CUDA initialization
    if torch.cuda.is_available():
        torch.cuda.init()
        # Create a small tensor on GPU to verify it works
        dummy_input = torch.ones(1, 3, 224, 224).to(device)
        _ = model(dummy_input)  # Forward pass to initialize CUDA context
        print("Model successfully using GPU")
    
    # Choose mode: "train", "inference" or "continue_training"
    mode = "inference"
    
    if mode == "train":
        print("Starting training...")
        model = train_model(model, train_loader, val_loader, device, num_epochs=15)
        print("Training complete!")
    
    elif mode == "inference":
        print("Loading best model...")
        model.load_state_dict(torch.load("efficientnet_b4_plantnet.pth"))
        run_inference(model, test_loader, device, train_dataset.classes)

    elif mode == "continue_training":
        print("Loading existing model for continued training...")
        model.load_state_dict(torch.load("efficientnet_b4_plantnet.pth"))
        print("Starting additional training...")
        model = train_model(model, train_loader, val_loader, device, num_epochs=10)
        print("Additional training complete!")

if __name__ == "__main__":
    main()