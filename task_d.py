import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
from tqdm import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision.transforms import functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import vgg19_bn

# Hyper Parameters
BATCH_SIZE = 16
NUM_CLASSES = 5
LEARNING_RATE = 0.00005
APTOS_EPOCHS = 20    # Reduced epochs for APTOS
DEEPDRID_EPOCHS = 50 # More epochs for DeepDRiD fine-tuning

class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, is_aptos=False, test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.is_aptos = is_aptos
        self.test = test
        self.data = self.load_data(ann_file)
        print(f"Loaded {len(self.data)} samples from {'APTOS' if is_aptos else 'DeepDRiD'} dataset")

    def load_data(self, ann_file):
        df = pd.read_csv(ann_file)
        print(f"Read {len(df)} rows from {ann_file}")
        
        if self.is_aptos:
            # For APTOS, use single images
            data = []
            for _, row in df.iterrows():
                file_info = {}
                file_info['img_path'] = os.path.join(self.image_dir, str(row['id_code']) + '.png')
                if not self.test:
                    file_info['dr_level'] = int(row['diagnosis'])
                
                if os.path.exists(file_info['img_path']):
                    data.append(file_info)
                else:
                    print(f"Warning: Missing file {file_info['img_path']}")
        else:
            # DeepDRiD dataset: group by patient and eye side
            df['prefix'] = df['img_path'].str.split('_').str[0]
            df['suffix'] = df['img_path'].str.split('_').str[1].str[0]
            grouped = df.groupby(['prefix', 'suffix'])

            data = []
            for _, group in grouped:
                if len(group) >= 2:  # Ensure we have pairs
                    file_info = {}
                    file_info['img_path1'] = os.path.join(self.image_dir, group.iloc[0]['img_path'])
                    file_info['img_path2'] = os.path.join(self.image_dir, group.iloc[1]['img_path'])
                    if not self.test:
                        file_info['dr_level'] = int(group.iloc[0]['patient_DR_Level'])
                    
                    if os.path.exists(file_info['img_path1']) and os.path.exists(file_info['img_path2']):
                        data.append(file_info)
                    else:
                        print(f"Warning: Missing files for pair {file_info['img_path1']} or {file_info['img_path2']}")

        print(f"Created {len(data)} valid {'samples' if self.is_aptos else 'pairs'}")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        
        if self.is_aptos:
            # Single image for APTOS
            img = Image.open(data['img_path']).convert('RGB')
            if self.transform:
                img = self.transform(img)
            if not self.test:
                label = torch.tensor(data['dr_level'], dtype=torch.int64)
                return [img, img], label
            return [img, img]
        else:
            # Dual images for DeepDRiD
            img1 = Image.open(data['img_path1']).convert('RGB')
            img2 = Image.open(data['img_path2']).convert('RGB')
            
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            
            if not self.test:
                label = torch.tensor(data['dr_level'], dtype=torch.int64)
                return [img1, img2], label
            return [img1, img2]

class DualDRModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        
        # Load two pretrained VGG19_bn backbones
        self.backbone1 = vgg19_bn(pretrained=pretrained)
        self.backbone2 = vgg19_bn(pretrained=pretrained)
        
        # Get the number of features (VGG19 has 4096 features in its last FC layer)
        num_features = 4096
        
        # Remove original classifiers
        self.backbone1.classifier = nn.Sequential(*list(self.backbone1.classifier.children())[:-1])
        self.backbone2.classifier = nn.Sequential(*list(self.backbone2.classifier.children())[:-1])
        
        # More complex classifier with additional regularization
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x1, x2 = x
        feat1 = self.backbone1(x1)
        feat2 = self.backbone2(x2)
        combined = torch.cat((feat1, feat2), dim=1)
        return self.classifier(combined)

    def freeze_backbones(self):
        """Freeze all layers except the classifier"""
        for param in self.backbone1.parameters():
            param.requires_grad = False
        for param in self.backbone2.parameters():
            param.requires_grad = False

    def unfreeze_backbones(self):
        """Unfreeze all layers"""
        for param in self.backbone1.parameters():
            param.requires_grad = True
        for param in self.backbone2.parameters():
            param.requires_grad = True

def plot_training_metrics(history):
    """Plot comprehensive training metrics including loss, accuracy, kappa score, and learning rate."""
    plt.style.use('seaborn-v0_8')
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot Loss
    ax1.plot(history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(history['val_loss'], 'r--', label='Validation Loss')
    ax1.set_title('Model Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Accuracy
    ax2.plot(history['train_acc'], 'g-', label='Training Accuracy')
    ax2.plot(history['val_acc'], 'r--', label='Validation Accuracy')
    ax2.set_title('Model Accuracy Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    ax2.legend()
    
    # Plot Kappa Score
    ax3.plot(history['train_kappa'], 'c-', label='Training Kappa')
    ax3.plot(history['val_kappa'], 'm--', label='Validation Kappa')
    ax3.set_title('Model Kappa Score Over Time')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Kappa Score')
    ax3.grid(True)
    ax3.legend()
    
    # Plot Learning Rate
    ax4.plot(history['learning_rate'], 'y-')
    ax4.set_title('Learning Rate Over Time')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_yscale('log')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_metrics(preds, labels):
    """Compute accuracy and kappa score."""
    return {
        'accuracy': accuracy_score(labels, preds),
        'kappa': cohen_kappa_score(labels, preds, weights='quadratic')
    }

def train_model(model, train_loader, val_loader, device, criterion, optimizer, num_epochs, checkpoint_path):
    best_val_kappa = -1.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_kappa': [], 'val_kappa': [],
        'learning_rate': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with tqdm(total=len(train_loader), desc='Training', unit='batch') as pbar:
            for images, labels in train_loader:
                images = [img.to(device) for img in images]
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.update(1)
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_metrics = compute_metrics(all_preds, all_labels)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = [img.to(device) for img in images]
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics(all_preds, all_labels)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_kappa'].append(train_metrics['kappa'])
        history['val_kappa'].append(val_metrics['kappa'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        print(f'Train Loss: {train_loss:.4f}, Kappa: {train_metrics["kappa"]:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Kappa: {val_metrics["kappa"]:.4f}')

        # Save best model
        if val_metrics['kappa'] > best_val_kappa:
            best_val_kappa = val_metrics['kappa']
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Saved new best model with kappa: {best_val_kappa:.4f}')
    
    # Plot training metrics
    plot_training_metrics(history)
    return best_val_kappa

def evaluate_model(model, loader, device, test_only=False, prediction_path=None):
    model.eval()
    all_preds = []
    all_labels = []
    all_image_ids = []

    with tqdm(total=len(loader), desc='Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(loader):
            if test_only:
                images = data
            else:
                images, labels = data

            # Move images to device
            if not isinstance(images, list):
                images = images.to(device)
            else:
                images = [x.to(device) for x in images]

            # Get predictions
            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

            # Handle predictions based on dataset type
            is_aptos = loader.dataset.is_aptos
            batch_size = len(images[0]) if isinstance(images, list) else len(images)
            batch_start = i * loader.batch_size
            batch_end = batch_start + batch_size

            if is_aptos:
                # For APTOS, use single image path
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(loader.dataset.data[idx]['img_path']) for idx in
                    range(batch_start, batch_end)
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.cpu().numpy())
            else:
                # For DeepDRiD, use dual image paths
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(batch_start, batch_end)
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.cpu().numpy())

            pbar.update(1)

    # Save predictions to CSV if in test mode
    if test_only and prediction_path:
        predictions_df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        predictions_df.to_csv(prediction_path, index=False)
        print(f'Saved predictions to {os.path.abspath(prediction_path)}')
        return None
    else:
        return cohen_kappa_score(all_labels, all_preds, weights='quadratic')

class PreprocessTransform:
    """Custom preprocessing transforms"""
    def __init__(self, technique='default'):
        self.technique = technique
    
    def ben_graham(self, image):
        """Ben Graham's preprocessing technique"""
        image = np.array(image)
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0), 10), -4, 128)
        return Image.fromarray(image)
    
    def circle_crop(self, image):
        """Circle crop the image"""
        image = np.array(image)
        height, width = image.shape[:2]
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width//2, height//2), min(width, height)//2, (255,255,255), -1)
        image = cv2.bitwise_and(image, image, mask=mask)
        return Image.fromarray(image)
    
    def clahe(self, image):
        """Apply CLAHE"""
        image = np.array(image)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(image)

    def __call__(self, image):
        if self.technique == 'ben_graham':
            return self.ben_graham(image)
        elif self.technique == 'circle_crop':
            return self.circle_crop(image)
        elif self.technique == 'clahe':
            return self.clahe(image)
        return image

def create_model_transforms(preprocessing='default'):
    """Create transforms with different preprocessing techniques"""
    train_transform = transforms.Compose([
        PreprocessTransform(technique=preprocessing),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        PreprocessTransform(technique=preprocessing),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

class EnsembleModel(nn.Module):
    def __init__(self, model_configs):
        super().__init__()
        self.models = nn.ModuleList()
        self.weights = nn.Parameter(torch.ones(len(model_configs)))
        
        for config in model_configs:
            model = DualDRModel(
                num_classes=NUM_CLASSES,
                pretrained=True,
                backbone=config['backbone']
            )
            if config['weights_path']:
                model.load_state_dict(torch.load(config['weights_path']))
            self.models.append(model)
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average ensemble
        weighted_outputs = torch.zeros_like(outputs[0])
        weights = F.softmax(self.weights, dim=0)
        for w, output in zip(weights, outputs):
            weighted_outputs += w * output
        
        return weighted_outputs

def main():
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create transforms with CLAHE preprocessing
    train_transform, val_transform = create_model_transforms(preprocessing='clahe')
    print("Using CLAHE preprocessing technique with VGG19_bn")

    # Stage 1: Train on APTOS dataset
    print("Stage 1: Training on APTOS dataset...")
    aptos_train_dataset = RetinopathyDataset(
        '/home/t/Desktop/koulu/deep_learning/project/APTOS2019/train_1.csv', 
        '/home/t/Desktop/koulu/deep_learning/project/APTOS2019/train_images/train_images', 
        transform=train_transform,
        is_aptos=True
    )
    aptos_val_dataset = RetinopathyDataset(
        '/home/t/Desktop/koulu/deep_learning/project/APTOS2019/valid.csv', 
        '/home/t/Desktop/koulu/deep_learning/project/APTOS2019/val_images/val_images', 
        transform=val_transform,
        is_aptos=True
    )

    aptos_train_loader = DataLoader(aptos_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    aptos_val_loader = DataLoader(aptos_val_dataset, batch_size=BATCH_SIZE)

    # Initialize model for Stage 1
    model = DualDRModel(num_classes=NUM_CLASSES, pretrained=True)
    model.unfreeze_backbones()  # Unfreeze all layers for APTOS training
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train on APTOS
    best_aptos_kappa = train_model(
        model, aptos_train_loader, aptos_val_loader, device, criterion, optimizer,
        APTOS_EPOCHS, 'aptos_model_dual_vgg19bn_clahe.pth'  # Updated filename
    )
    # Rename the plot for APTOS training
    os.rename('training_metrics.png', 'aptos_training_metrics_vgg19bn_clahe.png')
    print(f"Stage 1 completed. Best APTOS validation kappa: {best_aptos_kappa:.4f}")

    # Stage 2: Fine-tune on DeepDRiD
    print("\nStage 2: Fine-tuning on DeepDRiD dataset...")
    model.load_state_dict(torch.load('aptos_model_dual_vgg19bn_clahe.pth'))
    model.freeze_backbones()  # Freeze all layers except classifier

    # Load DeepDRiD datasets
    deepdrid_train_dataset = RetinopathyDataset(
        './DeepDRiD/train.csv', 
        './DeepDRiD/train/', 
        transform=train_transform
    )
    deepdrid_val_dataset = RetinopathyDataset(
        './DeepDRiD/val.csv', 
        './DeepDRiD/val/', 
        transform=val_transform
    )

    deepdrid_train_loader = DataLoader(deepdrid_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    deepdrid_val_loader = DataLoader(deepdrid_val_dataset, batch_size=BATCH_SIZE)

    # Only optimize classifier parameters
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # Fine-tune on DeepDRiD with more epochs
    best_deepdrid_kappa = train_model(
        model, deepdrid_train_loader, deepdrid_val_loader, device, criterion, optimizer,
        DEEPDRID_EPOCHS, 'final_model_dual_vgg19bn_clahe.pth'  # Updated filename
    )
    # Rename the plot for DeepDRiD training
    os.rename('training_metrics.png', 'deepdrid_training_metrics_vgg19bn_clahe.png')
    print(f"Stage 2 completed. Best DeepDRiD validation kappa: {best_deepdrid_kappa:.4f}")

    # After Stage 2, load the best model and generate predictions for test set
    print("\nGenerating predictions for DeepDRiD test set...")
    model.load_state_dict(torch.load('final_model_dual_vgg19bn_clahe.pth'))
    model.eval()

    # Load DeepDRiD test dataset
    test_dataset = RetinopathyDataset(
        './DeepDRiD/test.csv', 
        './DeepDRiD/test/', 
        transform=val_transform,
        is_aptos=False,
        test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Generate and save predictions
    evaluate_model(
        model, 
        test_loader, 
        device, 
        test_only=True, 
        prediction_path='test_predictions_vgg19bn_clahe.csv'  # Updated filename
    )

if __name__ == '__main__':
    main()
