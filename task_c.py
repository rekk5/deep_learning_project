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
import torch.nn.functional as F

# Hyper Parameters
BATCH_SIZE = 8
NUM_CLASSES = 5
LEARNING_RATE = 0.00005
APTOS_EPOCHS = 20    # Reduced epochs for APTOS
DEEPDRID_EPOCHS = 50 # More epochs for DeepDRiD fine-tuning

class RetinopathyDataset(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, test=False):
        self.image_dir = image_dir
        self.transform = transform
        self.test = test
        self.data = self.load_data(ann_file)
        print(f"Loaded {len(self.data)} samples from DeepDRiD dataset")

    def load_data(self, ann_file):
        df = pd.read_csv(ann_file)
        print(f"Read {len(df)} rows from {ann_file}")
        
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

        print(f"Created {len(data)} valid pairs")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        
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

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Reshape Q, K, V
        query = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, H * W)
        value = self.value(x).view(batch_size, -1, H * W)
        
        # Attention map
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Output
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class DualDRModel(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        
        # Load two pretrained EfficientNetB3 backbones
        if pretrained:
            self.backbone1 = models.efficientnet_b3(weights='IMAGENET1K_V1')
            self.backbone2 = models.efficientnet_b3(weights='IMAGENET1K_V1')
        else:
            self.backbone1 = models.efficientnet_b3()
            self.backbone2 = models.efficientnet_b3()
        
        # Get the number of features (1536 for EfficientNetB3)
        num_features = 1536
        
        # Add all three attention modules
        self.channel_attention1 = ChannelAttention(num_features)
        self.channel_attention2 = ChannelAttention(num_features)
        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.self_attention1 = SelfAttention(num_features)
        self.self_attention2 = SelfAttention(num_features)
        
        # Remove original classifiers
        self.backbone1.classifier = nn.Identity()
        self.backbone2.classifier = nn.Identity()
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
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
        
        # Get features from both backbones
        feat1 = self.backbone1.features(x1)
        feat2 = self.backbone2.features(x2)
        
        # Apply channel attention
        ca1 = self.channel_attention1(feat1)
        ca2 = self.channel_attention2(feat2)
        feat1 = feat1 * ca1
        feat2 = feat2 * ca2
        
        # Apply spatial attention
        sa1 = self.spatial_attention1(feat1)
        sa2 = self.spatial_attention2(feat2)
        feat1 = feat1 * sa1
        feat2 = feat2 * sa2
        
        # Apply self-attention
        feat1 = self.self_attention1(feat1)
        feat2 = self.self_attention2(feat2)
        
        # Global average pooling
        feat1 = self.gap(feat1).flatten(1)
        feat2 = self.gap(feat2).flatten(1)
        
        # Combine features and classify
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

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x[0].size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = [lam * x1 + (1 - lam) * x1[index] for x1 in x]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(model, train_loader, val_loader, device, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    best_val_kappa = -1.0
    history = {
        'epoch': [],
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

                # Apply mixup augmentation
                if epoch > 5:  # Start mixup after 5 epochs
                    images, labels_a, labels_b, lam = mixup_data(images, labels)
                    
                optimizer.zero_grad()
                outputs = model(images)
                
                # Use mixup loss if applicable
                if epoch > 5:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
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
        
        # Update learning rate based on validation kappa
        scheduler.step(val_metrics['kappa'])
        
        # Update history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_kappa'].append(train_metrics['kappa'])
        history['val_kappa'].append(val_metrics['kappa'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Save history to CSV after each epoch
        pd.DataFrame(history).to_csv('training_history.csv', index=False)
        
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

            # Handle predictions for DeepDRiD (always dual images)
            batch_size = len(images[0])
            batch_start = i * loader.batch_size
            batch_end = batch_start + batch_size

            # Dual images case
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
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
        return None
    else:
        return cohen_kappa_score(all_labels, all_preds, weights='quadratic')

def compute_final_metrics(model, val_loader, device):
    """Compute final metrics on validation set"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = [img.to(device) for img in images]
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    print("\nFinal Model Performance:")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Kappa Score: {kappa:.4f}")
    
    return accuracy, kappa

def main():
    # Data transforms for EfficientNetB3
    transform = transforms.Compose([
        transforms.Resize((300, 300)),  # EfficientNetB3 preferred size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load DeepDRiD datasets
    train_dataset = RetinopathyDataset(
        './DeepDRiD/train.csv', 
        './DeepDRiD/train/', 
        transform=transform
    )
    val_dataset = RetinopathyDataset(
        './DeepDRiD/val.csv', 
        './DeepDRiD/val/', 
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = DualDRModel(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # Train model with scheduler
    best_kappa = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        checkpoint_path='attention_model.pth'
    )
    print(f"Training completed. Best validation kappa: {best_kappa:.4f}")

    # After training, load best model and generate predictions
    print("\nGenerating predictions for test set...")
    model.load_state_dict(torch.load('attention_model.pth', weights_only=True))
    model.eval()

    # Load test dataset
    test_dataset = RetinopathyDataset(
        './DeepDRiD/test.csv', 
        './DeepDRiD/test/', 
        transform=val_transform,  # Use validation transform for test
        test=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Generate and save predictions
    evaluate_model(
        model, 
        test_loader, 
        device, 
        test_only=True, 
        prediction_path='test_predictions.csv'
    )

    # Add to main after training:
    model.load_state_dict(torch.load('attention_model.pth'))
    compute_final_metrics(model, val_loader, device)

if __name__ == '__main__':
    main()
