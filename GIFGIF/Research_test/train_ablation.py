"""
ABLATION STUDY: Hierarchical Emotion Taxonomy Evaluation
==========================================================

Runs 4 systematic experiments to validate hierarchical grouping:
1. 17 original emotions (baseline)
2. 11 filtered emotions (remove low-sample classes)
3. 6 hierarchical groups (your main contribution)
4. 3 simple groups (alternative hierarchy)

Usage:
    python train_ablation.py --experiment exp1_17emotions
    python train_ablation.py --experiment exp2_11emotions
    python train_ablation.py --experiment exp3_6groups
    python train_ablation.py --experiment exp4_3groups
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report
)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class SingleFrameEmotionClassifier(nn.Module):
    """Single-frame ResNet50 baseline (proven to work at 35%)"""
    
    def __init__(self, num_classes):
        super(SingleFrameEmotionClassifier, self).__init__()
        
        # ResNet50 backbone (frozen for stability)
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Classifier head (trainable)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output

# ============================================================================
# DATASET
# ============================================================================

class GIFSingleFrameDataset(Dataset):
    """Dataset that extracts middle frame from GIFs"""
    
    def __init__(self, csv_file, gif_dir, transform, emotion_to_idx, verbose=False):
        self.df = pd.read_csv(csv_file)
        self.gif_dir = Path(gif_dir)
        self.transform = transform
        self.emotion_to_idx = emotion_to_idx
        self.verbose = verbose
        self.error_count = 0
        
        # Filter to only include emotions in our mapping
        valid_emotions = set(emotion_to_idx.keys())
        initial_len = len(self.df)
        self.df = self.df[self.df['emotion_label'].isin(valid_emotions)].reset_index(drop=True)
        
        if self.verbose:
            print(f"   Filtered: {initial_len} → {len(self.df)} samples")
            print(f"   Emotion distribution:")
            for emotion, count in self.df['emotion_label'].value_counts().items():
                print(f"      {emotion}: {count}")
    
    def extract_middle_frame(self, gif_path):
        """Extract middle frame from GIF with safety measures"""
        try:
            with Image.open(gif_path) as gif:
                # Count frames (with safety limit)
                n_frames = 0
                max_frames = 200
                
                try:
                    while n_frames < max_frames:
                        gif.seek(n_frames)
                        n_frames += 1
                except EOFError:
                    pass
                
                if n_frames == 0:
                    raise ValueError("Empty GIF")
                
                # Extract middle frame
                middle_idx = n_frames // 2
                gif.seek(middle_idx)
                frame = gif.convert('RGB')
                
                # Validate frame size
                if frame.size[0] < 10 or frame.size[1] < 10:
                    raise ValueError(f"Frame too small: {frame.size}")
                
                if self.transform:
                    frame = self.transform(frame)
                
                return frame
                
        except Exception as e:
            self.error_count += 1
            if self.error_count <= 5 and self.verbose:
                print(f"   ⚠️  Error loading {gif_path.name}: {str(e)[:50]}")
            
            # Return black frame on error
            if self.transform:
                black_pil = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(black_pil)
            else:
                return torch.zeros(3, 224, 224)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gif_id = row['gif_id']
        gif_path = self.gif_dir / f"{gif_id}.gif"
        
        # Extract frame
        frame = self.extract_middle_frame(gif_path)
        
        # Get label
        emotion = row['emotion_label']
        label = self.emotion_to_idx[emotion]
        
        return frame, label

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Validation"):
            frames, labels = frames.to(device), labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def evaluate_detailed(model, dataloader, device, emotion_names):
    """Detailed evaluation with metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Testing"):
            frames = frames.to(device)
            outputs = model(frames)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)  # Handle division by zero
    
    # Class imbalance ratio
    class_counts = Counter(all_labels)
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    results = {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc.mean() * 100,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class_metrics': {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'per_class_acc': per_class_acc.tolist()
        },
        'confusion_matrix': cm.tolist(),
        'class_distribution': dict(class_counts),
        'imbalance_ratio': imbalance_ratio
    }
    
    return results, all_preds, all_labels

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, emotion_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=emotion_names,
        yticklabels=emotion_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved confusion matrix to {save_path}")

def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Saved training curves to {save_path}")

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_experiment(
    experiment_name,
    num_classes,
    emotion_to_idx,
    train_csv,
    val_csv,
    test_csv,
    gif_dir,
    output_dir='ablation_results',
    num_epochs=20,
    batch_size=32,
    learning_rate=0.0001,
    patience=5
):
    """Run one complete ablation experiment"""
    
    print("\n" + "="*70)
    print(f"🔬 EXPERIMENT: {experiment_name}")
    print("="*70)
    print(f"Number of classes: {num_classes}")
    print(f"Emotions: {list(emotion_to_idx.keys())}")
    print(f"Training CSV: {train_csv}")
    
    # Create output directory
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("\n📦 Creating datasets...")
    train_dataset = GIFSingleFrameDataset(
        train_csv, gif_dir, train_transform, emotion_to_idx, verbose=True
    )
    val_dataset = GIFSingleFrameDataset(
        val_csv, gif_dir, val_test_transform, emotion_to_idx, verbose=True
    )
    test_dataset = GIFSingleFrameDataset(
        test_csv, gif_dir, val_test_transform, emotion_to_idx, verbose=True
    )
    
    # DataLoaders
    print("\n📦 Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Model
    print("\n🧠 Creating model...")
    model = SingleFrameEmotionClassifier(num_classes).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Total parameters: {total_params:,}")
    
    # Compute class weights (matching your successful previous model!)
    print("\n⚖️  Computing class weights...")
    from sklearn.utils.class_weight import compute_class_weight
    
    # Extract all labels from training set
    emotion_labels = []
    for _, label in train_dataset:
        emotion_labels.append(label)
    emotion_labels = np.array(emotion_labels)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=emotion_labels
    )
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"   Class weights computed:")
    emotion_names = list(emotion_to_idx.keys())
    for idx, weight in enumerate(class_weights):
        print(f"      {emotion_names[idx]:20s}: {weight:.3f}")
    
    # Training setup (WITH CLASS WEIGHTS!)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training loop
    print(f"\n🚀 Starting training ({num_epochs} epochs)...")
    print("="*70)
    
    best_val_acc = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1}:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                model.state_dict(), 
                exp_dir / f'{experiment_name}_best.pth'
            )
            print(f"   ✅ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for testing
    print("\n📊 Evaluating on test set...")
    model.load_state_dict(torch.load(exp_dir / f'{experiment_name}_best.pth'))
    
    emotion_names = list(emotion_to_idx.keys())
    test_results, test_preds, test_labels = evaluate_detailed(
        model, test_loader, device, emotion_names
    )
    
    # Print test results
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print(f"Per-Class Accuracy: {test_results['per_class_accuracy']:.2f}%")
    print(f"Macro F1-Score: {test_results['macro_f1']:.4f}")
    print(f"Class Imbalance Ratio: {test_results['imbalance_ratio']:.2f}:1")
    
    print("\n📋 Per-Class Metrics:")
    for i, emotion in enumerate(emotion_names):
        print(f"   {emotion}:")
        print(f"      Precision: {test_results['per_class_metrics']['precision'][i]:.4f}")
        print(f"      Recall: {test_results['per_class_metrics']['recall'][i]:.4f}")
        print(f"      F1-Score: {test_results['per_class_metrics']['f1'][i]:.4f}")
        print(f"      Support: {test_results['per_class_metrics']['support'][i]}")
    
    # Save results
    final_results = {
        'experiment_name': experiment_name,
        'num_classes': num_classes,
        'emotion_to_idx': emotion_to_idx,
        'best_val_acc': best_val_acc,
        'test_results': test_results,
        'training_history': history,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'patience': patience
        }
    }
    
    with open(exp_dir / f'{experiment_name}_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to {exp_dir}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    
    plot_confusion_matrix(
        np.array(test_results['confusion_matrix']),
        emotion_names,
        exp_dir / f'{experiment_name}_confusion_matrix.png'
    )
    
    plot_training_curves(
        history,
        exp_dir / f'{experiment_name}_training_curves.png'
    )
    
    print("\n✅ Experiment complete!")
    
    return final_results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation experiment')
    parser.add_argument('--experiment', type=str, required=True,
                       choices=['exp1_17emotions', 'exp2_11emotions', 
                               'exp3_6groups', 'exp4_3groups'],
                       help='Which experiment to run')
    parser.add_argument('--gif_dir', type=str, required=True,
                       help='Path to GIF directory')
    parser.add_argument('--data_dir', type=str, default='.',
                       help='Path to CSV files')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Import experiment configurations
    from experiment_configs import get_experiment_config
    
    config = get_experiment_config(args.experiment, args.data_dir)
    
    results = run_experiment(
        experiment_name=args.experiment,
        num_classes=config['num_classes'],
        emotion_to_idx=config['emotion_to_idx'],
        train_csv=config['train_csv'],
        val_csv=config['val_csv'],
        test_csv=config['test_csv'],
        gif_dir=args.gif_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )