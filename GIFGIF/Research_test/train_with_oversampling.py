"""
Training Script with Minority Class Oversampling
=================================================

PROBLEM: contempt (0%) and positive_calm (0%) not learning even with batch_size=16

SOLUTION: Oversample minority classes to balance training
- Duplicate contempt samples (124 → ~500 samples)
- Duplicate positive_calm samples (316 → ~700 samples)
- This ensures model sees them frequently enough to learn

Expected Result: ALL 6 classes learn (no more 0%!)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import json
import random
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_CONFIG = {
    'exp3_6groups': {
        'name': '6 Hierarchical Groups',
        'train_csv': 'train_6_groups.csv',
        'val_csv': 'val_6_groups.csv',
        'test_csv': 'test_6_groups.csv',
        'num_classes': 6,
        'emotion_to_idx': {
            'contempt': 0,
            'negative_intense': 1,
            'negative_subdued': 2,
            'positive_calm': 3,
            'positive_energetic': 4,
            'surprise': 5
        }
    }
}

# ============================================================================
# WEIGHTED SAMPLER FOR OVERSAMPLING
# ============================================================================

def create_weighted_sampler(dataset, emotion_to_idx):
    """
    Create weighted sampler that oversamples minority classes
    
    Strategy:
    - Calculate sample weight for each class (inverse frequency)
    - Assign each sample its class weight
    - WeightedRandomSampler draws samples with these probabilities
    
    Result: Minority classes sampled more frequently!
    """
    
    # Get labels
    labels = [dataset.data.iloc[i]['emotion_label'] for i in range(len(dataset))]
    label_indices = [emotion_to_idx[label] for label in labels]
    
    # Count class frequencies
    class_counts = Counter(label_indices)
    
    # Calculate weights (inverse frequency)
    num_samples = len(dataset)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    
    print(f"\n⚖️  Class Sampling Weights (higher = more frequent sampling):")
    idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
    for cls, weight in sorted(class_weights.items(), key=lambda x: x[1], reverse=True):
        count = class_counts[cls]
        print(f"   {idx_to_emotion[cls]:20s}: {weight:.3f}× (count: {count})")
    
    # Assign weight to each sample
    sample_weights = [class_weights[label_idx] for label_idx in label_indices]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True  # Allow duplicates!
    )
    
    return sampler

# ============================================================================
# DATASET
# ============================================================================

class GIFEmotionDataset(Dataset):
    def __init__(self, csv_path: Path, gif_dir: Path, emotion_to_idx: Dict, transform=None):
        self.data = pd.read_csv(csv_path)
        self.gif_dir = gif_dir
        self.emotion_to_idx = emotion_to_idx
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        gif_id = row['gif_id']
        emotion = row['emotion_label']
        
        # Load middle frame
        gif_path = self.gif_dir / f"{gif_id}.gif"
        try:
            gif = Image.open(gif_path)
            n_frames = 0
            try:
                while True:
                    gif.seek(n_frames)
                    n_frames += 1
            except EOFError:
                pass
            
            middle_idx = n_frames // 2
            gif.seek(middle_idx)
            frame = gif.convert('RGB')
            
        except Exception as e:
            print(f"   ⚠️  Error loading {gif_id}.gif: {e}")
            # Return black frame as fallback
            frame = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            frame = self.transform(frame)
        
        label = self.emotion_to_idx[emotion]
        
        return frame, label

# ============================================================================
# MODEL
# ============================================================================

class SingleFrameEmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SingleFrameEmotionClassifier, self).__init__()
        
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet features
        for param in self.features.parameters():
            param.requires_grad = False
        
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
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, device, emotion_names):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Overall accuracy
    accuracy = 100 * (all_preds == all_labels).sum() / len(all_labels)
    
    # Per-class accuracy
    per_class_acc = []
    for i in range(len(emotion_names)):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=emotion_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return accuracy, np.mean(per_class_acc) * 100, report, cm, macro_f1

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment_with_oversampling(
    experiment_name,
    gif_dir,
    data_dir,
    output_dir='results',
    num_epochs=20,
    batch_size=32,
    learning_rate=0.0001
):
    
    # Setup
    config = EXPERIMENT_CONFIG[experiment_name]
    num_classes = config['num_classes']
    emotion_to_idx = config['emotion_to_idx']
    emotion_names = list(emotion_to_idx.keys())
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print(f"🔬 EXPERIMENT WITH OVERSAMPLING: {experiment_name}")
    print("="*70)
    print(f"Strategy: Weighted sampling to oversample minority classes")
    print(f"Device: {device}")
    print()
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    print("📦 Creating datasets...")
    gif_dir = Path(gif_dir)
    data_dir = Path(data_dir)
    
    train_dataset = GIFEmotionDataset(
        data_dir / config['train_csv'],
        gif_dir,
        emotion_to_idx,
        train_transform
    )
    
    val_dataset = GIFEmotionDataset(
        data_dir / config['val_csv'],
        gif_dir,
        emotion_to_idx,
        val_transform
    )
    
    test_dataset = GIFEmotionDataset(
        data_dir / config['test_csv'],
        gif_dir,
        emotion_to_idx,
        val_transform
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    print(f"   Test: {len(test_dataset)} samples")
    
    # Create weighted sampler for training
    train_sampler = create_weighted_sampler(train_dataset, emotion_to_idx)
    
    # Dataloaders (use sampler for training, shuffle=False required)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,  # Use weighted sampler!
        num_workers=0
    )
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\n📦 Dataloaders created:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    
    # Model
    print("\n🧠 Creating model...")
    model = SingleFrameEmotionClassifier(num_classes).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Training setup (NO class weights - oversampling handles it!)
    print("\n⚙️  Training setup:")
    print("   Strategy: Weighted sampling (oversampling minority classes)")
    print("   Reasoning: Ensures model sees minority classes frequently")
    
    criterion = nn.CrossEntropyLoss()  # No weights needed!
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training
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
    
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 70)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"📊 Epoch {epoch+1}:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path / f'{experiment_name}_best.pth')
            print(f"   ✅ Best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/5)")
        
        if patience_counter >= 5:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(torch.load(output_path / f'{experiment_name}_best.pth'))
    
    # Evaluate
    print(f"\n📊 Evaluating on test set...")
    test_acc, per_class_acc, report, cm, macro_f1 = evaluate(
        model, test_loader, device, emotion_names
    )
    
    print("\n" + "="*70)
    print("📊 TEST RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Per-Class Accuracy: {per_class_acc:.2f}%")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    print("\n📋 Per-Class Metrics:")
    for emotion in emotion_names:
        metrics = report[emotion]
        print(f"   {emotion}:")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall: {metrics['recall']:.4f}")
        print(f"      F1-Score: {metrics['f1-score']:.4f}")
        print(f"      Support: {int(metrics['support'])}")
    
    # Save results
    results = {
        'experiment': experiment_name,
        'test_accuracy': float(test_acc),
        'per_class_accuracy': float(per_class_acc),
        'macro_f1': float(macro_f1),
        'best_val_accuracy': float(best_val_acc),
        'per_class_metrics': {k: v for k, v in report.items() if k in emotion_names},
        'confusion_matrix': cm.tolist(),
        'history': history
    }
    
    # Convert numpy types to Python types for JSON
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {str(k): convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    results = convert_to_python_types(results)
    
    with open(output_path / f'{experiment_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results

# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with minority class oversampling')
    parser.add_argument('--experiment', type=str, default='exp3_6groups')
    parser.add_argument('--gif_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='csvs')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    
    args = parser.parse_args()
    
    run_experiment_with_oversampling(
        experiment_name=args.experiment,
        gif_dir=args.gif_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
