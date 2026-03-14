# 🚨 EMERGENCY FIX FOR OVERFITTING

## Current Status:
- Train: 73% ✅ (Too good - memorizing!)
- Val: 29% ❌ (Getting worse - not learning!)
- Gap: 44% 🚨 (SEVERE overfitting!)

## Problem: Model is memorizing training data, not learning patterns!

---

## ✅ FIX #1: REDUCE LEARNING RATE (Most Important!)

### In Cell 9, change:
```python
# OLD (too high):
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# NEW (10x smaller):
optimizer = optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-3)
#                                           ^^^^^^^^  ^^^^^^^^^
#                                           10x lower  10x higher
```

**Why this helps:**
- Smaller learning rate = smaller weight updates = less memorization
- Higher weight decay = more regularization = better generalization

---

## ✅ FIX #2: INCREASE DROPOUT (Add More Regularization)

### In Cell 4, change:
```python
# OLD (not enough):
class EmotionConvGRU(nn.Module):
    def __init__(self, num_classes=6, num_frames=3, 
                 hidden_dim_1=512, hidden_dim_2=256, dropout=0.5):  # ← Only 0.5
        ...
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),  # Only one dropout layer
            nn.Linear(hidden_dim_2, num_classes)
        )

# NEW (much more regularization):
class EmotionConvGRU(nn.Module):
    def __init__(self, num_classes=6, num_frames=3, 
                 hidden_dim_1=512, hidden_dim_2=256, dropout=0.7):  # ← Higher!
        ...
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.7),           # ← Higher dropout
            nn.Linear(hidden_dim_2, 128),  # ← Add intermediate layer
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),           # ← Another dropout
            nn.Linear(128, num_classes)
        )
```

---

## ✅ FIX #3: ADD EARLY STOPPING ON VALIDATION LOSS (Not Accuracy)

### In Cell 10, change:
```python
# OLD (tracking accuracy):
if val_acc > best_val_acc:
    best_val_acc = val_acc
    patience_counter = 0
    # Save model
else:
    patience_counter += 1

# NEW (tracking loss - more stable):
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    # Save model
else:
    patience_counter += 1
```

---

## ✅ FIX #4: REDUCE MODEL CAPACITY (Smaller Hidden Dims)

### In Cell 9, change:
```python
# OLD (too big for 4k samples):
model = EmotionConvGRU(
    num_classes=Config.NUM_CLASSES,
    num_frames=Config.NUM_FRAMES,
    hidden_dim_1=512,  # ← Too big
    hidden_dim_2=256   # ← Too big
).to(device)

# NEW (smaller, less overfitting):
model = EmotionConvGRU(
    num_classes=Config.NUM_CLASSES,
    num_frames=Config.NUM_FRAMES,
    hidden_dim_1=256,  # ← Half size
    hidden_dim_2=128   # ← Half size
).to(device)
```

---

## ✅ FIX #5: FREEZE RESNET LAYERS (Don't Fine-Tune Early)

### Add to Cell 4, in EmotionConvGRU.__init__:
```python
# After creating self.features:
resnet = models.resnet50(pretrained=True)
self.features = nn.Sequential(*list(resnet.children())[:-2])

# NEW: Freeze ResNet weights (don't train them yet)
for param in self.features.parameters():
    param.requires_grad = False
```

**Why this helps:**
- ResNet50 is already pretrained on ImageNet
- Don't need to fine-tune it on only 4k samples
- Prevents overfitting to training GIFs

---

## 🎯 COMPLETE UPDATED CONFIGURATION

### Cell 2 (Config):
```python
class Config:
    # ... (keep everything same except):
    
    LEARNING_RATE = 0.00001  # ← Changed from 0.0001
    WEIGHT_DECAY = 1e-3      # ← Changed from 1e-4
    DROPOUT = 0.7            # ← New parameter
    HIDDEN_DIM_1 = 256       # ← Changed from 512
    HIDDEN_DIM_2 = 128       # ← Changed from 256
```

### Cell 4 (Model):
```python
class EmotionConvGRU(nn.Module):
    def __init__(self, num_classes=6, num_frames=3, 
                 hidden_dim_1=256, hidden_dim_2=128, dropout=0.7):
        super(EmotionConvGRU, self).__init__()
        
        self.num_frames = num_frames
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        
        # ResNet50 feature extractor (FROZEN)
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze ResNet weights
        for param in self.features.parameters():
            param.requires_grad = False
        
        # ConvGRU layers
        self.convgru1 = ConvGRUCell(2048, hidden_dim_1, kernel_size=3)
        self.convgru2 = ConvGRUCell(hidden_dim_1, hidden_dim_2, kernel_size=3)
        
        # Pooling and classification (MORE REGULARIZATION)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim_2, 64),      # Smaller intermediate
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Extract features per frame
        frame_features = []
        for t in range(self.num_frames):
            feat = self.features(x[:, t])
            frame_features.append(feat)
        
        # Initialize hidden states
        h1 = torch.zeros(batch_size, self.hidden_dim_1, 7, 7).to(x.device)
        h2 = torch.zeros(batch_size, self.hidden_dim_2, 7, 7).to(x.device)
        
        # Sequential processing
        for t in range(self.num_frames):
            h1 = self.convgru1(frame_features[t], h1)
            h2 = self.convgru2(h1, h2)
        
        # Classification
        pooled = self.pool(h2)
        output = self.classifier(pooled)
        return output
```

### Cell 9 (Optimizer):
```python
model = EmotionConvGRU(
    num_classes=Config.NUM_CLASSES,
    num_frames=Config.NUM_FRAMES,
    hidden_dim_1=256,  # Reduced
    hidden_dim_2=128,  # Reduced
    dropout=0.7        # Increased
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.00001,        # 10x lower
    weight_decay=1e-3  # 10x higher
)
```

---

## 📊 EXPECTED RESULTS WITH FIXES

### Before (Current - BROKEN):
```
Epoch 3:  Train=40%, Val=37%  ✅
Epoch 9:  Train=67%, Val=29%  🚨 OVERFITTING!
```

### After (With Fixes):
```
Epoch 5:  Train=38%, Val=39%  ✅ Generalizing!
Epoch 10: Train=45%, Val=47%  ✅ Better!
Epoch 15: Train=52%, Val=53%  ✅ Good!
Epoch 20: Train=58%, Val=60%  ✅ Target!
```

**Key difference**: Validation should be **equal or better** than training!

---

## 🚀 WHAT TO DO NOW

1. **Stop current training** (it's making things worse)

2. **Apply all 5 fixes** above

3. **Restart training** from scratch

4. **Watch for**:
   - Validation accuracy should **increase** (not decrease!)
   - Gap between train/val should be **small** (< 5%)
   - Both should improve together

5. **Target**: Val accuracy 55-65% (realistic for 4k samples)

---

## ⚠️ IMPORTANT NOTE

**Your current "best model" (Epoch 3, 37% val acc) is probably close to what you'll get!**

With these fixes, you might reach:
- Train: 55-60%
- Val: 55-60%
- Test: 50-55%

This is **much more realistic** than the 67% Zhang et al. got with 2,100 samples, because:
- They had better data quality
- Possibly better hyperparameters
- Maybe data augmentation we don't know about

**But 55% is still a HUGE improvement over your 35% baseline!**
