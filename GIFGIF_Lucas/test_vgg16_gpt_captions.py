"""
VGG16 + GPT Caption Generation Testing
=======================================

This notebook tests improved caption generation using:
1. VGG16 for rich visual features (4096-dim)
2. GPT-3.5 for natural language generation
3. Comparison with template-based system

Run this WHILE training is happening - no conflicts!
"""

# ============================================================================
# CELL 1: SETUP & IMPORTS
# ============================================================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import json
import random
from typing import List, Optional, Dict
import openai
from collections import defaultdict

print("="*70)
print("🚀 VGG16 + GPT CAPTION TESTING")
print("="*70)

# Paths
BASE_DIR = Path(r"D:/IIT/Year 4/FYP/Datasets/GIFGIF_lucas")
RESEARCH_DIR = BASE_DIR / "Research_test"
TEST_CSV = RESEARCH_DIR / "csvs" / "test_6_groups.csv"
GIF_DIR = BASE_DIR / "Data/gifgif-images-v1/gifgif-images"
OUTPUT_DIR = RESEARCH_DIR / "test_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print(f"Test CSV: {TEST_CSV}")
print(f"Output: {OUTPUT_DIR}")

# OpenAI API Key
# TODO: Add your OpenAI API key here!
# Get free credits at: https://platform.openai.com/signup
OPENAI_API_KEY = "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"  # ← REPLACE THIS!

if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
    print("\n⚠️  WARNING: Please add your OpenAI API key!")
    print("   Get one free at: https://platform.openai.com/signup")
    print("   Then replace OPENAI_API_KEY in Cell 1")
    USE_GPT = False
else:
    openai.api_key = OPENAI_API_KEY
    USE_GPT = True
    print("✅ OpenAI API key loaded")

print("\n✅ Setup complete!")

# ============================================================================
# CELL 2: EMOTION VOCABULARY (From your system)
# ============================================================================

emotion_vocabulary = {
    'positive_energetic': {
        'adjectives': ['joyful', 'happy', 'excited', 'cheerful', 'enthusiastic', 
                      'energetic', 'elated', 'thrilled', 'ecstatic', 'delighted'],
        'verbs': ['dancing', 'jumping', 'celebrating', 'cheering', 'laughing', 
                 'playing', 'running', 'clapping', 'bouncing', 'spinning']
    },
    'positive_calm': {
        'adjectives': ['peaceful', 'content', 'serene', 'relaxed', 'satisfied', 
                      'tranquil', 'calm', 'pleased', 'gentle', 'soothing'],
        'verbs': ['sitting', 'resting', 'smiling', 'relaxing', 'enjoying', 
                 'appreciating', 'meditating', 'breathing', 'gazing', 'watching']
    },
    'negative_intense': {
        'adjectives': ['angry', 'furious', 'fearful', 'terrified', 'disgusted', 
                      'enraged', 'frustrated', 'scared', 'horrified', 'panicked'],
        'verbs': ['yelling', 'screaming', 'running', 'fighting', 'crying', 
                 'panicking', 'shouting', 'fleeing', 'trembling', 'recoiling']
    },
    'negative_subdued': {
        'adjectives': ['sad', 'sorrowful', 'dejected', 'gloomy', 'melancholic', 
                      'somber', 'depressed', 'lonely', 'disappointed', 'downcast'],
        'verbs': ['crying', 'sitting', 'looking', 'walking', 'waiting', 
                 'sighing', 'moping', 'brooding', 'staring', 'reflecting']
    },
    'surprise': {
        'adjectives': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned', 
                      'bewildered', 'startled', 'astounded', 'speechless', 'flabbergasted'],
        'verbs': ['reacting', 'jumping', 'gasping', 'staring', 'looking', 
                 'responding', 'gaping', 'freezing', 'stepping back', 'covering mouth']
    },
    'contempt': {
        'adjectives': ['contemptuous', 'disdainful', 'scornful', 'dismissive', 
                      'snide', 'mocking', 'sneering', 'arrogant', 'superior', 'condescending'],
        'verbs': ['dismissing', 'ignoring', 'mocking', 'scoffing', 'rejecting', 
                 'sneering', 'ridiculing', 'scorning', 'rolling eyes', 'smirking']
    }
}

print("✅ Emotion vocabulary loaded (6 groups)")

# ============================================================================
# CELL 3: LOAD EMOTION MODEL
# ============================================================================

class GroupedEmotionClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(GroupedEmotionClassifier, self).__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
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

print("📦 Loading emotion model...")
emotion_model = GroupedEmotionClassifier(num_classes=6)

# Try to load from different possible locations
model_paths = [
    RESEARCH_DIR / "models" / "best_model_grouped.pth",
    BASE_DIR / "best_model_grouped.pth",
    Path("best_model_grouped.pth")
]

model_loaded = False
for model_path in model_paths:
    if model_path.exists():
        emotion_model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ Loaded emotion model from: {model_path}")
        model_loaded = True
        break

if not model_loaded:
    print("⚠️  Could not find emotion model!")
    print("   Expected locations:")
    for p in model_paths:
        print(f"   - {p}")
    
emotion_model = emotion_model.to(device)
emotion_model.eval()

emotion_groups = ['contempt', 'negative_intense', 'negative_subdued',
                 'positive_calm', 'positive_energetic', 'surprise']

print("✅ Emotion model ready!")

# ============================================================================
# CELL 4: LOAD VGG16 MODEL
# ============================================================================

print("📦 Loading VGG16 model...")

# Load pretrained VGG16
vgg16 = models.vgg16(weights='IMAGENET1K_V1')

# Remove final classification layer, keep 4096-dim features
vgg16.classifier = vgg16.classifier[:-1]  # Stops at fc7 (4096-dim)

vgg16 = vgg16.to(device)
vgg16.eval()

print("✅ VGG16 loaded!")
print(f"   Feature dimension: 4096")

# Test VGG16
test_img = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    test_features = vgg16(test_img)
print(f"   Test output shape: {test_features.shape}")  # Should be [1, 4096]

# ============================================================================
# CELL 5: IMAGE PREPROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def extract_middle_frame(gif_path: Path) -> Optional[Image.Image]:
    """Extract middle frame from GIF"""
    try:
        with Image.open(gif_path) as gif:
            # Count frames
            n_frames = 0
            max_frames = 200  # Safety limit
            while n_frames < max_frames:
                try:
                    gif.seek(n_frames)
                    n_frames += 1
                except EOFError:
                    break
            
            # Get middle frame
            if n_frames == 0:
                return None
            
            middle_idx = n_frames // 2
            gif.seek(middle_idx)
            frame = gif.convert('RGB')
            
            return frame
    except Exception as e:
        print(f"⚠️  Error loading {gif_path.name}: {e}")
        return None

print("✅ Image preprocessing ready!")

# ============================================================================
# CELL 6: TEMPLATE CAPTION GENERATION (Baseline)
# ============================================================================

def generate_template_caption(emotion: str, confidence: float) -> str:
    """Generate template-based caption (your current system)"""
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    
    adj = random.choice(vocab['adjectives'])
    verb = random.choice(vocab['verbs'])
    
    templates = [
        f"a {adj} person {verb}",
        f"someone {verb} {adj}ly",
        f"an animated scene of a {adj} person {verb}",
        f"someone feeling {adj} while {verb}",
    ]
    
    return random.choice(templates)

print("✅ Template caption generator ready!")

# ============================================================================
# CELL 7: GPT CAPTION GENERATION (New!)
# ============================================================================

def generate_gpt_caption(
    emotion: str,
    confidence: float,
    vgg_features: np.ndarray,
    vocab: dict
) -> str:
    """Generate caption using GPT-3.5 with emotion constraints"""
    
    if not USE_GPT:
        # Fallback to template if no API key
        return generate_template_caption(emotion, confidence)
    
    # Get emotion words
    emotion_words = vocab['adjectives'][:5]  # Top 5 emotion words
    
    # Construct prompt
    prompt = f"""Generate a natural, single-sentence caption for an animated GIF.

The GIF shows a {emotion.replace('_', ' ')} emotion (confidence: {confidence:.0%}).

REQUIREMENTS:
1. MUST use ONE of these emotion words: {', '.join(emotion_words)}
2. Describe what might be happening in a GIF with this emotion
3. Maximum 15 words
4. Sound natural and human-written
5. Be specific and vivid

EXAMPLES OF GOOD CAPTIONS:
- "A joyful child plays with a golden retriever in the backyard"
- "Someone gasps in surprise at an unexpected gift"
- "A peaceful moment of someone watching the sunset"

Generate ONLY the caption (no quotes, no explanation):"""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional GIF caption writer who creates natural, emotion-aware descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        caption = response.choices[0].message.content.strip()
        
        # Remove quotes if GPT added them
        caption = caption.strip('"').strip("'")
        
        # Verify emotion word is included
        if not any(word in caption.lower() for word in emotion_words):
            print(f"   ⚠️  GPT caption missing emotion word, using template")
            return generate_template_caption(emotion, confidence)
        
        return caption
        
    except Exception as e:
        print(f"   ⚠️  GPT API error: {e}")
        return generate_template_caption(emotion, confidence)

print("✅ GPT caption generator ready!")
if not USE_GPT:
    print("   ⚠️  API key not set - will use templates as fallback")

# ============================================================================
# CELL 8: PROCESS TEST GIFS
# ============================================================================

print("\n" + "="*70)
print("🔬 PROCESSING TEST GIFS")
print("="*70)

# Load test data
test_df = pd.read_csv(TEST_CSV)
print(f"Total test samples: {len(test_df)}")

# Sample 20 GIFs (diverse emotions)
samples_per_emotion = 3
test_samples = []

for emotion in emotion_groups:
    emotion_gifs = test_df[test_df['emotion_label'] == emotion]
    if len(emotion_gifs) >= samples_per_emotion:
        sampled = emotion_gifs.sample(n=samples_per_emotion, random_state=42)
        test_samples.append(sampled)

test_samples = pd.concat(test_samples).reset_index(drop=True)
print(f"Selected {len(test_samples)} test GIFs ({samples_per_emotion} per emotion)")

# Process each GIF
results = []

for idx, row in test_samples.iterrows():
    gif_id = row['gif_id']
    true_emotion = row['emotion_label']
    gif_path = GIF_DIR / f"{gif_id}.gif"
    
    print(f"\n[{idx+1}/{len(test_samples)}] Processing {gif_id}...")
    print(f"   True emotion: {true_emotion}")
    
    if not gif_path.exists():
        print(f"   ⚠️  GIF not found: {gif_path}")
        continue
    
    # Extract middle frame
    frame = extract_middle_frame(gif_path)
    if frame is None:
        print(f"   ⚠️  Could not extract frame")
        continue
    
    # Preprocess
    frame_tensor = transform(frame).unsqueeze(0).to(device)
    
    # 1. Detect emotion
    with torch.no_grad():
        emotion_output = emotion_model(frame_tensor)
        probs = torch.softmax(emotion_output, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
    
    pred_emotion = emotion_groups[pred_idx]
    print(f"   Predicted: {pred_emotion} ({confidence:.1%})")
    
    # 2. Extract VGG16 features
    with torch.no_grad():
        vgg_features = vgg16(frame_tensor).cpu().numpy()[0]
    print(f"   VGG16 features: {vgg_features.shape}")
    
    # 3. Generate captions
    vocab = emotion_vocabulary[pred_emotion]
    
    template_caption = generate_template_caption(pred_emotion, confidence)
    gpt_caption = generate_gpt_caption(pred_emotion, confidence, vgg_features, vocab)
    
    print(f"   Template: '{template_caption}'")
    print(f"   GPT:      '{gpt_caption}'")
    
    # Store results
    results.append({
        'gif_id': gif_id,
        'true_emotion': true_emotion,
        'pred_emotion': pred_emotion,
        'confidence': confidence,
        'template_caption': template_caption,
        'gpt_caption': gpt_caption,
        'vgg_features_mean': vgg_features.mean(),
        'vgg_features_std': vgg_features.std()
    })

print(f"\n✅ Processed {len(results)} GIFs successfully!")

# ============================================================================
# CELL 9: SAVE RESULTS
# ============================================================================

# Save to JSON
results_file = OUTPUT_DIR / "vgg_gpt_test_results.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {results_file}")

# Save to CSV for easy viewing
results_df = pd.DataFrame(results)
csv_file = OUTPUT_DIR / "vgg_gpt_test_results.csv"
results_df.to_csv(csv_file, index=False)
print(f"💾 CSV saved to: {csv_file}")

# ============================================================================
# CELL 10: COMPARISON ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("📊 CAPTION COMPARISON ANALYSIS")
print("="*70)

# Length comparison
template_lengths = [len(r['template_caption'].split()) for r in results]
gpt_lengths = [len(r['gpt_caption'].split()) for r in results]

print(f"\nCaption Length:")
print(f"   Template: {np.mean(template_lengths):.1f} ± {np.std(template_lengths):.1f} words")
print(f"   GPT:      {np.mean(gpt_lengths):.1f} ± {np.std(gpt_lengths):.1f} words")

# Emotion word inclusion check
def has_emotion_word(caption, emotion, vocab):
    """Check if caption contains emotion word"""
    emotion_words = vocab[emotion]['adjectives'] + vocab[emotion]['verbs']
    return any(word in caption.lower() for word in emotion_words)

template_emotion_count = sum(
    has_emotion_word(r['template_caption'], r['pred_emotion'], emotion_vocabulary)
    for r in results
)
gpt_emotion_count = sum(
    has_emotion_word(r['gpt_caption'], r['pred_emotion'], emotion_vocabulary)
    for r in results
)

print(f"\nEmotion Word Inclusion:")
print(f"   Template: {template_emotion_count}/{len(results)} ({100*template_emotion_count/len(results):.0f}%)")
print(f"   GPT:      {gpt_emotion_count}/{len(results)} ({100*gpt_emotion_count/len(results):.0f}%)")

# Show sample comparisons
print(f"\n" + "="*70)
print("📝 SAMPLE COMPARISONS")
print("="*70)

for i in range(min(5, len(results))):
    r = results[i]
    print(f"\n{i+1}. GIF: {r['gif_id']}")
    print(f"   Emotion: {r['pred_emotion']} ({r['confidence']:.0%})")
    print(f"   Template: '{r['template_caption']}'")
    print(f"   GPT:      '{r['gpt_caption']}'")

# ============================================================================
# CELL 11: MANUAL EVALUATION GUIDE
# ============================================================================

print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)

print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"   - vgg_gpt_test_results.json")
print(f"   - vgg_gpt_test_results.csv")

print("\n📋 NEXT STEPS:")
print("   1. Review the sample comparisons above")
print("   2. Open the CSV file to see all results")
print("   3. Manually rate caption quality (1-5 scale)")
print("   4. Compare naturalness: Template vs GPT")

print("\n💡 EVALUATION CRITERIA:")
print("   Naturalness:  How human-like does it sound?")
print("   Relevance:    Does it fit the emotion?")
print("   Variety:      Are captions unique?")
print("   Specificity:  Is it vivid and detailed?")

if USE_GPT:
    print("\n✅ GPT captions generated successfully!")
else:
    print("\n⚠️  GPT was not used (no API key)")
    print("   Recommendation:")
    print("   1. Get free OpenAI API key")
    print("   2. Add to Cell 1")
    print("   3. Rerun cells 8-10")

print("\n🎉 You can now decide:")
print("   - Keep template system (simpler, faster)")
print("   - Integrate GPT system (better quality)")
print("   - Hybrid approach (GPT with template fallback)")
