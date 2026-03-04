"""
VGG16 + GEMINI Caption Testing Script (FREE VERSION!)
=======================================================

Uses Google Gemini API (completely FREE - no credit card needed!)

Setup:
1. Get FREE API key: https://aistudio.google.com/app/apikey
2. Set environment variable: GEMINI_API_KEY=your_key_here
3. Install: pip install google-generativeai
4. Run this script!

Features:
- VGG16 feature extraction (4096-dim)
- FREE Gemini Pro caption generation
- Side-by-side Template vs Gemini comparison
- Saves results to JSON and HTML
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import random
from typing import List, Optional, Dict
import google.generativeai as genai

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths (update these!)
BASE_DIR = Path(r"D:\IIT\Year 4\FYP\Datasets\GIFGIF_lucas")
TEST_CSV = BASE_DIR / "Research_test/csvs/test_6_groups.csv"
GIF_DIR = BASE_DIR / "Data/gifgif-images-v1/gifgif-images"
MODEL_PATH = BASE_DIR / "best_model_grouped.pth"
OUTPUT_DIR = BASE_DIR / "Research_test/gemini_test_results"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# GEMINI API SETUP
# ============================================================================

def setup_gemini():
    """Setup Gemini API with free key"""
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("\n" + "="*70)
        print("⚠️  GEMINI API KEY NOT FOUND!")
        print("="*70)
        print("\nQuick setup (2 minutes):")
        print("1. Go to: https://aistudio.google.com/app/apikey")
        print("2. Click 'Create API Key' (FREE - no credit card needed!)")
        print("3. Copy the key (starts with 'AIza...')")
        print("\n4. In PowerShell, run:")
        print('   $env:GEMINI_API_KEY="your_key_here"')
        print("\n5. Run this script again!")
        print("="*70)
        sys.exit(1)
    
    genai.configure(api_key=api_key)
    print("✅ Gemini API configured!")
    return genai.GenerativeModel('gemini-pro')

# ============================================================================
# EMOTION MODEL
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

# ============================================================================
# EMOTION VOCABULARY
# ============================================================================

emotion_vocabulary = {
    'positive_energetic': {
        'adjectives': ['joyful', 'happy', 'excited', 'cheerful', 'enthusiastic', 'energetic', 'elated', 'thrilled'],
        'verbs': ['dancing', 'jumping', 'celebrating', 'cheering', 'laughing', 'playing', 'clapping', 'bouncing']
    },
    'positive_calm': {
        'adjectives': ['peaceful', 'content', 'serene', 'relaxed', 'satisfied', 'tranquil', 'calm', 'pleased'],
        'verbs': ['sitting', 'resting', 'smiling', 'relaxing', 'enjoying', 'appreciating', 'breathing', 'gazing']
    },
    'negative_intense': {
        'adjectives': ['angry', 'furious', 'fearful', 'terrified', 'disgusted', 'enraged', 'frustrated', 'scared'],
        'verbs': ['yelling', 'screaming', 'running', 'fighting', 'crying', 'panicking', 'shouting', 'fleeing']
    },
    'negative_subdued': {
        'adjectives': ['sad', 'sorrowful', 'dejected', 'gloomy', 'melancholic', 'somber', 'depressed', 'lonely'],
        'verbs': ['crying', 'sitting', 'looking', 'walking', 'waiting', 'sighing', 'moping', 'staring']
    },
    'surprise': {
        'adjectives': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned', 'bewildered', 'startled'],
        'verbs': ['reacting', 'jumping', 'gasping', 'staring', 'looking', 'gaping', 'freezing', 'stepping back']
    },
    'contempt': {
        'adjectives': ['contemptuous', 'disdainful', 'scornful', 'dismissive', 'mocking', 'sneering', 'arrogant'],
        'verbs': ['dismissing', 'ignoring', 'mocking', 'scoffing', 'rejecting', 'sneering', 'rolling eyes', 'smirking']
    }
}

emotion_to_idx = {
    'contempt': 0,
    'negative_intense': 1,
    'negative_subdued': 2,
    'positive_calm': 3,
    'positive_energetic': 4,
    'surprise': 5
}

idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

# ============================================================================
# CAPTION GENERATORS
# ============================================================================

def generate_template_caption(emotion: str, confidence: float) -> str:
    """Generate template-based caption"""
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    adj = random.choice(vocab['adjectives'])
    verb = random.choice(vocab['verbs'])
    
    templates = [
        f"a {adj} person {verb}",
        f"someone {verb} {adj}ly",
        f"someone feeling {adj} while {verb}",
    ]
    
    return random.choice(templates)

def generate_gemini_caption(
    gemini_model,
    emotion: str,
    confidence: float,
    vgg_features: np.ndarray
) -> str:
    """Generate caption using FREE Gemini API"""
    
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    emotion_words = vocab['adjectives'][:5]
    emotion_verbs = vocab['verbs'][:5]
    
    # Build prompt
    prompt = f"""You are a GIF caption writer. Generate a natural, single-sentence caption.

EMOTION DETECTED: {emotion} (confidence: {confidence:.0%})
REQUIRED EMOTION WORDS (use at least ONE): {', '.join(emotion_words)}
SUGGESTED ACTIONS: {', '.join(emotion_verbs)}

RULES:
1. MUST include at least one emotion word from the list above
2. Maximum 15 words
3. Sound natural and human-like
4. Describe what's happening in the GIF
5. Be specific and vivid

EXAMPLES:
Good: "A joyful child dances excitedly in the living room"
Good: "Someone stares in shocked amazement at the surprise"
Good: "An angry person yells furiously at the situation"
Bad: "A person doing something" (too generic)
Bad: "Interesting moment captured" (no emotion word)

Generate ONLY the caption (no quotes, no explanation, no preamble):"""

    try:
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.7,
                'max_output_tokens': 50,
            }
        )
        
        caption = response.text.strip()
        
        # Remove quotes if Gemini added them
        caption = caption.strip('"\'')
        
        # Verify emotion word included
        caption_lower = caption.lower()
        has_emotion = any(word in caption_lower for word in emotion_words)
        
        if not has_emotion:
            print(f"   ⚠️  Gemini didn't include emotion word, using template fallback")
            return generate_template_caption(emotion, confidence)
        
        return caption
        
    except Exception as e:
        print(f"   ⚠️  Gemini API error: {e}")
        return generate_template_caption(emotion, confidence)

# ============================================================================
# VGG16 FEATURE EXTRACTION
# ============================================================================

def setup_vgg16():
    """Load VGG16 for feature extraction"""
    vgg16 = models.vgg16(pretrained=True)
    # Remove final classification layer, keep 4096-dim features
    vgg16.classifier = vgg16.classifier[:-1]
    vgg16.eval()
    return vgg16.to(device)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_middle_frame(gif_path: Path) -> Optional[Image.Image]:
    """Extract middle frame from GIF"""
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
        return gif.convert('RGB')
    except Exception as e:
        print(f"   ⚠️  Error loading {gif_path.name}: {e}")
        return None

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    print("="*70)
    print("🚀 GEMINI CAPTION TESTING (FREE VERSION)")
    print("="*70)
    print(f"Test CSV: {TEST_CSV}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Setup Gemini
    gemini_model = setup_gemini()
    print()
    
    # Load emotion model
    print("📦 Loading emotion model...")
    emotion_model = GroupedEmotionClassifier(num_classes=6)
    emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    emotion_model = emotion_model.to(device)
    emotion_model.eval()
    print(f"✅ Loaded emotion model from: {MODEL_PATH}")
    
    # Load VGG16
    print("📦 Loading VGG16...")
    vgg16 = setup_vgg16()
    print("✅ VGG16 loaded!")
    print()
    
    # Load test data
    df = pd.read_csv(TEST_CSV)
    print(f"Total test samples: {len(df)}")
    
    # Select 3 samples per emotion (18 total)
    samples_per_emotion = 3
    selected_samples = []
    
    for emotion in df['emotion_label'].unique():
        emotion_samples = df[df['emotion_label'] == emotion].sample(
            n=min(samples_per_emotion, len(df[df['emotion_label'] == emotion])),
            random_state=42
        )
        selected_samples.append(emotion_samples)
    
    test_df = pd.concat(selected_samples).reset_index(drop=True)
    print(f"Selected {len(test_df)} test GIFs ({samples_per_emotion} per emotion)")
    print()
    
    # Process each GIF
    print("="*70)
    print("🔬 PROCESSING TEST GIFS")
    print("="*70)
    
    results = []
    
    for idx, row in test_df.iterrows():
        gif_id = row['gif_id']
        true_emotion = row['emotion_label']
        gif_path = GIF_DIR / f"{gif_id}.gif"
        
        print(f"\n[{idx+1}/{len(test_df)}] Processing {gif_id}...")
        print(f"   True emotion: {true_emotion}")
        
        # Extract middle frame
        frame = extract_middle_frame(gif_path)
        if frame is None:
            continue
        
        # Predict emotion
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emotion_output = emotion_model(frame_tensor)
            probs = torch.softmax(emotion_output, dim=1)[0]
            pred_idx = probs.argmax().item()
            confidence = probs[pred_idx].item()
            pred_emotion = idx_to_emotion[pred_idx]
        
        print(f"   Predicted: {pred_emotion} ({confidence*100:.1f}%)")
        
        # Extract VGG16 features
        with torch.no_grad():
            vgg_features = vgg16(frame_tensor).cpu().numpy()[0]
        
        print(f"   VGG16 features: {vgg_features.shape}")
        
        # Generate captions
        template_caption = generate_template_caption(pred_emotion, confidence)
        gemini_caption = generate_gemini_caption(gemini_model, pred_emotion, confidence, vgg_features)
        
        print(f"   Template: '{template_caption}'")
        print(f"   Gemini:   '{gemini_caption}'")
        
        # Save result
        results.append({
            'gif_id': gif_id,
            'true_emotion': true_emotion,
            'predicted_emotion': pred_emotion,
            'confidence': float(confidence),
            'template_caption': template_caption,
            'gemini_caption': gemini_caption,
            'vgg_features_mean': float(vgg_features.mean()),
            'vgg_features_std': float(vgg_features.std())
        })
    
    print(f"\n✅ Processed {len(results)} GIFs successfully!")
    
    # Save results
    results_file = OUTPUT_DIR / 'gemini_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Generate HTML comparison
    generate_html_comparison(results, OUTPUT_DIR / 'gemini_comparison.html')
    
    print(f"\n✅ TESTING COMPLETE!")
    print(f"📊 Check results:")
    print(f"   - JSON: {results_file}")
    print(f"   - HTML: {OUTPUT_DIR / 'gemini_comparison.html'}")

# ============================================================================
# HTML REPORT GENERATION
# ============================================================================

def generate_html_comparison(results: List[Dict], output_path: Path):
    """Generate HTML comparison report"""
    
    html = """
<!DOCTYPE html>
<html>
<head>
    <title>Template vs Gemini Caption Comparison</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .sample { 
            background: white; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .emotion { 
            display: inline-block; 
            padding: 5px 10px; 
            border-radius: 4px; 
            font-weight: bold; 
            margin: 5px;
        }
        .correct { background: #d4edda; color: #155724; }
        .incorrect { background: #f8d7da; color: #721c24; }
        .caption { 
            padding: 15px; 
            margin: 10px 0; 
            border-left: 4px solid #007bff; 
            background: #f8f9fa;
        }
        .template { border-left-color: #28a745; }
        .gemini { border-left-color: #007bff; }
        .stats { 
            background: #e7f3ff; 
            padding: 10px; 
            border-radius: 4px; 
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Template vs Gemini Caption Comparison</h1>
        <div class="stats">
            <strong>Total GIFs tested:</strong> """ + str(len(results)) + """<br>
            <strong>Test Date:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M") + """
        </div>
"""
    
    for i, result in enumerate(results):
        is_correct = result['true_emotion'] == result['predicted_emotion']
        emotion_class = 'correct' if is_correct else 'incorrect'
        
        html += f"""
        <div class="sample">
            <h3>Sample {i+1}: {result['gif_id']}</h3>
            <div>
                <span class="emotion {emotion_class}">True: {result['true_emotion']}</span>
                <span class="emotion">Predicted: {result['predicted_emotion']} ({result['confidence']*100:.1f}%)</span>
            </div>
            
            <div class="caption template">
                <strong>📝 Template:</strong><br>
                "{result['template_caption']}"
            </div>
            
            <div class="caption gemini">
                <strong>🤖 Gemini (FREE):</strong><br>
                "{result['gemini_caption']}"
            </div>
        </div>
"""
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"📄 HTML report generated: {output_path}")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
