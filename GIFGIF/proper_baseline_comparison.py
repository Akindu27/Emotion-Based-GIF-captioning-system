"""
PROPER BASELINE COMPARISON SYSTEM
===================================

COMPARISON SETUP:
1. Baseline: Original template system (emotion + YOLO + VideoMAE)
2. Enhanced: Baseline + VGG16 scene + Groq LLM synthesis

This shows the SPECIFIC improvement from adding VGG16 + LLM!
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
from typing import List, Optional, Dict, Tuple
import cv2
import shutil

# Required imports
try:
    from groq import Groq
    from ultralytics import YOLO
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
except ImportError as e:
    print(f"⚠️  Missing package: {e}")
    print("Install: pip install groq ultralytics transformers")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"D:\IIT\Year 4\FYP\Datasets\GIFGIF_lucas")
TEST_CSV = BASE_DIR / "Research_test/csvs/test_6_groups.csv"
GIF_DIR = BASE_DIR / "Data/gifgif-images-v1/gifgif-images"
MODEL_PATH = BASE_DIR / "best_model_grouped.pth"
OUTPUT_DIR = BASE_DIR / "Research_test/proper_comparison_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# EMOTION VOCABULARY (EXACT FROM ORIGINAL SYSTEM)
# ============================================================================

emotion_vocabulary = {
    'positive_energetic': {
        'adjectives': ['joyful', 'happy', 'excited', 'cheerful', 'enthusiastic', 'energetic', 'elated', 'thrilled', 'ecstatic', 'delighted'],
        'verbs': ['dancing', 'jumping', 'celebrating', 'cheering', 'laughing', 'playing', 'running', 'clapping', 'bouncing', 'spinning']
    },
    'positive_calm': {
        'adjectives': ['peaceful', 'content', 'serene', 'relaxed', 'satisfied', 'tranquil', 'calm', 'pleased', 'gentle', 'soothing'],
        'verbs': ['sitting', 'resting', 'smiling', 'relaxing', 'enjoying', 'appreciating', 'meditating', 'breathing', 'gazing', 'watching']
    },
    'negative_intense': {
        'adjectives': ['angry', 'furious', 'fearful', 'terrified', 'disgusted', 'enraged', 'frustrated', 'scared', 'horrified', 'panicked'],
        'verbs': ['yelling', 'screaming', 'running', 'fighting', 'crying', 'panicking', 'shouting', 'fleeing', 'trembling', 'recoiling']
    },
    'negative_subdued': {
        'adjectives': ['sad', 'sorrowful', 'dejected', 'gloomy', 'melancholic', 'somber', 'depressed', 'lonely', 'disappointed', 'downcast'],
        'verbs': ['crying', 'sitting', 'looking', 'walking', 'waiting', 'sighing', 'moping', 'brooding', 'staring', 'reflecting']
    },
    'surprise': {
        'adjectives': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned', 'bewildered', 'startled', 'astounded', 'speechless', 'flabbergasted'],
        'verbs': ['reacting', 'jumping', 'gasping', 'staring', 'looking', 'responding', 'gaping', 'freezing', 'stepping back', 'covering mouth']
    },
    'contempt': {
        'adjectives': ['contemptuous', 'disdainful', 'scornful', 'dismissive', 'snide', 'mocking', 'sneering', 'arrogant', 'superior', 'condescending'],
        'verbs': ['dismissing', 'ignoring', 'mocking', 'scoffing', 'rejecting', 'sneering', 'ridiculing', 'scorning', 'rolling eyes', 'smirking']
    }
}

emotion_to_idx = {
    'contempt': 0, 'negative_intense': 1, 'negative_subdued': 2,
    'positive_calm': 3, 'positive_energetic': 4, 'surprise': 5
}

idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

# ============================================================================
# MODELS SETUP
# ============================================================================

def setup_groq():
    """Setup Groq API"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("⚠️  Set: $env:GROQ_API_KEY='gsk_...'")
        sys.exit(1)
    return Groq(api_key=api_key)

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

class VideoMAEActionDetector:
    def __init__(self, device):
        self.device = device
        print("📦 Loading VideoMAE...")
        try:
            model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            print("✅ VideoMAE loaded!")
        except:
            self.model = None
    
    def detect_action(self, frames: List[Image.Image]) -> Tuple[str, float]:
        if self.model is None:
            return None, 0.0
        
        try:
            num_frames = len(frames)
            if num_frames < 16:
                frames = frames * (16 // num_frames + 1)
            
            indices = np.linspace(0, len(frames)-1, 16, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            
            inputs = self.processor(sampled_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                top_prob, top_idx = probs.max(0)
            
            action_label = self.model.config.id2label[top_idx.item()]
            confidence = top_prob.item()
            
            # Clean action label
            action_label = action_label.replace('(', '').replace(')', '').strip()
            
            if confidence > 0.15:
                return action_label.lower(), confidence
            else:
                return None, confidence
                
        except Exception as e:
            return None, 0.0

class YOLOObjectDetector:
    def __init__(self):
        print("📦 Loading YOLO...")
        try:
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLO loaded!")
        except:
            self.model = None
    
    def detect_objects(self, frame: Image.Image) -> List[str]:
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, verbose=False)
            objects = []
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]
                    
                    if conf > 0.2:
                        objects.append((name, conf))
            
            objects.sort(key=lambda x: x[1], reverse=True)
            return [obj[0] for obj in objects[:5]]
            
        except:
            return []

class VGG16SceneAnalyzer:
    def __init__(self, device):
        self.device = device
        print("📦 Loading VGG16...")
        
        vgg16_full = models.vgg16(pretrained=True).to(device)
        vgg16_full.eval()
        
        self.conv_layers = vgg16_full.features
        self.fc_layers = nn.Sequential(*list(vgg16_full.classifier.children())[:-1])
        
        print("✅ VGG16 loaded!")
    
    def analyze_scene(self, image_tensor: torch.Tensor, image_pil: Image.Image) -> Dict[str, str]:
        with torch.no_grad():
            conv_features = self.conv_layers(image_tensor)
            final_features = self.fc_layers(conv_features.view(1, -1)).cpu().numpy()[0]
        
        img_array = np.array(image_pil.convert('RGB'))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        return {
            'content_type': self._detect_content_type(img_array, hsv),
            'setting': self._detect_setting(final_features, img_array),
            'lighting': self._detect_lighting(img_array)
        }
    
    def _detect_content_type(self, img_array, hsv):
        saturation = hsv[:, :, 1].mean()
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_ratio = unique_colors / (img_array.shape[0] * img_array.shape[1])
        
        return 'animated' if (saturation > 100 and color_ratio < 0.3) else 'real'
    
    def _detect_setting(self, features, img_array):
        mean_activation = features.mean()
        mean_brightness = img_array.mean()
        
        return 'outdoor' if (mean_activation > 0.2 and mean_brightness > 120) else 'indoor'
    
    def _detect_lighting(self, img_array):
        brightness = img_array.mean()
        return 'bright' if brightness > 140 else ('dim' if brightness < 80 else 'moderate')

# ============================================================================
# CAPTION GENERATION (EXACT FROM ORIGINAL)
# ============================================================================

def generate_baseline_caption(emotion: str, objects: List[str], action: Optional[str]) -> str:
    """
    BASELINE: Exact template system from original (emotion + YOLO + VideoMAE)
    This is what we're currently using in production!
    """
    
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    adj = random.choice(vocab['adjectives'])
    
    # Use detected action or fall back to emotion vocabulary
    if action:
        # Clean up action name
        action_words = action.lower().split()
        if len(action_words) > 2:
            verb = ' '.join(action_words[:2])
        else:
            verb = action.lower()
        
        # Ensure it ends with 'ing'
        if not verb.endswith('ing'):
            if ' ' in verb:
                parts = verb.split()
                if not parts[0].endswith('ing'):
                    parts[0] = parts[0] + 'ing'
                verb = ' '.join(parts)
            else:
                verb = verb + 'ing'
    else:
        verb = random.choice(vocab['verbs'])
    
    # Smart object prioritization (EXACT from original)
    if objects:
        animals = ['dog', 'cat', 'bird', 'horse', 'bear', 'elephant', 'giraffe', 
                   'zebra', 'lion', 'tiger', 'cow', 'sheep', 'monkey', 'rabbit']
        detected_animals = [o for o in objects if o.lower() in animals]
        
        interesting_objects = [o for o in objects if o.lower() not in 
                             ['person', 'man', 'woman', 'people', 'human']]
        
        has_person = any(o.lower() in ['person', 'man', 'woman', 'people'] for o in objects)
        
        if detected_animals:
            obj = random.choice(detected_animals)
            templates = [
                f"a {adj} person with a {obj}",
                f"someone {adj}ly {verb} with a {obj}",
                f"a {adj} person {verb} with a {obj}",
            ]
        elif interesting_objects:
            obj = random.choice(interesting_objects)
            templates = [
                f"a {adj} person {verb} with a {obj}",
                f"someone {adj}ly {verb} near a {obj}",
            ]
        elif has_person:
            templates = [
                f"a {adj} person {verb}",
                f"someone {verb} {adj}ly",
                f"someone feeling {adj} while {verb}",
            ]
        else:
            templates = [f"a {adj} scene", f"an {adj} moment"]
    else:
        templates = [
            f"a {adj} person {verb}",
            f"someone {verb} {adj}ly",
            f"someone feeling {adj} while {verb}",
        ]
    
    return random.choice(templates)

def generate_enhanced_caption(
    groq_client,
    emotion: str,
    objects: List[str],
    action: Optional[str],
    scene: Dict[str, str]
) -> str:
    """
    ENHANCED: Baseline features + VGG16 scene + Groq LLM synthesis
    This shows what VGG16 + LLM adds on top!
    """
    
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    emotion_words = vocab['adjectives'][:5]
    
    objects_str = ", ".join(objects[:3]) if objects else "no specific objects"
    action_str = action if action else "general movement"
    
    prompt = f"""Generate a vivid GIF caption using all features.

FEATURES:
- Emotion: {emotion} (use: {', '.join(emotion_words)})
- Action: {action_str}
- Objects: {objects_str}
- Scene: {scene['content_type']}, {scene['setting']}, {scene['lighting']} lighting

RULES:
1. Include ONE emotion word
2. Include the action if provided
3. Reference content type (animated/real person)
4. Mention scene context
5. Max 20 words, natural language

Generate caption only (no quotes):"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Create vivid, accurate GIF captions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=50,
        )
        
        caption = response.choices[0].message.content.strip().strip('"\'')
        
        # Verify emotion word
        has_emotion = any(word in caption.lower() for word in emotion_words)
        if not has_emotion:
            adj = random.choice(emotion_words)
            caption = f"A {adj} " + caption
        
        return caption
        
    except Exception as e:
        print(f"   ⚠️  Groq error: {e}")
        # Fallback to baseline
        return generate_baseline_caption(emotion, objects, action)

# ============================================================================
# PROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_all_frames(gif_path: Path) -> Tuple[List[Image.Image], Image.Image]:
    try:
        gif = Image.open(gif_path)
        frames = []
        
        try:
            while True:
                frames.append(gif.copy().convert('RGB'))
                gif.seek(len(frames))
        except EOFError:
            pass
        
        middle_frame = frames[len(frames) // 2]
        return frames, middle_frame
    except:
        return [], None

def main():
    print("="*70)
    print("🔬 PROPER BASELINE COMPARISON")
    print("="*70)
    print("Baseline: Original template (emotion + YOLO + VideoMAE)")
    print("Enhanced: Baseline + VGG16 scene + Groq LLM")
    print()
    
    # Setup
    groq_client = setup_groq()
    
    print("\n📦 Loading models...")
    emotion_model = GroupedEmotionClassifier(num_classes=6)
    emotion_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    emotion_model = emotion_model.to(device)
    emotion_model.eval()
    print("✅ Emotion model loaded!")
    
    videomae = VideoMAEActionDetector(device)
    yolo = YOLOObjectDetector()
    vgg16 = VGG16SceneAnalyzer(device)
    
    print()
    
    # Load test data
    df = pd.read_csv(TEST_CSV)
    selected_samples = []
    
    for emotion in df['emotion_label'].unique():
        emotion_samples = df[df['emotion_label'] == emotion].sample(n=3, random_state=42)
        selected_samples.append(emotion_samples)
    
    test_df = pd.concat(selected_samples).reset_index(drop=True)
    print(f"Selected {len(test_df)} test GIFs")
    print()
    
    print("="*70)
    print("🔬 PROCESSING")
    print("="*70)
    
    results = []
    
    for idx, row in test_df.iterrows():
        gif_id = row['gif_id']
        true_emotion = row['emotion_label']
        gif_path = GIF_DIR / f"{gif_id}.gif"
        
        print(f"\n[{idx+1}/{len(test_df)}] {gif_id}")
        
        all_frames, middle_frame = extract_all_frames(gif_path)
        if middle_frame is None:
            continue
        
        # 1. Emotion
        frame_tensor = transform(middle_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            emotion_output = emotion_model(frame_tensor)
            probs = torch.softmax(emotion_output, dim=1)[0]
            pred_idx = probs.argmax().item()
            emotion_conf = probs[pred_idx].item()
            pred_emotion = idx_to_emotion[pred_idx]
        
        # 2. Action (VideoMAE)
        action, action_conf = videomae.detect_action(all_frames)
        
        # 3. Objects (YOLO)
        objects = yolo.detect_objects(middle_frame)
        
        # 4. Scene (VGG16)
        scene = vgg16.analyze_scene(frame_tensor, middle_frame)
        
        print(f"   Emotion: {pred_emotion} ({emotion_conf*100:.1f}%)")
        print(f"   Action: {action} ({action_conf*100:.1f}%)" if action else "   Action: none")
        print(f"   Objects: {', '.join(objects)}" if objects else "   Objects: none")
        print(f"   Scene: {scene['content_type']}, {scene['setting']}, {scene['lighting']}")
        
        # Generate BOTH captions
        baseline = generate_baseline_caption(pred_emotion, objects, action)
        enhanced = generate_enhanced_caption(groq_client, pred_emotion, objects, action, scene)
        
        print(f"   📝 BASELINE: '{baseline}'")
        print(f"   ✨ ENHANCED: '{enhanced}'")
        
        results.append({
            'gif_id': gif_id,
            'true_emotion': true_emotion,
            'predicted_emotion': pred_emotion,
            'emotion_confidence': float(emotion_conf),
            'action': action,
            'action_confidence': float(action_conf) if action else 0.0,
            'objects': objects,
            'content_type': scene['content_type'],
            'setting': scene['setting'],
            'lighting': scene['lighting'],
            'baseline_caption': baseline,
            'enhanced_caption': enhanced,
        })
    
    print(f"\n✅ Processed {len(results)} GIFs!")
    
    # Save
    results_file = OUTPUT_DIR / 'comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # HTML
    generate_html(results, OUTPUT_DIR / 'comparison.html', GIF_DIR)
    
    print(f"\n✅ COMPLETE!")
    print(f"📊 Open: {OUTPUT_DIR / 'comparison.html'}")

def generate_html(results: List[Dict], output_path: Path, gif_dir: Path):
    """Generate comparison HTML"""
    
    # Copy GIFs
    images_dir = output_path.parent / 'gif_previews'
    images_dir.mkdir(exist_ok=True)
    
    for r in results:
        src = gif_dir / f"{r['gif_id']}.gif"
        dst = images_dir / f"{r['gif_id']}.gif"
        if src.exists():
            shutil.copy2(src, dst)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Baseline vs Enhanced Comparison</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .summary {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .sample {{ background: white; padding: 25px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .gif-container {{ text-align: center; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .gif-container img {{ max-width: 400px; max-height: 300px; border: 3px solid #007bff; border-radius: 8px; }}
        .caption {{ padding: 20px; margin: 15px 0; border-radius: 6px; font-size: 16px; }}
        .baseline {{ background: #fff3cd; border-left: 4px solid #ffc107; }}
        .enhanced {{ background: #d1ecf1; border-left: 4px solid #007bff; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Baseline vs Enhanced Comparison</h1>
        <div class="summary">
            <h2>System Comparison</h2>
            <p><strong>BASELINE (Current):</strong> Emotion (ResNet50) + Objects (YOLO) + Action (VideoMAE) + Templates</p>
            <p><strong>ENHANCED (Proposed):</strong> Baseline + Scene Understanding (VGG16) + LLM Synthesis (Groq)</p>
            <p><strong>Goal:</strong> Show specific improvement from VGG16 + LLM integration</p>
        </div>
"""
    
    for i, r in enumerate(results):
        objects_str = ", ".join(r['objects'][:3]) if r['objects'] else "none"
        action_str = r['action'] if r['action'] else "none"
        
        html += f"""
        <div class="sample">
            <h3>#{i+1}: {r['gif_id']}</h3>
            
            <div class="gif-container">
                <img src="gif_previews/{r['gif_id']}.gif" loading="lazy">
            </div>
            
            <p><strong>Detected:</strong> Emotion: {r['predicted_emotion']} | Action: {action_str} | Objects: {objects_str}</p>
            <p><strong>Scene (VGG16):</strong> {r['content_type']}, {r['setting']}, {r['lighting']} lighting</p>
            
            <div class="caption baseline">
                <strong>📝 BASELINE (Current System):</strong><br>
                "{r['baseline_caption']}"
            </div>
            
            <div class="caption enhanced">
                <strong>✨ ENHANCED (+ VGG16 + LLM):</strong><br>
                "{r['enhanced_caption']}"
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

if __name__ == "__main__":
    main()
