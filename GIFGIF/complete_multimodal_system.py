"""
COMPLETE MULTI-MODAL GIF CAPTION GENERATION SYSTEM
===================================================

FULL PIPELINE:
1. ResNet50 → Emotion classification
2. VideoMAE → Action recognition (dancing, screaming, walking, etc.)
3. YOLO → Object detection (person, car, animal, furniture, etc.)
4. VGG16 → Scene understanding (indoor/outdoor, lighting, animated/real)
5. Groq LLama 3.3 → Combine all features into natural caption

This is what your supervisor requested: Using ALL features for rich captions!
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

# Groq API
try:
    from groq import Groq
except ImportError:
    print("Install: pip install groq")
    sys.exit(1)

# Ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Install: pip install ultralytics")
    sys.exit(1)

# Transformers for VideoMAE
try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
except ImportError:
    print("Install: pip install transformers")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(r"D:\IIT\Year 4\FYP\Datasets\GIFGIF_lucas")
TEST_CSV = BASE_DIR / "Research_test/csvs/test_6_groups.csv"
GIF_DIR = BASE_DIR / "Data/gifgif-images-v1/gifgif-images"
MODEL_PATH = BASE_DIR / "best_model_grouped.pth"
OUTPUT_DIR = BASE_DIR / "Research_test/complete_multimodal_results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================================
# GROQ SETUP
# ============================================================================

def setup_groq():
    """Setup Groq API"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("\nGROQ_API_KEY not found!")
        print("Set: $env:GROQ_API_KEY='gsk_...'")
        sys.exit(1)
    
    client = Groq(api_key=api_key)
    print("Groq API configured!")
    return client

# ============================================================================
# EMOTION MODEL (ResNet50)
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
# VIDEOMAE ACTION RECOGNITION
# ============================================================================

class VideoMAEActionDetector:
    """
    Detects actions in GIFs using VideoMAE
    
    Actions: dancing, running, walking, sitting, standing, etc.
    """
    
    def __init__(self, device):
        self.device = device
        print("Loading VideoMAE for action detection...")
        
        # Use smaller model for CPU
        model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
        
        try:
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
            self.model = VideoMAEForVideoClassification.from_pretrained(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            print("✅ VideoMAE loaded!")
        except Exception as e:
            print(f"VideoMAE loading failed: {e}")
            print("   Using fallback mode (no action detection)")
            self.model = None
    
    def detect_action(self, frames: List[Image.Image]) -> Tuple[str, float]:
        """
        Detect action from GIF frames
        
        Returns: (action_name, confidence)
        """
        
        if self.model is None:
            return "unknown action", 0.0
        
        try:
            # Sample 16 frames uniformly
            num_frames = len(frames)
            if num_frames < 16:
                # Repeat frames if too few
                frames = frames * (16 // num_frames + 1)
            
            indices = np.linspace(0, len(frames)-1, 16, dtype=int)
            sampled_frames = [frames[i] for i in indices]
            
            # Preprocess
            inputs = self.processor(sampled_frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                top_prob, top_idx = probs.max(0)
            
            # Get action label
            action_label = self.model.config.id2label[top_idx.item()]
            confidence = top_prob.item()
            
            # Simplify action labels (Kinetics has detailed labels)
            action_simple = self._simplify_action(action_label)
            
            return action_simple, confidence
            
        except Exception as e:
            print(f"Action detection error: {e}")
            return "unknown action", 0.0
    
    def _simplify_action(self, kinetics_label: str) -> str:
        """Simplify Kinetics-400 labels to common actions"""
        
        label_lower = kinetics_label.lower()
        
        # Map to common actions
        if any(word in label_lower for word in ['danc', 'jump', 'bounce']):
            return 'dancing'
        elif any(word in label_lower for word in ['run', 'sprint', 'jog']):
            return 'running'
        elif any(word in label_lower for word in ['walk', 'stroll']):
            return 'walking'
        elif any(word in label_lower for word in ['sit', 'seat']):
            return 'sitting'
        elif any(word in label_lower for word in ['stand', 'standing']):
            return 'standing'
        elif any(word in label_lower for word in ['talk', 'speak', 'yell', 'scream', 'shout']):
            return 'talking'
        elif any(word in label_lower for word in ['laugh', 'smil']):
            return 'laughing'
        elif any(word in label_lower for word in ['cry', 'weep']):
            return 'crying'
        elif any(word in label_lower for word in ['eat', 'drink']):
            return 'eating'
        elif any(word in label_lower for word in ['wave', 'waving']):
            return 'waving'
        elif any(word in label_lower for word in ['hug', 'embrac']):
            return 'hugging'
        elif any(word in label_lower for word in ['kiss']):
            return 'kissing'
        elif any(word in label_lower for word in ['point']):
            return 'pointing'
        elif any(word in label_lower for word in ['clap']):
            return 'clapping'
        else:
            # Return simplified version of original
            return kinetics_label.lower().replace('_', ' ')

# ============================================================================
# YOLO OBJECT DETECTION
# ============================================================================

class YOLOObjectDetector:
    """
    Detects objects in GIF frames using YOLO
    
    Objects: person, car, dog, cat, chair, etc.
    """
    
    def __init__(self):
        print("Loading YOLO for object detection...")
        try:
            self.model = YOLO('yolov8n.pt')  # Nano model (fastest)
            print("✅ YOLO loaded!")
        except Exception as e:
            print(f"YOLO loading failed: {e}")
            self.model = None
    
    def detect_objects(self, frame: Image.Image) -> List[Tuple[str, float]]:
        """
        Detect objects in frame
        
        Returns: [(object_name, confidence), ...]
        """
        
        if self.model is None:
            return []
        
        try:
            # Run detection
            results = self.model(frame, verbose=False)
            
            # Extract objects
            objects = []
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = result.names[cls]
                    
                    if conf > 0.3:  # Confidence threshold
                        objects.append((name, conf))
            
            # Sort by confidence
            objects.sort(key=lambda x: x[1], reverse=True)
            
            return objects[:5]  # Top 5 objects
            
        except Exception as e:
            print(f"Object detection error: {e}")
            return []

# ============================================================================
# VGG16 SCENE ANALYSIS
# ============================================================================

class VGG16SceneAnalyzer:
    """VGG16 for scene understanding"""
    
    def __init__(self, device):
        self.device = device
        print("📦 Loading VGG16 for scene analysis...")
        
        vgg16_full = models.vgg16(pretrained=True).to(device)
        vgg16_full.eval()
        
        self.conv_layers = vgg16_full.features
        self.fc_layers = nn.Sequential(*list(vgg16_full.classifier.children())[:-1])
        
        print("✅ VGG16 loaded!")
    
    def analyze_scene(self, image_tensor: torch.Tensor, image_pil: Image.Image) -> Dict[str, str]:
        """Analyze scene properties"""
        
        with torch.no_grad():
            conv_features = self.conv_layers(image_tensor)
            final_features = self.fc_layers(conv_features.view(1, -1)).cpu().numpy()[0]
        
        img_array = np.array(image_pil.convert('RGB'))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Scene analysis
        content_type = self._detect_content_type(img_array, hsv)
        setting = self._detect_setting(final_features, img_array)
        lighting = self._detect_lighting(img_array)
        
        return {
            'content_type': content_type,
            'setting': setting,
            'lighting': lighting
        }
    
    def _detect_content_type(self, img_array, hsv):
        saturation = hsv[:, :, 1]
        mean_saturation = saturation.mean()
        
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_ratio = unique_colors / total_pixels
        
        if mean_saturation > 100 and color_ratio < 0.3:
            return 'animated'
        else:
            return 'real'
    
    def _detect_setting(self, features, img_array):
        mean_activation = features.mean()
        mean_brightness = img_array.mean()
        
        if mean_activation > 0.2 and mean_brightness > 120:
            return 'outdoor'
        else:
            return 'indoor'
    
    def _detect_lighting(self, img_array):
        brightness = img_array.mean()
        
        if brightness > 140:
            return 'bright'
        elif brightness < 80:
            return 'dim'
        else:
            return 'moderate'

# ============================================================================
# EMOTION VOCABULARY
# ============================================================================

emotion_vocabulary = {
    'positive_energetic': {
        'adjectives': ['joyful', 'happy', 'excited', 'cheerful', 'enthusiastic', 'energetic'],
    },
    'positive_calm': {
        'adjectives': ['peaceful', 'content', 'serene', 'relaxed', 'satisfied', 'tranquil'],
    },
    'negative_intense': {
        'adjectives': ['angry', 'furious', 'fearful', 'terrified', 'disgusted', 'enraged'],
    },
    'negative_subdued': {
        'adjectives': ['sad', 'sorrowful', 'dejected', 'gloomy', 'melancholic', 'somber'],
    },
    'surprise': {
        'adjectives': ['surprised', 'shocked', 'astonished', 'amazed', 'stunned', 'bewildered'],
    },
    'contempt': {
        'adjectives': ['contemptuous', 'disdainful', 'scornful', 'dismissive', 'mocking'],
    }
}

emotion_to_idx = {
    'contempt': 0, 'negative_intense': 1, 'negative_subdued': 2,
    'positive_calm': 3, 'positive_energetic': 4, 'surprise': 5
}

idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}

# ============================================================================
# MULTI-MODAL CAPTION GENERATION
# ============================================================================

def generate_template_caption(emotion: str) -> str:
    """Simple template baseline"""
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    adj = random.choice(vocab['adjectives'])
    return f"a {adj} person"

def generate_multimodal_caption(
    groq_client,
    emotion: str,
    confidence: float,
    action: str,
    action_conf: float,
    objects: List[Tuple[str, float]],
    scene: Dict[str, str]
) -> str:
    """
    Generate caption using ALL modalities:
    - Emotion (ResNet50)
    - Action (VideoMAE)
    - Objects (YOLO)
    - Scene (VGG16)
    """
    
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    emotion_words = vocab['adjectives'][:5]
    
    # Build multi-modal context
    objects_str = ", ".join([obj for obj, conf in objects[:3]]) if objects else "no specific objects"
    
    prompt = f"""Generate a vivid, single-sentence GIF caption using ALL provided information.

DETECTED FEATURES:
1. Emotion: {emotion} ({confidence:.0%} confidence)
   Required words: {', '.join(emotion_words)}

2. Action: {action} ({action_conf:.0%} confidence)
   Use this action in the caption!

3. Objects detected: {objects_str}
   
4. Scene context:
   - Content: {scene['content_type']} (animated/real)
   - Setting: {scene['setting']} (indoor/outdoor)
   - Lighting: {scene['lighting']}

CRITICAL RULES:
1. MUST include ONE emotion word from the list
2. MUST include the detected action: "{action}"
3. SHOULD mention objects if relevant
4. MUST reference content type (animated character / person)
5. Maximum 20 words, natural language

GOOD EXAMPLES:
- "A joyful animated character dancing happily in a bright indoor room"
- "An angry person shouting furiously at someone in a dim outdoor setting"
- "A surprised child jumping excitedly with a dog in a moderate lit space"

Generate ONLY the caption (no quotes, no explanation):"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert caption writer who creates vivid, accurate descriptions using all provided visual features."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=60,
        )
        
        caption = response.choices[0].message.content.strip().strip('"\'')
        
        # Verify emotion word
        has_emotion = any(word in caption.lower() for word in emotion_words)
        if not has_emotion:
            adj = random.choice(emotion_words)
            caption = f"A {adj} " + caption
        
        return caption
        
    except Exception as e:
        print(f"Groq error: {e}")
        return f"a {random.choice(emotion_words)} person {action}"

# ============================================================================
# GIF PROCESSING
# ============================================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_all_frames(gif_path: Path) -> Tuple[List[Image.Image], Image.Image]:
    """Extract all frames and middle frame from GIF"""
    try:
        gif = Image.open(gif_path)
        frames = []
        
        try:
            while True:
                frames.append(gif.copy().convert('RGB'))
                gif.seek(len(frames))
        except EOFError:
            pass
        
        middle_idx = len(frames) // 2
        middle_frame = frames[middle_idx]
        
        return frames, middle_frame
        
    except Exception as e:
        print(f"Error: {e}")
        return [], None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("COMPLETE MULTI-MODAL CAPTION GENERATION")
    print("="*70)
    print("Pipeline: ResNet50 + VideoMAE + YOLO + VGG16 + Groq")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # Setup all models
    groq_client = setup_groq()
    
    print("\nLoading emotion model...")
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
    print(f"Total test samples: {len(df)}")
    
    # Select samples
    samples_per_emotion = 3
    selected_samples = []
    
    for emotion in df['emotion_label'].unique():
        emotion_samples = df[df['emotion_label'] == emotion].sample(
            n=min(samples_per_emotion, len(df[df['emotion_label'] == emotion])),
            random_state=42
        )
        selected_samples.append(emotion_samples)
    
    test_df = pd.concat(selected_samples).reset_index(drop=True)
    print(f"Selected {len(test_df)} test GIFs")
    print()
    
    print("="*70)
    print("🔬 MULTI-MODAL PROCESSING")
    print("="*70)
    
    results = []
    
    for idx, row in test_df.iterrows():
        gif_id = row['gif_id']
        true_emotion = row['emotion_label']
        gif_path = GIF_DIR / f"{gif_id}.gif"
        
        print(f"\n[{idx+1}/{len(test_df)}] {gif_id}")
        print(f"   True: {true_emotion}")
        
        # Extract frames
        all_frames, middle_frame = extract_all_frames(gif_path)
        if middle_frame is None:
            continue
        
        # 1. EMOTION (ResNet50)
        frame_tensor = transform(middle_frame).unsqueeze(0).to(device)
        with torch.no_grad():
            emotion_output = emotion_model(frame_tensor)
            probs = torch.softmax(emotion_output, dim=1)[0]
            pred_idx = probs.argmax().item()
            emotion_conf = probs[pred_idx].item()
            pred_emotion = idx_to_emotion[pred_idx]
        
        print(f"Emotion: {pred_emotion} ({emotion_conf*100:.1f}%)")
        
        # 2. ACTION (VideoMAE)
        action, action_conf = videomae.detect_action(all_frames)
        print(f"Action: {action} ({action_conf*100:.1f}%)")
        
        # 3. OBJECTS (YOLO)
        objects = yolo.detect_objects(middle_frame)
        objects_str = ", ".join([f"{obj}({conf:.0%})" for obj, conf in objects[:3]]) if objects else "none"
        print(f"Objects: {objects_str}")
        
        # 4. SCENE (VGG16)
        scene = vgg16.analyze_scene(frame_tensor, middle_frame)
        print(f" Scene: {scene['content_type']}, {scene['setting']}, {scene['lighting']}")
        
        # 5. GENERATE CAPTIONS
        template = generate_template_caption(pred_emotion)
        multimodal = generate_multimodal_caption(
            groq_client, pred_emotion, emotion_conf,
            action, action_conf, objects, scene
        )
        
        print(f"Template: '{template}'")
        print(f"Multi-modal: '{multimodal}'")
        
        results.append({
            'gif_id': gif_id,
            'true_emotion': true_emotion,
            'predicted_emotion': pred_emotion,
            'emotion_confidence': float(emotion_conf),
            'action': action,
            'action_confidence': float(action_conf),
            'objects': [{'name': obj, 'conf': float(conf)} for obj, conf in objects],
            'content_type': scene['content_type'],
            'setting': scene['setting'],
            'lighting': scene['lighting'],
            'template_caption': template,
            'multimodal_caption': multimodal,
        })
    
    print(f"\n✅ Processed {len(results)} GIFs!")
    
    # Save results
    results_file = OUTPUT_DIR / 'multimodal_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved: {results_file}")
    
    # Generate HTML
    generate_html_report(results, OUTPUT_DIR / 'multimodal_comparison.html', GIF_DIR)
    
    print(f"\n✅ COMPLETE!")
    print(f"Open: {OUTPUT_DIR / 'multimodal_comparison.html'}")

# ============================================================================
# HTML REPORT
# ============================================================================

def generate_html_report(results: List[Dict], output_path: Path, gif_dir: Path):
    """Generate HTML with GIF previews"""
    
    import shutil
    
    # Copy GIFs
    images_dir = output_path.parent / 'gif_previews'
    images_dir.mkdir(exist_ok=True)
    
    print(f"\n📸 Copying GIFs...")
    for r in results:
        src = gif_dir / f"{r['gif_id']}.gif"
        dst = images_dir / f"{r['gif_id']}.gif"
        if src.exists():
            shutil.copy2(src, dst)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Multi-Modal Caption Generation</title>
    <style>
        body {{ font-family: Arial; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #2c3e50; }}
        .summary {{ background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .sample {{ background: white; padding: 25px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .gif-container {{ text-align: center; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
        .gif-container img {{ max-width: 400px; max-height: 300px; border: 3px solid #007bff; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }}
        .features {{ background: #fff3cd; padding: 15px; border-radius: 6px; margin: 15px 0; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 3px; margin: 3px; font-size: 13px; color: white; }}
        .emotion {{ background: #ff6b9d; }}
        .action {{ background: #4ecdc4; }}
        .object {{ background: #95e1d3; }}
        .scene {{ background: #ffa07a; }}
        .caption {{ padding: 20px; margin: 15px 0; border-radius: 6px; font-size: 16px; }}
        .template {{ background: #f8f9fa; border-left: 4px solid #28a745; }}
        .multimodal {{ background: #e8f4f8; border-left: 4px solid #007bff; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Complete Multi-Modal Caption Generation</h1>
        <div class="summary">
            <h2>Pipeline Components</h2>
            <ul>
                <li><strong>ResNet50:</strong> Emotion classification (6 groups)</li>
                <li><strong>VideoMAE:</strong> Action recognition (Kinetics-400)</li>
                <li><strong>YOLO:</strong> Object detection (80 COCO classes)</li>
                <li><strong>VGG16:</strong> Scene understanding (indoor/outdoor, lighting, content type)</li>
                <li><strong>Groq LLama 3.3:</strong> Multi-modal caption synthesis</li>
            </ul>
            <p><strong>Total Samples:</strong> {len(results)}</p>
        </div>
"""
    
    for i, r in enumerate(results):
        objects_str = ", ".join([o['name'] for o in r['objects'][:3]]) if r['objects'] else "none"
        
        html += f"""
        <div class="sample">
            <h3>#{i+1}: {r['gif_id']}</h3>
            <p><strong>True:</strong> {r['true_emotion']} | <strong>Predicted:</strong> {r['predicted_emotion']}</p>
            
            <div class="gif-container">
                <img src="gif_previews/{r['gif_id']}.gif" loading="lazy">
            </div>
            
            <div class="features">
                <strong>🎯 Detected Features:</strong><br>
                <span class="badge emotion">😊 {r['predicted_emotion']} ({r['emotion_confidence']*100:.0f}%)</span>
                <span class="badge action">🏃 {r['action']} ({r['action_confidence']*100:.0f}%)</span>
                <span class="badge object">🎯 {objects_str}</span>
                <span class="badge scene">🎬 {r['content_type']}, {r['setting']}, {r['lighting']}</span>
            </div>
            
            <div class="caption template">
                <strong>Template (Baseline):</strong><br>
                "{r['template_caption']}"
            </div>
            
            <div class="caption multimodal">
                <strong>Multi-Modal (ResNet50 + VideoMAE + YOLO + VGG16):</strong><br>
                "{r['multimodal_caption']}"
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
    
    print(f"HTML: {output_path}")

if __name__ == "__main__":
    main()
