"""
SENTIVUE BACKEND - BASELINE vs ENHANCED COMPARISON
===================================================

Features:
- Baseline mode: Emotion + YOLO + VideoMAE + Templates (current system)
- Enhanced mode: + VGG16 scene + Groq LLM (proposed improvement)
- Toggle between modes via API parameter
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
import random
import io
import cv2
import numpy as np
from typing import List, Optional, Dict
import os

# Optional Groq (for enhanced mode)
try:
    from groq import Groq
    GROQ_AVAILABLE = os.getenv('GROQ_API_KEY') is not None
    if GROQ_AVAILABLE:
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
except:
    GROQ_AVAILABLE = False

# Optional VideoMAE
try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    VIDEOMAE_AVAILABLE = True
except:
    VIDEOMAE_AVAILABLE = False

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(title="SentiVue API - Comparison", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURATION
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")

emotion_groups = ['contempt', 'negative_intense', 'negative_subdued',
                 'positive_calm', 'positive_energetic', 'surprise']

emotion_vocabulary = {
    'positive_energetic': {
        'adjectives': ['joyful', 'happy', 'excited', 'cheerful', 'enthusiastic', 'energetic', 'elated', 'thrilled'],
        'verbs': ['dancing', 'jumping', 'celebrating', 'cheering', 'laughing', 'playing', 'running', 'clapping']
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
        'adjectives': ['contemptuous', 'disdainful', 'scornful', 'dismissive', 'mocking', 'sneering'],
        'verbs': ['dismissing', 'ignoring', 'mocking', 'scoffing', 'rejecting', 'sneering', 'rolling eyes']
    }
}

# ============================================================
# MODELS
# ============================================================

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
emotion_model.load_state_dict(torch.load('models/best_model_grouped.pth', map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()
print("✅ Emotion model loaded!")

print("📦 Loading YOLO...")
object_detector = YOLO('yolov8n.pt')
print("✅ YOLO loaded!")

# VideoMAE (optional)
if VIDEOMAE_AVAILABLE:
    try:
        print("📦 Loading VideoMAE...")
        videomae_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        videomae_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        videomae_model.eval()
        print("✅ VideoMAE loaded!")
    except:
        VIDEOMAE_AVAILABLE = False
        print("⚠️  VideoMAE loading failed")

# VGG16 (for enhanced mode)
VGG16_AVAILABLE = True
try:
    print("📦 Loading VGG16...")
    vgg16_full = models.vgg16(pretrained=True).to(device)
    vgg16_full.eval()
    vgg16_conv = vgg16_full.features
    vgg16_fc = nn.Sequential(*list(vgg16_full.classifier.children())[:-1])
    print("✅ VGG16 loaded!")
except:
    VGG16_AVAILABLE = False
    print("⚠️  VGG16 loading failed")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_middle_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    """Extract middle frame"""
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        n_frames = 0
        try:
            while True:
                gif.seek(n_frames)
                n_frames += 1
        except EOFError:
            pass
        
        gif.seek(n_frames // 2)
        return gif.convert('RGB')
    except:
        return None

def detect_emotion(frame: Image.Image) -> tuple[str, float]:
    """Detect emotion"""
    try:
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = emotion_model(frame_tensor)
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, idx].item()
        
        return emotion_groups[idx], confidence
    except:
        return 'positive_energetic', 0.3

def detect_objects(frame: Image.Image) -> List[str]:
    """Detect objects with YOLO"""
    try:
        results = object_detector(frame, verbose=False)
        objects = []
        
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf >= 0.2:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    objects.append((class_name, conf))
        
        objects.sort(key=lambda x: x[1], reverse=True)
        return [obj[0] for obj in objects[:5]]
    except:
        return []

def extract_frames_for_action(gif_bytes: bytes, num_frames: int = 16) -> Optional[List[Image.Image]]:
    """Extract frames for VideoMAE"""
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        
        n_frames = 0
        try:
            while True:
                gif.seek(n_frames)
                n_frames += 1
        except EOFError:
            pass
        
        indices = np.linspace(0, n_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            gif.seek(int(idx))
            frames.append(gif.convert('RGB'))
        
        return frames
    except:
        return None

def detect_action(gif_bytes: bytes) -> Optional[str]:
    """Detect action with VideoMAE"""
    if not VIDEOMAE_AVAILABLE:
        return None
    
    try:
        frames = extract_frames_for_action(gif_bytes)
        if not frames:
            return None
        
        inputs = videomae_processor(frames, return_tensors="pt")
        
        with torch.no_grad():
            outputs = videomae_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            top_idx = probs.argmax().item()
            confidence = probs[top_idx].item()
        
        if confidence > 0.15:
            action = videomae_model.config.id2label[top_idx]
            return action.replace('(', '').replace(')', '').strip().lower()
        
        return None
    except:
        return None

def analyze_scene_vgg16(frame: Image.Image) -> Dict[str, str]:
    """Analyze scene with VGG16"""
    if not VGG16_AVAILABLE:
        return {'content_type': 'unknown', 'setting': 'unknown', 'lighting': 'unknown'}
    
    try:
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            conv_features = vgg16_conv(frame_tensor)
            final_features = vgg16_fc(conv_features.view(1, -1)).cpu().numpy()[0]
        
        img_array = np.array(frame.convert('RGB'))
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Content type
        saturation = hsv[:, :, 1].mean()
        unique_colors = len(np.unique(img_array.reshape(-1, 3), axis=0))
        color_ratio = unique_colors / (img_array.shape[0] * img_array.shape[1])
        content_type = 'animated' if (saturation > 100 and color_ratio < 0.3) else 'real'
        
        # Setting
        mean_activation = final_features.mean()
        mean_brightness = img_array.mean()
        setting = 'outdoor' if (mean_activation > 0.2 and mean_brightness > 120) else 'indoor'
        
        # Lighting
        brightness = img_array.mean()
        lighting = 'bright' if brightness > 140 else ('dim' if brightness < 80 else 'moderate')
        
        return {
            'content_type': content_type,
            'setting': setting,
            'lighting': lighting
        }
    except:
        return {'content_type': 'unknown', 'setting': 'unknown', 'lighting': 'unknown'}

def generate_baseline_caption(emotion: str, objects: List[str], action: Optional[str]) -> str:
    """
    BASELINE: Template system (current production)
    """
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    adj = random.choice(vocab['adjectives'])
    
    # Use detected action or fallback
    if action:
        action_words = action.lower().split()
        verb = ' '.join(action_words[:2]) if len(action_words) > 2 else action.lower()
        
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
    
    # Smart object prioritization
    if objects:
        animals = ['dog', 'cat', 'bird', 'horse', 'bear', 'elephant', 'giraffe']
        detected_animals = [o for o in objects if o.lower() in animals]
        interesting_objects = [o for o in objects if o.lower() not in ['person', 'man', 'woman', 'people']]
        has_person = any(o.lower() in ['person', 'man', 'woman', 'people'] for o in objects)
        
        if detected_animals:
            obj = random.choice(detected_animals)
            templates = [
                f"a {adj} person with a {obj}",
                f"someone {adj}ly {verb} with a {obj}",
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
            templates = [f"a {adj} scene"]
    else:
        templates = [
            f"a {adj} person {verb}",
            f"someone {verb} {adj}ly",
            f"someone feeling {adj} while {verb}",
        ]
    
    return random.choice(templates)

def generate_enhanced_caption(
    emotion: str,
    objects: List[str],
    action: Optional[str],
    scene: Dict[str, str]
) -> str:
    """
    ENHANCED: Baseline + VGG16 scene + Groq LLM
    """
    if not GROQ_AVAILABLE:
        # Fallback to baseline
        return generate_baseline_caption(emotion, objects, action)
    
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    emotion_words = vocab['adjectives'][:5]
    
    objects_str = ", ".join(objects[:3]) if objects else "none"
    action_str = action if action else "general movement"
    
    prompt = f"""Generate a natural GIF caption.

Features:
- Emotion: {emotion} (use: {', '.join(emotion_words)})
- Action: {action_str}
- Objects: {objects_str}
- Scene: {scene['content_type']}, {scene['setting']}, {scene['lighting']}

Rules:
1. Include ONE emotion word
2. Mention action if provided
3. Reference content type (animated/real)
4. Include scene context
5. Max 20 words, natural

Caption only (no quotes):"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Create vivid GIF captions."},
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
    except:
        # Fallback
        return generate_baseline_caption(emotion, objects, action)

# ============================================================
# RESPONSE MODEL
# ============================================================

class CaptionResponse(BaseModel):
    emotion: str
    confidence: float
    caption: str  # The caption to display (baseline or enhanced based on mode)
    objects: List[str]
    action: Optional[str]
    scene: Optional[dict] = None  # VGG16 scene info (if enhanced mode)
    content_warning: str = ""
    # Optional comparison data (for testing/development)
    baseline_caption: Optional[str] = None
    enhanced_caption: Optional[str] = None
    mode: str = "baseline"

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "name": "SentiVue API - Comparison Mode",
        "version": "3.0.0",
        "features": {
            "emotion_detection": True,
            "object_detection": True,
            "action_detection": VIDEOMAE_AVAILABLE,
            "vgg16_scene": VGG16_AVAILABLE,
            "groq_llm": GROQ_AVAILABLE
        }
    }

@app.post("/analyze", response_model=CaptionResponse)
async def analyze_gif(
    file: UploadFile = File(...),
    mode: str = "baseline"  # "baseline" or "enhanced"
):
    """
    Analyze GIF and generate caption
    
    mode="baseline": Template system (current - default)
    mode="enhanced": + VGG16 scene + Groq LLM
    
    Returns the caption in the 'caption' field based on mode
    """
    
    try:
        gif_bytes = await file.read()
        
        # Extract middle frame
        frame = extract_middle_frame(gif_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process GIF")
        
        # 1. Emotion (always)
        emotion, confidence = detect_emotion(frame)
        
        # 2. Objects (always)
        objects = detect_objects(frame)
        
        # 3. Action (if available)
        action = detect_action(gif_bytes) if VIDEOMAE_AVAILABLE else None
        
        # 4. Generate baseline caption (always)
        baseline_caption = generate_baseline_caption(emotion, objects, action)
        
        # 5. Scene + enhanced caption (if enhanced mode)
        scene = None
        enhanced_caption = None
        
        if mode == "enhanced":
            if VGG16_AVAILABLE:
                scene = analyze_scene_vgg16(frame)
            if scene and GROQ_AVAILABLE:
                enhanced_caption = generate_enhanced_caption(emotion, objects, action, scene)
        
        # Determine which caption to return as main caption
        final_caption = enhanced_caption if (mode == "enhanced" and enhanced_caption) else baseline_caption
        
        # Content warning (empty for now, can add logic later)
        content_warning = ""
        
        return CaptionResponse(
            emotion=emotion,
            confidence=confidence,
            caption=final_caption,  # Frontend uses this!
            objects=objects,
            action=action,
            scene=scene,
            content_warning=content_warning,
            baseline_caption=baseline_caption,  # For comparison
            enhanced_caption=enhanced_caption,  # For comparison
            mode=mode
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_caption_endpoint(
    file: UploadFile = File(...),
    mode: str = "baseline"
):
    """
    Frontend compatibility endpoint - same as /analyze but returns simplified response
    This matches your original main_2.py /generate endpoint
    """
    
    try:
        # Call the main analyze function
        result = await analyze_gif(file, mode)
        
        # Return simplified format for frontend
        return {
            "emotion": result.emotion,
            "confidence": result.confidence,
            "caption": result.caption,
            "content_warning": result.content_warning,
            "objects": result.objects,
            "action": result.action
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-both")
async def analyze_both_modes(file: UploadFile = File(...)):
    """
    Analyze with BOTH baseline and enhanced modes for comparison
    """
    
    try:
        gif_bytes = await file.read()
        
        frame = extract_middle_frame(gif_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process GIF")
        
        # Detect features
        emotion, confidence = detect_emotion(frame)
        objects = detect_objects(frame)
        action = detect_action(gif_bytes) if VIDEOMAE_AVAILABLE else None
        scene = analyze_scene_vgg16(frame) if VGG16_AVAILABLE else None
        
        # Generate both captions
        baseline = generate_baseline_caption(emotion, objects, action)
        enhanced = generate_enhanced_caption(emotion, objects, action, scene) if scene and GROQ_AVAILABLE else None
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "objects": objects,
            "action": action,
            "scene": scene,
            "baseline_caption": baseline,
            "enhanced_caption": enhanced,
            "improvement": enhanced if enhanced else "Groq not available"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)