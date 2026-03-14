# ============================================================
# SENTIVUE BACKEND - IMPROVED VERSION WITH CONTENT FILTERING
# ============================================================

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
from typing import List, Optional

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(title="SentiVue API - Improved", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEVICE & CONFIGURATION
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# Emotion groups
emotion_groups = ['contempt', 'negative_intense', 'negative_subdued',
                 'positive_calm', 'positive_energetic', 'surprise']

# Emotion vocabulary (expanded)
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

# ============================================================
# EMOTION MODEL
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

print("📦 Loading emotion detection model...")
emotion_model = GroupedEmotionClassifier(num_classes=6)
emotion_model.load_state_dict(torch.load('models/best_model_grouped.pth', map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()
print("✅ Emotion model loaded!")

print("📦 Loading object detection model...")
object_detector = YOLO('models/yolov8n.pt')
print("✅ Object detector loaded!")

# Action detection
print("📦 Loading action detection model...")
try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    action_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model.eval()
    ACTION_DETECTION_ENABLED = True
    print("✅ Action detector loaded!")
except Exception as e:
    print(f"⚠️  Action detector not available: {e}")
    ACTION_DETECTION_ENABLED = False

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# CONTENT TYPE DETECTION
# ============================================================

def detect_content_type(frame: Image.Image) -> tuple[str, float]:
    """
    Detect if GIF is real-world or cartoon/anime
    Returns: (content_type, confidence)
    """
    try:
        # Convert to numpy array
        img_array = np.array(frame)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect edges (cartoons have more sharp edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        
        # Count unique colors (cartoons have fewer)
        img_flat = img_array.reshape(-1, img_array.shape[2])
        unique_colors = len(np.unique(img_flat, axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_ratio = unique_colors / total_pixels
        
        # Calculate saturation (cartoons often more saturated)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = np.mean(hsv[:, :, 1]) / 255.0
        
        # Debug logging
        print(f"🔍 Content Detection - Edge: {edge_ratio:.3f}, Color: {color_ratio:.3f}, Sat: {saturation:.3f}")
        
        # Scoring system (adjusted thresholds to be less strict)
        cartoon_score = 0
        
        # Very high edge ratio suggests cartoon (increased threshold)
        if edge_ratio > 0.18:  # Was 0.12, now stricter
            cartoon_score += 1
        
        # Very low color diversity suggests cartoon (decreased threshold)
        if color_ratio < 0.2:  # Was 0.3, now stricter
            cartoon_score += 1
        
        # Very high saturation suggests cartoon (increased threshold)
        if saturation > 0.65:  # Was 0.5, now stricter
            cartoon_score += 1
        
        # Determine content type - require all 3 indicators for cartoon
        if cartoon_score >= 3:  # Was 2, now requires all 3
            confidence = cartoon_score / 3.0
            return "cartoon", confidence
        else:
            confidence = (3 - cartoon_score) / 3.0
            return "real_world", confidence
            
    except Exception as e:
        print(f"❌ Error detecting content type: {e}")
        return "unknown", 0.5

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_middle_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    """Extract middle frame from GIF bytes"""
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
    except Exception as e:
        print(f"❌ Error extracting frame: {e}")
        return None

def detect_emotion(frame: Image.Image) -> tuple[str, float]:
    """Detect emotion from frame"""
    try:
        frame_tensor = transform(frame).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = emotion_model(frame_tensor)
            probs = torch.softmax(output, dim=1)
            idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, idx].item()
        
        emotion = emotion_groups[idx]
        return emotion, confidence
    except Exception as e:
        print(f"❌ Error detecting emotion: {e}")
        return 'positive_energetic', 0.3

def detect_objects(frame: Image.Image) -> List[str]:
    """Detect objects in frame with lower threshold for better detection"""
    try:
        results = object_detector(frame, verbose=False)
        objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0])
                # Lower threshold for better animal detection
                if conf >= 0.2:  # Was 0.3
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    objects.append((class_name, conf))
        
        # Sort by confidence and return top 5
        objects.sort(key=lambda x: x[1], reverse=True)
        return [obj[0] for obj in objects[:5]]
    except Exception as e:
        print(f"❌ Error detecting objects: {e}")
        return []

def extract_frames_for_action(gif_bytes: bytes, num_frames: int = 16) -> Optional[List[Image.Image]]:
    """Extract evenly spaced frames for action detection"""
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
            frame = gif.convert('RGB')
            frames.append(frame)
        
        return frames
    except Exception as e:
        print(f"❌ Error extracting frames for action: {e}")
        return None

def detect_action(gif_bytes: bytes) -> Optional[str]:
    """Detect action in GIF using VideoMAE"""
    if not ACTION_DETECTION_ENABLED:
        return None
    
    try:
        frames = extract_frames_for_action(gif_bytes, num_frames=16)
        if frames is None:
            return None
        
        inputs = action_processor(frames, return_tensors="pt")
        
        with torch.no_grad():
            outputs = action_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = probs.argmax().item()
            confidence = probs[top_idx].item()
        
        if confidence > 0.15:
            action_name = action_model.config.id2label[top_idx]
            action_name = action_name.replace('(', '').replace(')', '').strip()
            return action_name
        
        return None
    except Exception as e:
        print(f"❌ Error detecting action: {e}")
        return None

def generate_caption(emotion: str, objects: Optional[List[str]] = None, action: Optional[str] = None, content_type: str = "real_world") -> str:
    """Generate emotion-aware caption with smart object and action handling"""
    
    # If cartoon/anime, use generic caption
    if content_type == "cartoon":
        emotion_word = emotion.replace('_', ' ')
        return f"an animated {emotion_word} moment"
    
    # For real-world content, ALWAYS use full emotion-based caption
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary['positive_energetic'])
    adj = random.choice(vocab['adjectives'])
    
    # Use detected action or fall back to emotion vocabulary
    if action:
        action_words = action.lower().split()
        if len(action_words) > 2:
            verb = ' '.join(action_words[:2])
        else:
            verb = action.lower()
        
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
        # Priority 1: Animals
        animals = ['dog', 'cat', 'bird', 'horse', 'bear', 'elephant', 'giraffe', 
                   'zebra', 'lion', 'tiger', 'cow', 'sheep', 'monkey', 'rabbit']
        detected_animals = [o for o in objects if o.lower() in animals]
        
        # Priority 2: Interesting objects (not person)
        interesting_objects = [o for o in objects if o.lower() not in 
                             ['person', 'man', 'woman', 'people', 'human']]
        
        # Priority 3: Check for person
        has_person = any(o.lower() in ['person', 'man', 'woman', 'people'] for o in objects)
        
        if detected_animals:
            obj = random.choice(detected_animals)
            templates = [
                f"a {adj} person with a {obj}",
                f"someone {adj}ly {verb} with a {obj}",
                f"a {adj} person {verb} with a {obj}",
                f"a {adj} moment with a {obj}",
            ]
        elif interesting_objects:
            obj = random.choice(interesting_objects)
            templates = [
                f"a {adj} person {verb} with a {obj}",
                f"someone {adj}ly {verb} near a {obj}",
                f"a {adj} person {verb} beside a {obj}",
            ]
        elif has_person:
            templates = [
                f"a {adj} person {verb}",
                f"someone {verb} {adj}ly",
                f"a {adj} person {verb}",
                f"someone feeling {adj} while {verb}",
            ]
        else:
            templates = [
                f"a {adj} scene",
                f"an {adj} moment",
                f"something {adj} happening",
            ]
    else:
        templates = [
            f"a {adj} person {verb}",
            f"someone {verb} {adj}ly",
            f"a {adj} person {verb}",
            f"an animated scene of a {adj} person {verb}",
            f"someone feeling {adj} while {verb}",
        ]
    
    caption = random.choice(templates)
    print(f"💬 Generated caption: '{caption}' (emotion: {emotion}, objects: {objects}, action: {action})")
    return caption

# ============================================================
# RESPONSE MODEL
# ============================================================

class CaptionResponse(BaseModel):
    emotion: str
    caption: str
    confidence: Optional[float] = None
    objects: Optional[List[str]] = None
    content_type: Optional[str] = None
    content_warning: Optional[str] = None

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SentiVue API - Improved",
        "version": "2.0.0",
        "features": {
            "emotion_detection": True,
            "object_detection": True,
            "action_detection": ACTION_DETECTION_ENABLED,
            "content_filtering": True
        }
    }

@app.post("/generate", response_model=CaptionResponse)
async def generate_gif_caption(file: UploadFile = File(...)):
    """
    Generate emotion-aware caption for uploaded GIF with content type detection
    """
    try:
        # Read GIF file
        gif_bytes = await file.read()
        
        # Extract middle frame
        frame = extract_middle_frame(gif_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process GIF")
        
        # Detect content type (real-world vs cartoon)
        content_type, content_confidence = detect_content_type(frame)
        
        # Detect emotion
        emotion, emotion_confidence = detect_emotion(frame)
        
        # Generate warning for low-quality detections
        content_warning = None
        
        if content_type == "cartoon":
            content_warning = "This appears to be animated/cartoon content. Results may be less accurate."
        elif emotion_confidence < 0.25:  # Changed from 0.35 to 0.25 - only warn on very low confidence
            content_warning = "Low confidence detection. Caption may not be accurate."
        
        # Detect objects (always try)
        objects = detect_objects(frame)
        
        # Detect action (for all real-world content, skip for cartoons)
        action = None
        if content_type == "real_world":
            action = detect_action(gif_bytes)
        
        # Generate caption (always use full caption generation for real_world)
        caption = generate_caption(emotion, objects, action, content_type)
        
        # Return response
        return CaptionResponse(
            emotion=emotion,
            caption=caption,
            confidence=emotion_confidence,
            objects=objects,
            content_type=content_type,
            content_warning=content_warning
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "device": str(device),
        "models": {
            "emotion": "loaded",
            "objects": "loaded",
            "action": "loaded" if ACTION_DETECTION_ENABLED else "disabled"
        },
        "emotion_groups": emotion_groups
    }

# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🎬 SENTIVUE BACKEND API - IMPROVED v2.0")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Emotion groups: {len(emotion_groups)}")
    print(f"Action detection: {'✅ Enabled' if ACTION_DETECTION_ENABLED else '⚠️  Disabled'}")
    print(f"Content filtering: ✅ Enabled")
    print("=" * 60)
    print("\n🚀 Starting server on http://127.0.0.1:8000")
    print("📖 API docs: http://127.0.0.1:8000/docs\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)