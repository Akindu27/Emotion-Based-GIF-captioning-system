# ============================================================
# SENTIVUE BACKEND - FINAL (CAPTIONING)
# Grouped Emotion (ResNet50) + Multi-frame Objects + Person Count + Lighting
# ============================================================

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image, ImageSequence
from ultralytics import YOLO

import random
import io
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Set

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(title="SentiVue API - Final", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ============================================================
# EMOTION LABELS (GROUPED MODEL ORDER)
# IMPORTANT: This must match training label order for best_model_grouped.pth
# ============================================================

emotion_groups = [
    "contempt",
    "negative_intense",
    "negative_subdued",
    "positive_calm",
    "positive_energetic",
    "surprise",
]

emotion_vocabulary = {
    "positive_energetic": {
        "adjectives": ["joyful", "happy", "excited", "cheerful", "enthusiastic", "energetic", "elated", "thrilled", "ecstatic", "delighted"],
        "verbs": ["dancing", "jumping", "celebrating", "cheering", "laughing", "playing", "running", "clapping", "bouncing", "spinning"],
    },
    "positive_calm": {
        "adjectives": ["peaceful", "content", "serene", "relaxed", "satisfied", "tranquil", "calm", "pleased", "gentle", "soothing"],
        "verbs": ["sitting", "resting", "smiling", "relaxing", "enjoying", "appreciating", "meditating", "breathing", "gazing", "watching"],
    },
    "negative_intense": {
        "adjectives": ["angry", "furious", "fearful", "terrified", "disgusted", "enraged", "frustrated", "scared", "horrified", "panicked"],
        "verbs": ["yelling", "screaming", "running", "fighting", "crying", "panicking", "shouting", "fleeing", "trembling", "recoiling"],
    },
    "negative_subdued": {
        "adjectives": ["sad", "sorrowful", "dejected", "gloomy", "melancholic", "somber", "depressed", "lonely", "disappointed", "downcast"],
        "verbs": ["crying", "sitting", "looking", "walking", "waiting", "sighing", "moping", "brooding", "staring", "reflecting"],
    },
    "surprise": {
        "adjectives": ["surprised", "shocked", "astonished", "amazed", "stunned", "bewildered", "startled", "astounded", "speechless", "flabbergasted"],
        "verbs": ["reacting", "jumping", "gasping", "staring", "looking", "responding", "gaping", "freezing", "stepping back", "covering mouth"],
    },
    "contempt": {
        "adjectives": ["contemptuous", "disdainful", "scornful", "dismissive", "snide", "mocking", "sneering", "arrogant", "superior", "condescending"],
        "verbs": ["dismissing", "ignoring", "mocking", "scoffing", "rejecting", "sneering", "ridiculing", "scorning", "rolling eyes", "smirking"],
    },
}

# ============================================================
# MODELS
# ============================================================

class GroupedEmotionClassifier(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

print("📦 Loading emotion detection model (grouped ResNet50)...")
emotion_model = GroupedEmotionClassifier(num_classes=6)
emotion_model.load_state_dict(torch.load("models/best_model_grouped.pth", map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()
print("✅ Emotion model loaded!")

print("📦 Loading YOLO object detector...")
object_detector = YOLO("models/yolov8n.pt")
print("✅ YOLO loaded!")

print("📦 Loading action detection model (VideoMAE)...")
try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    action_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model.to(device)
    action_model.eval()
    ACTION_DETECTION_ENABLED = True
    print("✅ VideoMAE loaded!")
except Exception as e:
    print(f"⚠️  VideoMAE not available: {e}")
    ACTION_DETECTION_ENABLED = False

# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# HELPERS
# ============================================================

def extract_middle_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        frames = [fr.convert("RGB") for fr in ImageSequence.Iterator(gif)]
        if not frames:
            return None
        return frames[len(frames) // 2]
    except Exception as e:
        print(f"❌ Error extracting middle frame: {e}")
        return None

def extract_k_frames_evenly(gif_bytes: bytes, k: int = 8) -> List[Image.Image]:
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        frames = [fr.convert("RGB") for fr in ImageSequence.Iterator(gif)]
        if not frames:
            return []
        if k <= 1:
            return [frames[len(frames)//2]]
        idxs = np.linspace(0, len(frames) - 1, k, dtype=int)
        return [frames[i] for i in idxs]
    except Exception as e:
        print(f"❌ Error extracting frames: {e}")
        return []

@torch.no_grad()
def detect_emotion(frame: Image.Image) -> Tuple[str, float]:
    try:
        x = transform(frame).unsqueeze(0).to(device)
        out = emotion_model(x)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        return emotion_groups[idx], conf
    except Exception as e:
        print(f"❌ Error detecting emotion: {e}")
        return "positive_energetic", 0.30

def detect_content_type(frame: Image.Image) -> Tuple[str, float]:
    """
    Improved heuristic using face detection, edge complexity, and color variance.
    Real-world GIFs (even compressed ones from Giphy) have complex edges and
    natural color gradients that cartoons lack.
    """
    try:
        img_array = np.array(frame)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        # --- Face detection (strong real_world signal) ---
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0

        # --- Color ratio (unique colors / total pixels) ---
        # NOTE: Web GIFs are palettized to 256 colors max, so real-world GIFs will
        # also have a low color_ratio. We use a much stricter threshold.
        img_flat = img_array.reshape(-1, img_array.shape[2])
        unique_colors = len(np.unique(img_flat, axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_ratio = unique_colors / max(total_pixels, 1)

        # --- Edge complexity (Laplacian variance) ---
        # Real-world images have far more edge detail than flat cartoons
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # --- Color variance (standard deviation across channels) ---
        # Real-world images have high local color variance; cartoons are flat
        color_std = float(np.std(img_array.astype(np.float32)))

        # --- Saturation ---
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

        print(f"🔍 Content Detection - ColorRatio: {color_ratio:.4f}, Sat: {saturation:.3f}, "
              f"Face: {has_face}, EdgeVar: {lap_var:.1f}, ColorStd: {color_std:.1f}")

        # Face detected → almost certainly real_world
        if has_face:
            return "real_world", 0.95

        # High edge complexity or high color std → real_world
        # (real-world GIFs from Giphy have lap_var >> 100 and color_std >> 40)
        if lap_var > 200 or color_std > 50:
            return "real_world", 0.85

        # Only classify as cartoon if edges are very smooth AND colors are flat
        # Tighten thresholds so real-world GIFs aren't caught here
        if color_ratio < 0.003 and saturation > 0.45 and lap_var < 100 and color_std < 35:
            return "cartoon", 0.88

        if color_ratio < 0.001 and lap_var < 80:
            return "cartoon", 0.80

        # Default to real_world
        return "real_world", 0.65

    except Exception as e:
        print(f"❌ Error detecting content type: {e}")
        return "real_world", 0.50

def analyze_lighting(frame: Image.Image) -> Dict[str, float | str]:
    """
    Returns simple brightness stats for 'scene understanding' without extra models:
    - brightness: mean grayscale intensity (0-255)
    - lighting_label: dim / moderate / bright
    """
    try:
        img = np.array(frame.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        b = float(np.mean(gray))

        if b < 80:
            label = "dim"
        elif b > 140:
            label = "bright"
        else:
            label = "moderate"

        return {"brightness": b, "lighting_label": label}
    except Exception as e:
        print(f"❌ Lighting analysis error: {e}")
        return {"brightness": -1.0, "lighting_label": "unknown"}

def detect_objects_multiframe_vote(
    gif_bytes: bytes,
    k: int = 8,
    top_n: int = 2,
    min_votes: int = 2,
    conf_thresh: float = 0.20,
    drop_labels: Optional[Set[str]] = None,
) -> List[str]:
    """
    Vote objects across multiple frames to improve recall.
    Removes generic human labels.
    """
    if drop_labels is None:
        drop_labels = {"person", "man", "woman", "people", "human"}

    frames = extract_k_frames_evenly(gif_bytes, k=k)
    if not frames:
        return []

    from collections import Counter
    all_labels: List[str] = []

    for fr in frames:
        objs: List[Tuple[str, float]] = []
        try:
            results = object_detector(fr, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = object_detector.names.get(cls_id, str(cls_id)).strip().lower()
                    if name and name not in drop_labels:
                        objs.append((name, conf))
        except Exception as e:
            print(f"❌ YOLO frame error: {e}")
            continue

        objs.sort(key=lambda x: x[1], reverse=True)
        kept = [n for (n, c) in objs if c >= conf_thresh]
        if not kept and objs:
            kept = [objs[0][0]]  # top-1 fallback
        all_labels.extend(kept[:2])

    if not all_labels:
        return []

    counts = Counter(all_labels)
    winners = [lbl for lbl, c in counts.most_common() if c >= min_votes]
    if not winners:
        winners = [lbl for lbl, _ in counts.most_common(top_n)]
    return winners[:top_n]

def count_people(frame: Image.Image, conf_thresh: float = 0.25) -> int:
    """
    Count persons in a frame using YOLO 'person' class.
    """
    try:
        results = object_detector(frame, verbose=False)
        person_count = 0
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = r.names.get(cls_id, str(cls_id)).strip().lower()
                if name == "person" and conf >= conf_thresh:
                    person_count += 1
        return person_count
    except Exception as e:
        print(f"❌ Person count error: {e}")
        return 0

def extract_frames_for_action(gif_bytes: bytes, num_frames: int = 16) -> Optional[List[Image.Image]]:
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        frames = [fr.convert("RGB") for fr in ImageSequence.Iterator(gif)]
        if not frames:
            return None
        idxs = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in idxs]
    except Exception as e:
        print(f"❌ Error extracting frames for action: {e}")
        return None

def detect_action(gif_bytes: bytes) -> Optional[str]:
    if not ACTION_DETECTION_ENABLED:
        return None
    try:
        frames = extract_frames_for_action(gif_bytes, num_frames=16)
        if not frames:
            return None

        inputs = action_processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = action_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            top_idx = int(probs.argmax().item())
            conf = float(probs[top_idx].item())

        if conf > 0.15:
            label = action_model.config.id2label[top_idx]
            return label.replace("(", "").replace(")", "").strip().lower()
        return None
    except Exception as e:
        print(f"❌ Action detection error: {e}")
        return None

def motion_based_fallback_action(gif_bytes: bytes) -> str:
    frames = extract_k_frames_evenly(gif_bytes, k=6)
    if len(frames) < 2:
        return "reacting"
    arrs = [np.asarray(f.resize((128, 128)), dtype=np.float32) for f in frames]
    diffs = []
    for a1, a2 in zip(arrs[:-1], arrs[1:]):
        diffs.append(float(np.mean(np.abs(a2 - a1)) / 255.0))
    m = float(np.mean(diffs)) if diffs else 0.0
    if m > 0.10:
        return "moving"
    if m > 0.05:
        return "gesturing"
    return "reacting"

# ============================================================
# CAPTION GENERATION (CLEAN TGIF-LIKE)
# ============================================================

def generate_caption(
    emotion: str,
    objects: Optional[List[str]] = None,
    action: Optional[str] = None,
    content_type: str = "real_world",
) -> str:
    objects = objects or []

    # Cartoon: keep short + safe (you used this earlier)
    if content_type == "cartoon":
        emotion_word = emotion.replace("_", " ")
        return f"an animated {emotion_word} moment"

    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary["positive_energetic"])
    adj = random.choice(vocab["adjectives"])

    # action phrase
    if action:
        words = action.lower().split()
        verb = " ".join(words[:2]) if len(words) > 2 else action.lower()
        if not verb.endswith("ing"):
            if " " in verb:
                parts = verb.split()
                if not parts[0].endswith("ing"):
                    parts[0] = parts[0] + "ing"
                verb = " ".join(parts)
            else:
                verb = verb + "ing"
    else:
        verb = random.choice(vocab["verbs"])

    # object selection
    if objects:
        animals = {"dog", "cat", "bird", "horse", "bear", "elephant", "giraffe", "zebra", "lion", "tiger", "cow", "sheep", "monkey", "rabbit"}
        interesting = [o for o in objects if o.lower() not in {"person", "man", "woman", "people", "human"}]
        detected_animals = [o for o in interesting if o.lower() in animals]

        if detected_animals:
            obj = random.choice(detected_animals)
            templates = [
                f"a {adj} person {verb} with a {obj}",
                f"someone {adj}ly {verb} with a {obj}",
            ]
        elif interesting:
            obj = random.choice(interesting)
            templates = [
                f"a {adj} person {verb} with a {obj}",
                f"someone {adj}ly {verb} near a {obj}",
            ]
        else:
            templates = [
                f"a {adj} person {verb}",
                f"someone feeling {adj} while {verb}",
            ]
    else:
        templates = [
            f"a {adj} person {verb}",
            f"someone feeling {adj} while {verb}",
        ]

    caption = random.choice(templates)
    print(f"💬 Generated caption: '{caption}' (emotion={emotion}, objects={objects}, action={action}, type={content_type})")
    return caption

# ============================================================
# RESPONSE MODEL
# ============================================================

class CaptionResponse(BaseModel):
    emotion: str
    caption: str
    confidence: Optional[float] = None
    objects: Optional[List[str]] = None
    action: Optional[str] = None
    content_type: Optional[str] = None
    content_warning: Optional[str] = None
    person_count: Optional[int] = None
    lighting: Optional[dict] = None  # {"brightness":..., "lighting_label":...}

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "SentiVue API - Final",
        "version": "2.1.0",
        "features": {
            "emotion_grouped": True,
            "object_multiframe": True,
            "person_count": True,
            "lighting": True,
            "action_detection": ACTION_DETECTION_ENABLED,
            "content_filtering": True,
            "depth_in_backend": False,  # handled in separate notebook
        },
    }

@app.post("/generate", response_model=CaptionResponse)
async def generate_gif_caption(file: UploadFile = File(...)):
    try:
        gif_bytes = await file.read()

        frame = extract_middle_frame(gif_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process GIF")

        # Content type
        content_type, _ = detect_content_type(frame)

        # Lighting analysis (always)
        lighting = analyze_lighting(frame)

        # Emotion (always)
        emotion, emotion_conf = detect_emotion(frame)

        # Person count (frame-based)
        person_count = count_people(frame, conf_thresh=0.25)

        # Multi-frame objects vote
        objects = detect_objects_multiframe_vote(
            gif_bytes, k=8, top_n=2, min_votes=2, conf_thresh=0.20
        )

        # Action (real-world only)
        action = None
        if content_type == "real_world":
            action = detect_action(gif_bytes)
            if action is None:
                action = motion_based_fallback_action(gif_bytes)

        # Caption
        caption = generate_caption(emotion, objects, action, content_type)

        # Warning logic (minimal)
        content_warning = None
        if content_type == "cartoon":
            content_warning = "Animated/cartoon content detected; emotion/action may be less reliable."
        elif emotion_conf < 0.25:
            content_warning = "Low confidence emotion detection; caption may be less accurate."

        return CaptionResponse(
            emotion=emotion,
            caption=caption,
            confidence=emotion_conf,
            objects=objects,
            action=action,
            content_type=content_type,
            content_warning=content_warning,
            person_count=person_count,
            lighting=lighting,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models": {
            "emotion": "loaded",
            "objects": "loaded",
            "action": "loaded" if ACTION_DETECTION_ENABLED else "disabled",
        },
        "emotion_groups": emotion_groups,
    }

# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🎬 SENTIVUE BACKEND API - FINAL v2.1")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Emotion groups: {emotion_groups}")
    print(f"Action detection: {'✅ Enabled' if ACTION_DETECTION_ENABLED else '⚠️ Disabled'}")
    print("Content filtering: ✅ Enabled")
    print("Object voting: ✅ Multi-frame")
    print("Person count: ✅ YOLO person class")
    print("Lighting: ✅ Brightness heuristic")
    print("Depth: ❌ (use separate notebook)")
    print("=" * 60)
    print("\n🚀 Starting server on http://127.0.0.1:8000")
    print("📖 API docs: http://127.0.0.1:8000/docs\n")

    uvicorn.run(app, host="127.0.0.1", port=8000)