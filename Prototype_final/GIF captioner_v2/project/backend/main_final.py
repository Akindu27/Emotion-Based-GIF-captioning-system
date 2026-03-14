"""
SentiVue Backend API
====================
Emotion-aware GIF captioning system using:
- Grouped Emotion Detection (ResNet50)
- Multi-frame Object Detection (YOLO)
- Action Recognition (VideoMAE)
- Person Counting & Lighting Analysis

Author: Akindu27
License: MIT
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

import torch
import torch.nn as nn
from torchvision import models, transforms

from PIL import Image, ImageSequence
from ultralytics import YOLO

import random
import io
import os
import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from typing import List, Optional, Tuple, Dict, Set

# ============================================================
# LOGGING SETUP
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# ENVIRONMENT & CONFIGURATION
# ============================================================

HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "Akindu27/sentivue-models")
MODELS_DIR = os.getenv("MODELS_DIR", "models")
API_VERSION = "2.1.0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ============================================================
# FASTAPI APP SETUP
# ============================================================

app = FastAPI(
    title="SentiVue API",
    description="Emotion-aware GIF captioning system",
    version=API_VERSION
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# EMOTION CONFIGURATION
# ============================================================

EMOTION_GROUPS = [
    "contempt",
    "negative_intense",
    "negative_subdued",
    "positive_calm",
    "positive_energetic",
    "surprise",
]

EMOTION_VOCABULARY = {
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
# MODEL LOADING UTILITIES
# ============================================================

def ensure_model_file(filename: str) -> str:
    """
    Download model from Hugging Face Hub if not cached locally.
    
    Args:
        filename: Name of the model file to download
        
    Returns:
        Path to the model file
        
    Raises:
        RuntimeError: If download fails
        FileNotFoundError: If file not found after download
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, filename)
    
    if os.path.exists(filepath):
        logger.info(f"Model file already cached: {filepath}")
        return filepath

    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )

    try:
        logger.info(f"Downloading {filename} from {HF_MODEL_REPO}...")
        download_kwargs = {
            "repo_id": HF_MODEL_REPO,
            "repo_type": "model",
            "filename": filename,
            "local_dir": MODELS_DIR,
        }
        if token:
            download_kwargs["token"] = token
        
        hf_hub_download(**download_kwargs)
        logger.info(f"Successfully downloaded {filename}")
    except Exception as e:
        error_msg = (
            f"Failed to download {filename}: {e}. "
            "If the repository is private, set HF_TOKEN environment variable."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found after download: {filepath}")

    return filepath


# ============================================================
# NEURAL NETWORK MODELS
# ============================================================

class GroupedEmotionClassifier(nn.Module):
    """
    ResNet50-based emotion classifier for grouped emotions.
    
    Architecture:
    - ResNet50 feature extractor
    - 2-layer classifier with dropout and batch normalization
    """
    
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# MODEL INITIALIZATION
# ============================================================

logger.info("Initializing models...")

# Emotion model
logger.info("Loading emotion detection model (grouped ResNet50)...")
emotion_model = GroupedEmotionClassifier(num_classes=6)
model_path = ensure_model_file("best_model_grouped.pth")
emotion_model.load_state_dict(torch.load(model_path, map_location=device))
emotion_model = emotion_model.to(device)
emotion_model.eval()
logger.info("✓ Emotion model loaded")

# Object detection model
logger.info("Loading YOLO object detector...")
yolo_path = ensure_model_file("yolov8n.pt")
object_detector = YOLO(yolo_path)
logger.info("✓ YOLO loaded")

# Action detection model
logger.info("Loading action detection model (VideoMAE)...")
ACTION_DETECTION_ENABLED = False
action_processor = None
action_model = None

try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    action_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model.to(device)
    action_model.eval()
    ACTION_DETECTION_ENABLED = True
    logger.info("✓ VideoMAE loaded")
except Exception as e:
    logger.warning(f"VideoMAE not available: {e}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_middle_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    """Extract the middle frame from a GIF."""
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        frames = [fr.convert("RGB") for fr in ImageSequence.Iterator(gif)]
        if not frames:
            return None
        return frames[len(frames) // 2]
    except Exception as e:
        logger.error(f"Error extracting middle frame: {e}")
        return None


def extract_k_frames_evenly(gif_bytes: bytes, k: int = 8) -> List[Image.Image]:
    """Extract k frames evenly distributed across the GIF."""
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
        logger.error(f"Error extracting frames: {e}")
        return []


@torch.no_grad()
def detect_emotion(frame: Image.Image) -> Tuple[str, float]:
    """
    Detect emotion from a frame.
    
    Returns:
        Tuple of (emotion_label, confidence)
    """
    try:
        x = transform(frame).unsqueeze(0).to(device)
        out = emotion_model(x)
        probs = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
        return EMOTION_GROUPS[idx], conf
    except Exception as e:
        logger.error(f"Error detecting emotion: {e}")
        return "positive_energetic", 0.30


def detect_content_type(frame: Image.Image) -> Tuple[str, float]:
    """
    Classify content as real_world or cartoon using heuristics.
    
    Heuristics:
    - Presence of faces -> real_world
    - Low color count + high saturation -> cartoon
    
    Returns:
        Tuple of (content_type, confidence)
    """
    try:
        img_array = np.array(frame)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        img_flat = img_array.reshape(-1, img_array.shape[2])
        unique_colors = len(np.unique(img_flat, axis=0))
        total_pixels = img_array.shape[0] * img_array.shape[1]
        color_ratio = unique_colors / max(total_pixels, 1)

        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = float(np.mean(hsv[:, :, 1]) / 255.0)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        has_face = len(faces) > 0

        logger.debug(f"Content - Color ratio: {color_ratio:.3f}, Saturation: {saturation:.3f}, Face: {has_face}")

        if has_face:
            return "real_world", 0.95
        if color_ratio < 0.01 and saturation > 0.3:
            return "cartoon", 0.90
        if color_ratio < 0.005:
            return "cartoon", 0.80
        if color_ratio < 0.02 and saturation > 0.4:
            return "cartoon", 0.70
        return "real_world", 0.60
    except Exception as e:
        logger.error(f"Error detecting content type: {e}")
        return "real_world", 0.50


def analyze_lighting(frame: Image.Image) -> Dict[str, float | str]:
    """
    Analyze lighting/brightness characteristics of a frame.
    
    Returns:
        Dict with 'brightness' (0-255) and 'lighting_label' (dim/moderate/bright)
    """
    try:
        img = np.array(frame.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))

        if brightness < 80:
            label = "dim"
        elif brightness > 140:
            label = "bright"
        else:
            label = "moderate"

        return {"brightness": brightness, "lighting_label": label}
    except Exception as e:
        logger.error(f"Lighting analysis error: {e}")
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
    Detect objects across multiple frames and vote to improve recall.
    
    Args:
        gif_bytes: GIF file bytes
        k: Number of frames to sample
        top_n: Maximum objects to return
        min_votes: Minimum votes required for an object to be included
        conf_thresh: Confidence threshold for detections
        drop_labels: Labels to exclude (e.g., generic "person")
        
    Returns:
        List of detected object labels, ranked by frequency
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
            logger.debug(f"YOLO frame error: {e}")
            continue

        objs.sort(key=lambda x: x[1], reverse=True)
        kept = [n for (n, c) in objs if c >= conf_thresh]
        if not kept and objs:
            kept = [objs[0][0]]
        all_labels.extend(kept[:2])

    if not all_labels:
        return []

    counts = Counter(all_labels)
    winners = [lbl for lbl, c in counts.most_common() if c >= min_votes]
    if not winners:
        winners = [lbl for lbl, _ in counts.most_common(top_n)]
    return winners[:top_n]


def count_people(frame: Image.Image, conf_thresh: float = 0.25) -> int:
    """Count persons in a frame using YOLO 'person' class."""
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
        logger.error(f"Person count error: {e}")
        return 0


def extract_frames_for_action(gif_bytes: bytes, num_frames: int = 16) -> Optional[List[Image.Image]]:
    """Extract evenly distributed frames for action detection."""
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        frames = [fr.convert("RGB") for fr in ImageSequence.Iterator(gif)]
        if not frames:
            return None
        idxs = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        return [frames[i] for i in idxs]
    except Exception as e:
        logger.error(f"Error extracting frames for action: {e}")
        return None


def detect_action(gif_bytes: bytes) -> Optional[str]:
    """Detect action/activity in a GIF using VideoMAE."""
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
        logger.error(f"Action detection error: {e}")
        return None


def motion_based_fallback_action(gif_bytes: bytes) -> str:
    """Fallback action detection based on optical flow/motion magnitude."""
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
# CAPTION GENERATION
# ============================================================

def generate_caption(
    emotion: str,
    objects: Optional[List[str]] = None,
    action: Optional[str] = None,
    content_type: str = "real_world",
) -> str:
    """
    Generate a natural language caption describing the GIF.
    
    Uses emotion-specific vocabulary, objects, and actions to create
    contextually appropriate descriptions.
    
    Args:
        emotion: Emotion label from emotion classifier
        objects: Detected objects in the GIF
        action: Detected action/activity
        content_type: Type of content (real_world or cartoon)
        
    Returns:
        Natural language caption string
    """
    objects = objects or []

    # Cartoon content gets shorter, safer captions
    if content_type == "cartoon":
        emotion_word = emotion.replace("_", " ")
        return f"an animated {emotion_word} moment"

    vocab = EMOTION_VOCABULARY.get(emotion, EMOTION_VOCABULARY["positive_energetic"])
    adj = random.choice(vocab["adjectives"])

    # Construct action phrase
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

    # Build caption with objects if available
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
    logger.info(f"Generated caption: '{caption}' (emotion={emotion}, objects={objects}, action={action})")
    return caption

# ============================================================
# API RESPONSE MODEL
# ============================================================

class CaptionResponse(BaseModel):
    """Response model for caption generation endpoint."""
    emotion: str
    caption: str
    confidence: Optional[float] = None
    objects: Optional[List[str]] = None
    action: Optional[str] = None
    content_type: Optional[str] = None
    content_warning: Optional[str] = None
    person_count: Optional[int] = None
    lighting: Optional[dict] = None


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", tags=["Health"])
async def health() -> dict:
    """
    Health check endpoint.
    
    Returns information about the API and loaded models.
    """
    return {
        "status": "healthy",
        "service": "SentiVue API",
        "version": API_VERSION,
        "features": {
            "emotion_detection": True,
            "object_detection": True,
            "person_counting": True,
            "lighting_analysis": True,
            "action_detection": ACTION_DETECTION_ENABLED,
            "content_type_detection": True,
        },
        "emotion_groups": EMOTION_GROUPS,
    }


@app.post("/generate", response_model=CaptionResponse, tags=["Caption Generation"])
async def generate_gif_caption(file: UploadFile = File(...)) -> CaptionResponse:
    """
    Generate an emotion-aware caption for an uploaded GIF.
    
    Args:
        file: GIF file to process
        
    Returns:
        CaptionResponse with caption, emotion, objects, and metadata
        
    Raises:
        HTTPException: 400 if GIF cannot be processed, 500 if internal error occurs
    """
    try:
        gif_bytes = await file.read()

        # Extract middle frame for analysis
        frame = extract_middle_frame(gif_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process GIF")

        # Detect content type (real_world vs cartoon)
        content_type, _ = detect_content_type(frame)

        # Analyze lighting
        lighting = analyze_lighting(frame)

        # Detect emotion from middle frame
        emotion, emotion_conf = detect_emotion(frame)

        # Count persons
        person_count = count_people(frame, conf_thresh=0.25)

        # Multi-frame object detection
        objects = detect_objects_multiframe_vote(
            gif_bytes, k=8, top_n=2, min_votes=2, conf_thresh=0.20
        )

        # Action detection (real_world only)
        action = None
        if content_type == "real_world":
            action = detect_action(gif_bytes)
            if action is None:
                action = motion_based_fallback_action(gif_bytes)

        # Generate caption
        caption = generate_caption(emotion, objects, action, content_type)

        # Content warning logic
        content_warning = None
        if content_type == "cartoon":
            content_warning = "Animated content detected; emotion/action may be less reliable."
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
        logger.exception(f"Unexpected error during caption generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health", tags=["Health"])
async def health_detailed() -> dict:
    """Detailed health check with model status."""
    return {
        "status": "healthy",
        "device": str(device),
        "models": {
            "emotion": "loaded",
            "objects": "loaded",
            "action": "loaded" if ACTION_DETECTION_ENABLED else "disabled",
        },
        "emotion_groups": EMOTION_GROUPS,
    }

# ============================================================
# STARTUP
# ============================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 70)
    logger.info("🎬 SentiVue Backend API - Starting")
    logger.info("=" * 70)
    logger.info(f"Version: {API_VERSION}")
    logger.info(f"Device: {device}")
    logger.info(f"Emotion Groups: {len(EMOTION_GROUPS)} - {', '.join(EMOTION_GROUPS)}")
    logger.info(f"Action Detection: {'Enabled ✓' if ACTION_DETECTION_ENABLED else 'Disabled ✗'}")
    logger.info(f"API Docs: http://0.0.0.0:7860/docs")
    logger.info("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=7860)