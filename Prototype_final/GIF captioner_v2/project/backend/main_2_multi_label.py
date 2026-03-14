# ============================================================
# SENTIVUE BACKEND - IMPROVED VERSION WITH CONTENT FILTERING
# + Multi-label Emotion (ResNet18) integrated cleanly
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
from typing import List, Optional, Tuple, Set, Dict

# ============================================================
# MODEL: MULTI-LABEL RESNET18 (6 groups)
# ============================================================

class MultiLabelResNet18(nn.Module):
    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        # wrap a standard ResNet18, but keep the classifier as a sequential block
        # so that state dict keys match the checkpoint used during training.
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        # original training used an `nn.Sequential` with a dropout layer followed by
        # a linear head, which produced keys like ``fc.1.weight`` in the saved
        # state dict.  Replicate that here so loading works cleanly.
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)  # logits (B, num_classes)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# IMPORTANT: label order MUST match training order
# (If your training used a different order, update this mapping accordingly.)
id2label: Dict[int, str] = {
    0: "positive_energetic",
    1: "positive_calm",
    2: "negative_intense",
    3: "negative_subdued",
    4: "surprise",
    5: "contempt",
}
label2id = {v: k for k, v in id2label.items()}
emotion_groups = [id2label[i] for i in range(len(id2label))]

# Multi-label threshold chosen during validation sweep
EMO_THRESH = 0.35

# Emotion vocabulary (expanded)
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
# LOAD MODELS
# ============================================================

print("📦 Loading emotion detection model (multi-label ResNet18)...")
# model path should be resolved relative to this script rather than the
# current working directory, so that imports from other dirs still work.
import os
BASE_DIR = os.path.dirname(__file__)
EMO_CKPT = os.path.join(BASE_DIR, "models", "best_multilabel_6group_resnet18_final.pth")

# build model and load weights
emotion_model = MultiLabelResNet18(num_classes=6).to(device)
state = torch.load(EMO_CKPT, map_location=device)
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

# the checkpoint was saved from a bare ResNet18 where the top-level keys
# start with "conv1", "bn1", etc.  our class nests the backbone under the
# attribute ``backbone`` so we need to prefix the names accordingly.  once the
# keys line up the load will succeed.
if not any(k.startswith("backbone.") for k in state):
    prefixed = {f"backbone.{k}": v for k, v in state.items()}
    state = prefixed

# load with strict=True now that names match
emotion_model.load_state_dict(state, strict=True)
emotion_model.eval()
print("✅ Emotion model loaded!")

print("📦 Loading object detection model...")
object_detector = YOLO("models/yolov8n.pt")
print("✅ Object detector loaded!")

print("📦 Loading action detection model...")
try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    action_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    action_model.to(device)
    action_model.eval()
    ACTION_DETECTION_ENABLED = True
    print("✅ Action detector loaded!")
except Exception as e:
    print(f"⚠️  Action detector not available: {e}")
    ACTION_DETECTION_ENABLED = False

# Image transform for emotion model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# CONTENT TYPE DETECTION
# ============================================================

def detect_content_type(frame: Image.Image) -> Tuple[str, float]:
    """
    Detect if GIF is real-world or cartoon/anime
    Returns: (content_type, confidence)
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

        print(f"🔍 Content Detection - Color: {color_ratio:.3f}, Sat: {saturation:.3f}, Face: {has_face}")

        if has_face:
            print("   → Real-world (face detected)")
            return "real_world", 0.95

        if color_ratio < 0.01 and saturation > 0.3:
            print("   → Cartoon (very low colors + saturated)")
            return "cartoon", 0.9

        if color_ratio < 0.005:
            print("   → Cartoon (extremely low colors)")
            return "cartoon", 0.8

        if color_ratio < 0.02 and saturation > 0.4:
            print("   → Cartoon (low colors + saturated)")
            return "cartoon", 0.7

        print("   → Real-world (default)")
        return "real_world", 0.6

    except Exception as e:
        print(f"❌ Error detecting content type: {e}")
        return "real_world", 0.5

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_middle_frame(gif_bytes: bytes) -> Optional[Image.Image]:
    try:
        gif = Image.open(io.BytesIO(gif_bytes))
        n_frames = 0
        try:
            while True:
                gif.seek(n_frames)
                n_frames += 1
        except EOFError:
            pass
        gif.seek(max(n_frames // 2, 0))
        return gif.convert("RGB")
    except Exception as e:
        print(f"❌ Error extracting frame: {e}")
        return None

@torch.no_grad()
def detect_emotion(frame: Image.Image) -> Tuple[str, float, List[str]]:
    """
    Multi-label emotion detection:
    - Uses sigmoid for probabilities
    - Returns ONE label for captioning (argmax)
    - Also returns active labels >= EMO_THRESH for analysis
    """
    try:
        x = transform(frame).unsqueeze(0).to(device)
        logits = emotion_model(x)                  # (1,6)
        probs = torch.sigmoid(logits).squeeze(0)   # (6,)

        probs_np = probs.detach().cpu().numpy()
        top_idx = int(np.argmax(probs_np))
        top_prob = float(probs_np[top_idx])

        active = [id2label[i] for i, p in enumerate(probs_np) if float(p) >= EMO_THRESH]

        chosen = id2label[top_idx]
        return chosen, top_prob, active

    except Exception as e:
        print(f"❌ Error detecting emotion: {e}")
        return "positive_energetic", 0.30, ["positive_energetic"]
    
@torch.no_grad()
def detect_emotion_multiframe(gif_bytes: bytes, k: int = 8) -> Tuple[str, float, List[str], List[float]]:
    frames = extract_k_frames_evenly(gif_bytes, k=k)
    if not frames:
        return "positive_energetic", 0.30, ["positive_energetic"], [0.0]*len(id2label)

    probs_acc = np.zeros((len(id2label),), dtype=np.float32)

    for fr in frames:
        x = transform(fr).unsqueeze(0).to(device)
        logits = emotion_model(x)
        probs = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy().astype(np.float32)
        probs_acc += probs

    probs_mean = probs_acc / float(len(frames))

    top_idx = int(np.argmax(probs_mean))
    top_score = float(probs_mean[top_idx])
    active = [id2label[i] for i,p in enumerate(probs_mean) if float(p) >= EMO_THRESH]

    return id2label[top_idx], top_score, active, probs_mean.tolist()

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
    Includes top-1 fallback per frame when nothing passes threshold.
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
                if not hasattr(r, "boxes") or r.boxes is None:
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
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            top_idx = int(probs.argmax().item())
            confidence = float(probs[top_idx].item())

        # keep threshold modest for recall
        if confidence > 0.15:
            action_name = action_model.config.id2label[top_idx]
            action_name = action_name.replace("(", "").replace(")", "").strip()
            return action_name

        return None
    except Exception as e:
        print(f"❌ Error detecting action: {e}")
        return None

# ============================================================
# CAPTION GENERATION (UPDATED)
# ============================================================

def generate_caption(
    emotion: str,
    objects: Optional[List[str]] = None,
    action: Optional[str] = None,
    content_type: str = "real_world"
) -> str:

    objects = objects or []

    # 1) Clean/dedupe objects
    norm = []
    seen = set()
    for o in objects:
        if not o:
            continue
        oo = str(o).strip().lower()
        if oo and oo not in seen:
            seen.add(oo)
            norm.append(oo)

    person_words = {"person", "man", "woman", "people", "human", "boy", "girl"}
    objs_wo_person = [o for o in norm if o not in person_words]

    # 2) emotion adjective
    vocab = emotion_vocabulary.get(emotion, emotion_vocabulary["positive_energetic"])
    adj = random.choice(vocab["adjectives"])

    # 3) helpers
    def a_or_an(noun: str) -> str:
        n = (noun or "").strip().lower()
        if not n:
            return "a"
        return "an" if n[0] in "aeiou" else "a"

    def ingify_first(word: str) -> str:
        w = (word or "").strip().lower()
        if not w:
            return w
        if w.endswith("ing"):
            return w
        if w.endswith("ie"):
            return w[:-2] + "ying"
        if w.endswith("e") and len(w) > 2 and not w.endswith(("ee", "ye")):
            return w[:-1] + "ing"
        return w + "ing"

    def to_is_ing(phrase: str) -> str:
        p = (phrase or "").strip().lower()
        if not p:
            return ""
        parts = p.split()
        parts[0] = ingify_first(parts[0])
        return "is " + " ".join(parts)

    def insert_article_in_action_phrase(action_text: str, hint_objects: List[str]) -> str:
        if not action_text:
            return action_text
        a = action_text.strip().lower()
        parts = a.split()
        if len(parts) != 2:
            return action_text
        verb, noun = parts[0], parts[1]
        objset = set([o.lower() for o in (hint_objects or [])])

        common_nouns = {"bag", "ball", "pipe", "phone", "camera", "guitar", "horse", "dog", "cat", "door"}
        if noun in objset or noun in common_nouns:
            return f"{verb} {a_or_an(noun)} {noun}"
        return action_text

    # 4) Action phrase
    if action:
        cleaned_action = insert_article_in_action_phrase(action, objs_wo_person)
        verb_phrase = to_is_ing(cleaned_action)
    else:
        v = random.choice(vocab["verbs"])
        verb_phrase = to_is_ing(v.replace("is ", "").strip())

    # 5) object phrase
    vehicles = {"airplane", "motorcycle", "car", "bus", "truck", "train", "boat", "ship", "bicycle", "bike"}

    chosen_objs = objs_wo_person[:2] if objs_wo_person else []
    action_text = (action or "").lower()
    if chosen_objs:
        chosen_objs = [o for o in chosen_objs if o not in action_text]

    def obj_phrase_for(o: str) -> str:
        prep = "near" if o in vehicles else "with"
        art = a_or_an(o)
        return f"{prep} {art} {o}"

    obj_phrase = ""
    if chosen_objs:
        if len(chosen_objs) == 1:
            obj_phrase = obj_phrase_for(chosen_objs[0])
        else:
            p1 = obj_phrase_for(chosen_objs[0])
            p2 = obj_phrase_for(chosen_objs[1])
            if p1.split()[0] == p2.split()[0]:
                prep = p1.split()[0]
                obj_phrase = f"{prep} {a_or_an(chosen_objs[0])} {chosen_objs[0]} and {a_or_an(chosen_objs[1])} {chosen_objs[1]}"
            else:
                obj_phrase = f"{p1} and {p2}"

    # 6) final template
    adj_article = a_or_an(adj)
    if content_type == "cartoon":
        subject = f"an animated {adj} person"
    else:
        subject = f"{adj_article} {adj} person"

    if verb_phrase and obj_phrase:
        caption = f"{subject} {verb_phrase} {obj_phrase}"
    elif verb_phrase:
        caption = f"{subject} {verb_phrase}"
    elif obj_phrase:
        caption = f"{subject} {obj_phrase}"
    else:
        caption = f"{subject}"

    print(f"💬 Generated caption: '{caption}' (emotion: {emotion}, objects: {objects}, action: {action}, type: {content_type})")
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
    # optional: expose multi-label info for debugging/research
    active_emotions: Optional[List[str]] = None

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "status": "healthy",
        "service": "SentiVue API - Improved",
        "version": "2.0.0",
        "features": {
            "emotion_detection": True,
            "object_detection": True,
            "action_detection": ACTION_DETECTION_ENABLED,
            "content_filtering": True,
            "emotion_multilabel": True,
            "emotion_threshold": EMO_THRESH,
        },
    }

@app.post("/generate", response_model=CaptionResponse)
async def generate_gif_caption(file: UploadFile = File(...)):
    try:
        gif_bytes = await file.read()

        frame = extract_middle_frame(gif_bytes)
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to process GIF")

        content_type, content_confidence = detect_content_type(frame)

        emotion, emotion_confidence, active_emotions, emo_scores = detect_emotion_multiframe(gif_bytes, k=8)

        content_warning = None
        if content_type == "cartoon":
            content_warning = "This appears to be animated/cartoon content. Results may be less accurate."
        elif emotion_confidence < 0.25:
            content_warning = "Low confidence emotion detection. Caption may not be accurate."

        objects = detect_objects_multiframe_vote(
            gif_bytes, k=8, top_n=2, min_votes=2, conf_thresh=0.20
        )

        action = None
        if content_type == "real_world":
            action = detect_action(gif_bytes)
            if action is None:
                action = motion_based_fallback_action(gif_bytes)

        caption = generate_caption(emotion, objects, action, content_type)

        return CaptionResponse(
            emotion=emotion,
            caption=caption,
            confidence=emotion_confidence,
            objects=objects,
            content_type=content_type,
            content_warning=content_warning,
            active_emotions=active_emotions,
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
        "emotion_threshold": EMO_THRESH,
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
    print(f"Emotion groups: {len(emotion_groups)} -> {emotion_groups}")
    print(f"Action detection: {'✅ Enabled' if ACTION_DETECTION_ENABLED else '⚠️  Disabled'}")
    print("Content filtering: ✅ Enabled")
    print("Emotion model: ✅ Multi-label ResNet18")
    print("=" * 60)
    print("\n🚀 Starting server on http://127.0.0.1:8000")
    print("📖 API docs: http://127.0.0.1:8000/docs\n")

    uvicorn.run(app, host="127.0.0.1", port=8000)