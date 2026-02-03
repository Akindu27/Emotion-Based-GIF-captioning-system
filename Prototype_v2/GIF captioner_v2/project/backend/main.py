import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, ViTModel, VideoMAEModel, GPT2Tokenizer, GPT2LMHeadModel
from PIL import Image, ImageSequence
import uvicorn

# ------------------------------------------------------------------
# 1. CONFIGURATION & PATHS
# ------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Update this path to where your .pth file is on your computer!
CHECKPOINT_PATH = "model_final_v5.pth" 

print(f"🚀 Running on: {DEVICE}")

# ------------------------------------------------------------------
# 2. MODEL DEFINITION
# ------------------------------------------------------------------
class VideoGPT2Captioner(nn.Module):
    def __init__(self, visual_dim=2304, prefix_len=1):
        super().__init__()
        self.prefix_len = prefix_len
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        self.projection = nn.Linear(visual_dim, prefix_len * 768)
        self.ln = nn.LayerNorm(768)

    def encode_visual(self, visual_feat):
        projected = self.projection(visual_feat)
        projected = projected.view(-1, self.prefix_len, 768)
        return self.ln(projected)

# ------------------------------------------------------------------
# 3. GLOBAL MODEL INITIALIZATION
# ------------------------------------------------------------------
print("📥 Loading Models into Memory...")
action_proc = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
action_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()

vit_proc = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(DEVICE).eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = VideoGPT2Captioner(visual_dim=2304, prefix_len=1).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    print("✅ Local Checkpoint Loaded Successfully!")
else:
    print(f"⚠️ Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using base weights.")
model.eval()

# ------------------------------------------------------------------
# 4. HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_emotion_label(feat):
    magnitude = torch.norm(feat).item()
    if magnitude > 0.5: return "energetic and happy"
    elif magnitude > 0.3: return "focused"
    else: return "calm"

def extract_live_features(gif_path):
    gif = Image.open(gif_path)
    frames = [f.convert("RGB") for f in ImageSequence.Iterator(gif)]
    
    # Temporal Sampling (16 frames)
    if len(frames) >= 16:
        idx = torch.linspace(0, len(frames)-1, 16).long()
        frames = [frames[i] for i in idx]
    else:
        frames = frames + [frames[-1]] * (16 - len(frames))

    # Action Features (VideoMAE)
    inputs_a = action_proc(images=frames, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        f_act = action_model(**inputs_a).last_hidden_state.mean(dim=1).squeeze(0)

    # Appearance Features (ViT)
    img = Image.open(gif_path).convert("RGB")
    inputs_v = vit_proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        f_app = vit_model(**inputs_v).last_hidden_state[:, 0, :].squeeze(0)

    # Feature Fusion
    f_emo = f_app.clone() * 5.0
    visual_feat = torch.cat([f_app, f_act, f_emo], dim=-1)
    return F.normalize(visual_feat, p=2, dim=-1)

# ------------------------------------------------------------------
# 5. API ENDPOINTS
# ------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For local dev, "*" is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_endpoint(file: UploadFile = File(...)):
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        with torch.no_grad():
            feat = extract_live_features(temp_filename)
            emotion_word = get_emotion_label(feat)

            # Start the caption
            prompt = "A video of"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
            prefix_embeds = model.encode_visual(feat.unsqueeze(0))
            prompt_embeds = model.gpt2.transformer.wte(prompt_ids)
            full_embeds = torch.cat((prefix_embeds, prompt_embeds), dim=1)

            # Generate with Sampling for better variety
            output_ids = model.gpt2.generate(
                inputs_embeds=full_embeds,
                max_new_tokens=25,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=2.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

            decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Combine emotion and AI text
            final_caption = f"A video of a {emotion_word} {decoded_text.replace('A video of', '').strip()}"

        return {
            "emotion": emotion_word,
            "caption": final_caption
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ------------------------------------------------------------------
# 6. SERVER ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("\n✅ Sentivue Local Backend is starting...")
    print("🔗 API will be available at: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)