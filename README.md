# SentiVue - Emotion-Based GIF Captioning System

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-latest-009485.svg)](https://fastapi.tiangolo.com/)
[![React + TypeScript](https://img.shields.io/badge/React-TypeScript-61DAFB.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

_An intelligent system that analyzes GIFs to detect emotions, objects, actions, and generates contextually-aware natural language captions._

[🚀 Live Demo](https://sentivue.vercel.app) • [📝 Documentation](#features) • [🔧 Installation](#installation) • [💻 Development](#development)

</div>

---

## 📋 Overview

SentiVue is a multimodal AI system that combines:

- **Emotion Detection** - Classifies 6 emotion groups using ResNet50
- **Object Recognition** - Multi-frame YOLO detection for robust object identification
- **Action Recognition** - VideoMAE-based activity detection
- **Scene Understanding** - Lighting analysis and content type classification
- **Caption Generation** - Context-aware natural language descriptions

Upload a GIF, and SentiVue generates rich, emotionally-aware captions like:

> _"a joyful person dancing with a dog in bright lighting"_

---

## ✨ Features

### Core Capabilities

| Feature                    | Technology     | Details                                                                                                     |
| -------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------- |
| **Emotion Detection**      | ResNet50       | 6 emotion groups: positive_energetic, positive_calm, negative_intense, negative_subdued, surprise, contempt |
| **Object Detection**       | YOLO v8 Nano   | Multi-frame voting ensures accurate detection across animation frames                                       |
| **Action Recognition**     | VideoMAE       | Detects activities (dancing, jumping, etc.) with motion-based fallback                                      |
| **Person Counting**        | YOLO           | Counts people in the GIF with confidence thresholds                                                         |
| **Lighting Analysis**      | OpenCV         | Classifies brightness as dim, moderate, or bright                                                           |
| **Content Classification** | Heuristics     | Detects real-world vs animated/cartoon content                                                              |
| **Caption Generation**     | Template-based | Context-aware English descriptions with emotion-specific vocabulary                                         |

### API Features

- ✅ RESTful endpoints with FastAPI
- ✅ Automatic API documentation (Swagger UI)
- ✅ CORS enabled for cross-domain requests
- ✅ Error handling and validation
- ✅ Health check endpoints
- ✅ Detailed metadata in responses

### Frontend Features

- 🎨 Modern, responsive UI (React + Tailwind CSS)
- 🖼️ File upload and URL-based GIF input
- ⚡ Real-time processing feedback
- 📊 Detailed result visualization
- 🎭 Emotion and metadata display
- 🌓 Dark theme with gradient design

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│              Vercel Frontend (React)                    │
│         (sentivue.vercel.app)                           │
│                                                          │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP REST API
                       ▼
┌─────────────────────────────────────────────────────────┐
│                                                          │
│         HuggingFace Spaces Backend (FastAPI)            │
│   (Akindu27-sentivue-backend.hf.space)                  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Emotion Model (ResNet50)                        │  │
│  │  Object Detector (YOLO v8n)                      │  │
│  │  Action Recognizer (VideoMAE)                    │  │
│  │  Analysis Pipeline (OpenCV, NumPy)              │  │
│  │  Caption Generator (Template Engine)             │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  Models from: Akindu27/sentivue-models (HF Hub)        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Tech Stack

**Backend:**

- Python 3.11+
- FastAPI (REST API framework)
- PyTorch (Deep learning)
- YOLO v8 (Object detection)
- VideoMAE (Action recognition)
- OpenCV (Image processing)
- Hugging Face Hub (Model distribution)

**Frontend:**

- React 18+
- TypeScript
- Tailwind CSS (Styling)
- Vite (Build tool)
- Vercel (Deployment)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 16+ (for frontend development)
- CUDA 11.8+ (optional, for GPU acceleration)
- Git

### Backend Installation

```bash
# Clone the repository
git clone https://github.com/Akindu27/Emotion-Based-GIF-captioning-system.git
cd Prototype_final/GIF\ captioner_v2/project/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run locally (development)
python main_final.py
```

The backend will start at `http://localhost:7860` with:

- API: `http://localhost:7860/generate` (POST)
- Docs: `http://localhost:7860/docs`
- Health: `http://localhost:7860/health`

### Frontend Installation

```bash
# Navigate to project root
cd Prototype_final/GIF\ captioner_v2/project

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Environment Variables

#### Backend (`.env` or Space secrets)

```env
# Model repository (Hugging Face Hub)
HF_MODEL_REPO=Akindu27/sentivue-models

# Local models directory
MODELS_DIR=models

# Optional: HuggingFace token for authenticated downloads
HF_TOKEN=your_hf_token_here
```

#### Frontend (`.env.local` or `.env.production`)

```env
# Local development
VITE_API_URL=http://localhost:7860

# Production (HF Space)
VITE_API_URL=https://Akindu27-sentivue-backend.hf.space
```

---

## 🔌 API Usage

### Generate Caption

**Endpoint:** `POST /generate`

**Request:**

```bash
curl -X POST "http://localhost:7860/generate" \
  -F "file=@your_gif.gif"
```

**Response:**

```json
{
  "emotion": "positive_energetic",
  "caption": "a joyful person dancing with a dog",
  "confidence": 0.92,
  "objects": ["dog", "person"],
  "action": "dancing",
  "content_type": "real_world",
  "content_warning": null,
  "person_count": 1,
  "lighting": {
    "brightness": 120.5,
    "lighting_label": "moderate"
  }
}
```

### Health Check

**Endpoint:** `GET /health`

**Response:**

```json
{
  "status": "healthy",
  "device": "cpu",
  "models": {
    "emotion": "loaded",
    "objects": "loaded",
    "action": "loaded"
  },
  "emotion_groups": ["contempt", "negative_intense", ...]
}
```

### Root/Info

**Endpoint:** `GET /`

Returns API information and available features.

---

## 🎯 Model Details

### Emotion Classifier

- **Architecture:** ResNet50 (feature extractor) + 2-layer classifier
- **Number of Classes:** 6 emotion groups
- **Input Size:** 224×224 RGB
- **Output:** Emotion label + confidence score
- **Training Data:** GIFGIF dataset (grouped emotions)

### Object Detector

- **Architecture:** YOLO v8 Nano
- **Detection Strategy:** Multi-frame voting across 8 sampled frames
- **Confidence Threshold:** 0.20
- **Post-processing:** Removes generic "person" labels, keeps specific objects
- **Top Results:** 2 most frequently detected objects

### Action Recognizer

- **Architecture:** VideoMAE (vision transformer for video)
- **Input:** 16 evenly sampled frames
- **Confidence Threshold:** 0.15
- **Fallback:** Motion-based heuristics if confidence is low
- **Outputs:** Actions like "dancing", "jumping", "moving", "gesturing"

---

## 💻 Development

### Project Structure

```
Prototype_final/GIF captioner_v2/project/
├── backend/
│   ├── main_final.py          # Production backend
│   ├── app.py                 # ASGI entry point
│   ├── requirements.txt        # Python dependencies
│   ├── Dockerfile             # Docker build config
│   ├── .env                   # Environment variables
│   └── models/                # Downloaded model cache
│
├── src/
│   ├── components/
│   │   ├── MainApp.tsx        # Main application component
│   │   └── LandingPage.tsx    # Landing page
│   ├── services/
│   │   └── captionService.ts  # API communication
│   ├── App.tsx                # Root component
│   ├── index.css              # Global styles
│   └── main.tsx               # Entry point
│
├── .env.local                 # Local dev config
├── .env.production            # Production config
├── package.json               # Node dependencies
├── tsconfig.json              # TypeScript config
├── tailwind.config.js         # Tailwind CSS config
├── vite.config.ts             # Vite config
└── vercel.json                # Vercel deployment config
```

### Running Tests

```bash
# Backend health check
curl http://localhost:7860/health

# Frontend build
cd project && npm run build

# Check for TypeScript errors
npm run type-check
```

### Code Quality

- **Linting:** ESLint configured for React + TypeScript
- **Formatting:** Prettier (run via VS Code or `npm run format`)
- **Type Safety:** Full TypeScript coverage on frontend
- **Error Handling:** Try-catch blocks with logging in backend

---

## 📦 Deployment

### Backend Deployment (HuggingFace Spaces)

1. Create a new Space: https://huggingface.co/new-space
2. Select **Docker** SDK
3. Connect GitHub repository
4. Add required secrets:
   - `HF_TOKEN` (optional, if using private models)
5. Space automatically rebuilds and deploys on push

**Deployed at:** `https://Akindu27-sentivue-backend.hf.space`

### Frontend Deployment (Vercel)

1. Push to GitHub repository
2. Connect to Vercel: https://vercel.com/import
3. Configure environment variable:
   - `VITE_API_URL=https://Akindu27-sentivue-backend.hf.space`
4. Vercel auto-deploys on `main` branch push

**Deployed at:** https://sentivue.vercel.app

---

## 🐛 Troubleshooting

### General Issues

| Issue                   | Solution                                               |
| ----------------------- | ------------------------------------------------------ |
| Models fail to download | Set `HF_TOKEN` environment variable                    |
| CORS errors             | Verify backend and frontend URLs match in `.env` files |
| Out of memory           | Use CPU mode (default) or reduce GIF size              |
| Slow processing         | Ensure backend is running on appropriate device        |

### Backend Issues

```bash
# Check if backend is running
curl http://localhost:7860/health

# View logs (development)
python main_final.py

# Verify models are downloaded
ls -la backend/models/
```

### Frontend Issues

```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install

# Build and test locally
npm run build
npm run preview
```

---

## 📊 Performance

- **GIF Processing:** ~2-5 seconds (depends on GIF size and complexity)
- **Model Inference:** Optimized for CPU; GPU recommended for production
- **Memory Usage:** ~2GB RAM for all models
- **Concurrent Requests:** Limited by available resources

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more emotion categories
- [ ] Improve action recognition accuracy
- [ ] Optimize model inference speed
- [ ] Add batch processing capability
- [ ] Expand caption template diversity
- [ ] Multi-language support

### Development Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
npm run dev          # frontend
python main_final.py # backend

# Commit and push
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature

# Create pull request
```

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](../../LICENSE) file for details.

---

## 👨‍💻 Author

**Akindu27** - [GitHub](https://github.com/Akindu27) • [HuggingFace Hub](https://huggingface.co/Akindu27)

---

## 🙏 Acknowledgments

- GIFGIF dataset for emotion labels
- YOLO community for object detection models
- VideoMAE team for action recognition
- FastAPI for excellent API framework
- Hugging Face for model hosting

---

## 📚 References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [React Documentation](https://react.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [YOLO Documentation](https://docs.ultralytics.com/)

---

<div align="center">

**[⬆ back to top](#sentivue---emotion-based-gif-captioning-system)**

Made with ❤️ by Akindu27

</div>
