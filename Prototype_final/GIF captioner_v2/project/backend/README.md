---
title: "SentiVue Backend"
emoji: "📦"
colorFrom: "gray"
colorTo: "blue"
sdk: "docker"
app_file: "app.py"
app_port: 8000
pinned: false
---

# SentiVue Backend

This FastAPI backend is used by the SentiVue frontend.

## Run locally

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python main_final.py
```

## Deploy to Hugging Face Spaces (Docker)

1. Create a new Space with SDK `Docker`.
2. Add this repo in space.
3. Ensure `requirements.txt`, `app.py`, and `Dockerfile` are present.
4. HF Space runtime will build and expose at `7860`.

## API info

- `GET /` health
- `POST /generate` upload form field `file`

CORS is allowed from localhost and wildcard currently, adjust in `main_final.py` if needed.
