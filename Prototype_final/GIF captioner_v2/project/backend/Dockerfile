FROM python:3.11-slim

WORKDIR /app

# Ensure system libraries needed by opencv and ultralytics are available
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libxcb1 \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 \
    && rm -rf /var/lib/apt/lists/*

# copy backend app files
COPY . /app

# install dependencies (headless OpenCV to avoid GUI/X11 requirement)
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# expose port 8000
EXPOSE 8000

# run uvicorn with app wrapper (compatible with HF Spaces)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
