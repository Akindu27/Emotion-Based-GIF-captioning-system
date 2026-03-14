from main_final import app

# For Hugging Face Spaces or any ASGI container that expects `app` in root.


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
