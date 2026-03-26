"""ASGI app wrapper for compatibility with various deployment platforms.

This module re-exports the FastAPI app from main_final.py for platforms
that expect an `app` object at the module level (e.g., some ASGI servers).
"""

from main_final import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
