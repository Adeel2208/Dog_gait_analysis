"""
run.py — Start the Dog Limping Detection server.
Usage:  python run.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        app_dir=".",
    )
