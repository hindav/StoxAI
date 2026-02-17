# Deployment Guide

Since your application uses **TensorFlow** (which is very large) and runs a custom **Flask + FastAPI** architecture (using subprocesses), it is **not suitable for Vercel's standard serverless environment**. Vercel has a function size limit (250MB) that TensorFlow exceeds, and serverless functions cannot run long-running background processes.

Instead, the best free platform for this machine learning application is **Hugging Face Spaces** (using Docker) or **Render**.

## Option 1: Deploy to Hugging Face Spaces (Recommended for ML)

Hugging Face Spaces provides generous free resources (16GB RAM) which is perfect for TensorFlow apps.

1.  **Create a Hugging Face Account**: Go to [huggingface.co](https://huggingface.co/) and sign up.
2.  **Create a New Space**:
    *   Click "New Space".
    *   Enter a name (e.g., `stock-prediction`).
    *   Select **Docker** as the SDK.
    *   Select **Blank** template.
    *   Click "Create Space".
3.  **Upload Files**:
    *   You can upload files directly via the browser or use git.
    *   Essential files to upload: `Dockerfile`, `requirements.txt`, `flask_app.py`, `Api/`, `Models/`, `static/`, `templates/`.
    *   **(Important)**: Do NOT upload `.env` or your local virtual environment folders.
4.  **Set Environment Variables** (Secrets):
    *   Go to "Settings" -> "Variables and secrets" in your Space.
    *   Add your API keys if needed (though your app asks for them in the UI, so this might strictly be optional unless you hardcoded some).

Your app will build and run automatically!

## Option 2: Deploy to Render (Alternative)

Render is great but has a 512MB RAM limit on the free tier, which might be tight for TensorFlow.

1.  **Push your code to GitHub**.
2.  **Create a new Web Service** on [render.com](https://render.com/).
3.  Connect your GitHub repository.
4.  Select **Python 3** environment.
5.  **Build Command**: `pip install -r requirements.txt`
6.  **Start Command**: `python flask_app.py`
7.  **Environment Variables**: Add any keys if necessary.

*Note: If the app crashes with "Out of Memory", switch to Hugging Face Spaces.*
