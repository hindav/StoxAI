FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port that Hugging Face Spaces uses (default is 7860, but our app reads PORT env var)
EXPOSE 7860

# Command to run the application
# We use python flask_app.py directly instead of gunicorn because existing app handles its own server
CMD ["python", "flask_app.py"]
