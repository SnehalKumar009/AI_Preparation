FROM python:3.12-slim

# Faster, cleaner Python in containers.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code.
COPY study_buddy/ ./study_buddy/
COPY notebooks/ ./notebooks/
COPY webui/ ./webui/
COPY notes/ ./notes/

# Reach Ollama running on the Docker host by default (overridable via .env).
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8501 8888

# Default: run the Streamlit capstone.
CMD ["streamlit", "run", "webui/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
