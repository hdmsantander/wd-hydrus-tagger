# Start from Python 3.11 slim image
FROM python:3.11-slim

# OpenCV / InsightFace runtime libs (InsightFace may pull opencv-python; slim needs X/GL shims)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WD_TAGGER_LOG_LEVEL=INFO

# Fixed UID/GID so bind-mounted host dirs can be chowned once (see docs/DOCKER.md).
RUN groupadd -g 1000 tagger && useradd -u 1000 -g tagger -M -s /usr/sbin/nologin tagger

# Set the working directory
WORKDIR /app

# Copy requirement files first to leverage Docker cache
COPY pyproject.toml requirements.txt ./

# Install dependencies (runtime + face tagging extras; no [dev])
# insightface may install opencv-python — drop it so headless build is used in containers.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ".[face]" && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless

# Create the required directories with needed permissions
RUN mkdir -p /app/models /app/models/face /app/face_data /app/logs /app/ort_traces && \
    chown -R tagger:tagger /app

# Switch to the non-root user
USER tagger

# Copy the rest of the application
COPY --chown=tagger:tagger . .

# Expose the application port
EXPOSE 8199

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8199/api/app/status', timeout=5).read()"

# Run the application
CMD ["python", "run.py"]
