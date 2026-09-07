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

# cpu (default) | rocm (AMD: onnxruntime-migraphx from AMD's ROCm index) | cuda (onnxruntime-gpu)
ARG GPU_BACKEND=cpu
# Official ROCm 7.2.x wheels — matches host ROCm 7.2.x; override at build if needed.
ARG ROCM_ORT_INDEX=https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/

# Install dependencies (runtime + face tagging extras; no [dev])
# insightface may install opencv-python — drop it so headless build is used in containers.
# GPU wheels replace CPU onnxruntime last so they are not overwritten.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ".[face]" && \
    pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless && \
    if [ "$GPU_BACKEND" = "rocm" ]; then \
      pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-rocm 2>/dev/null || true; \
      pip install --no-cache-dir "numpy==1.26.4"; \
      pip install --no-cache-dir onnxruntime-migraphx -f "$ROCM_ORT_INDEX"; \
    elif [ "$GPU_BACKEND" = "cuda" ]; then \
      pip uninstall -y onnxruntime onnxruntime-migraphx onnxruntime-rocm 2>/dev/null || true; \
      pip install --no-cache-dir "onnxruntime-gpu>=1.20"; \
    fi

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
