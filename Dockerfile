FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install small set of build tools (kept minimal). Remove lists to reduce image size.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc g++ curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8501

# Start Streamlit on the assigned port (Render/Cloud Run set PORT env).
# Use shell form so ${PORT:-8501} is expanded at container runtime.
CMD ["sh", "-c", "streamlit run main.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
