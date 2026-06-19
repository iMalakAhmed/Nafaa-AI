FROM ollama/ollama:latest

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    python3-dev \
    curl \
    git \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =========================
# Create virtual environment (FIX)
# =========================
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# =========================
# Python dependencies
# =========================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =========================
# Copy app
# =========================
COPY . .

# =========================
# Environment
# =========================
ENV OLLAMA_HOST=http://localhost:11434
# =========================
# Entrypoint  
# =========================
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
 
