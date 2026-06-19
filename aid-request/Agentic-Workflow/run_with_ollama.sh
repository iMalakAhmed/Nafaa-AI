#!/bin/bash
set -e

# Ensure we run from repository root
cd "$(dirname "$0")"

# Require Ollama installed
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama is not installed. Install from https://ollama.ai/install.sh"
  exit 1
fi

# Start Ollama if not already running
if ! curl -s http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
  echo "Starting Ollama in the background..."
  ollama serve >/tmp/ollama.log 2>&1 &
  OLLAMA_PID=$!
  echo "Waiting for Ollama to become available..."
  for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  if ! curl -s http://127.0.0.1:11434/v1/models >/dev/null 2>&1; then
    echo "ERROR: Ollama did not start in time. Check /tmp/ollama.log"
    kill "$OLLAMA_PID" >/dev/null 2>&1 || true
    exit 1
  fi
  echo "Ollama is running."
else
  echo "Ollama already running."
fi

# Run the Python app
source venv/bin/activate
python main.py
