#!/bin/bash

set -e

echo "🚀 Starting Ollama server..."

ollama serve &

echo "⏳ Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done

echo "📦 Pulling model..."
ollama pull qwen2.5:7b

echo "🤖 Starting Python app..."
exec python3 handler.py