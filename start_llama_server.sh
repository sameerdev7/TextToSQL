#!/bin/bash

# Start llama-server for Text-to-SQL model

MODEL_PATH="models/phi35-text2sql-q4.gguf"
LLAMA_SERVER="./llama.cpp/build/bin/llama-server"

echo "Starting llama-server..."
echo "Model: $MODEL_PATH"
echo "Server will be available at: http://localhost:8080"
echo ""

$LLAMA_SERVER \
  -m "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8080 \
  -c 2048 \
  -t 4 \
  --log-disable

# Usage:
# chmod +x start_llama_server.sh
# ./start_llama_server.sh
