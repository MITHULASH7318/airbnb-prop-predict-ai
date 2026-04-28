#!/usr/bin/env bash
# setup.sh — one-command project setup
# Usage: bash setup.sh

set -e
echo "=== PropPredict AI Setup ==="

echo "[1/4] Installing dependencies..."
pip install -r requirements.txt -q

echo "[2/4] Generating datasets..."
python3 data/generate_dataset.py

echo "[3/4] Training ML models..."
python3 models/train_models.py

echo "[4/4] Setup complete!"
echo ""
echo "To run the app:"
echo "  streamlit run app/app.py"
echo ""
echo "To enable the AI chatbot:"
echo "  export ANTHROPIC_API_KEY=your_key_here"
echo "  streamlit run app/app.py"
