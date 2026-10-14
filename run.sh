#!/usr/bin/env bash
# run.sh — Launch the QUANT.IDX Institutional Terminal
cd "$(dirname "$0")"
PORT=3000
echo "────────────────────────────────────────────────────────"
echo "  🚀 Starting QUANT.IDX Terminal on http://127.0.0.1:$PORT"
echo "────────────────────────────────────────────────────────"

python3 backend/main.py
