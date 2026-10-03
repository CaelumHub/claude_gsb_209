#!/usr/bin/env bash
# 音频处理与音乐信息检索系统 — 启动脚本
# Audio Processing & Music Information Retrieval System — launcher
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "→ 启动音频处理与音乐信息检索系统  http://${HOST}:${PORT}"
echo "→ 按 Ctrl+C 停止"

exec "$PYTHON" app.py --host "$HOST" --port "$PORT"
