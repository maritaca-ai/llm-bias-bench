#!/bin/bash
# Run benchmark on all 13 subject models.
# User-LLM: Opus 4.6 (default), Judge: Qwen 3.5 (default)
# Each model: 34 topics × 6 conditions = 204 conversations
# Resume-safe: skips already-completed jobs.

set -e

export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY}"
export MARITACA_API_KEY="${MARITACA_API_KEY:?Set MARITACA_API_KEY}"

PARALLEL=20

echo "========================================="
echo "llm-bias-bench — full run"
echo "User-LLM: anthropic/claude-opus-4.6"
echo "Judge: qwen/qwen3.5-397b-a17b"
echo "Parallel: $PARALLEL"
echo "========================================="

# OpenRouter models
for model in \
    "anthropic/claude-opus-4.6" \
    "openai/gpt-5.4" \
    "x-ai/grok-4.20" \
    "google/gemini-3.1-pro-preview" \
    "qwen/qwen3.5-397b-a17b" \
    "moonshotai/kimi-k2-thinking" \
    "mistralai/mistral-large-2512" \
    "meta-llama/llama-4-maverick" \
    "anthropic/claude-haiku-4.5" \
    "openai/gpt-5.4-mini" \
    "google/gemini-3.1-flash-lite-preview" \
; do
    echo ""
    echo ">>> $model"
    python3 bias_bench.py \
        --subject-model "$model" \
        --parallel "$PARALLEL" \
    || echo "  [WARNING] $model exited with code $?"
done

# Maritaca models (different API endpoint)
for model in "sabia-4" "sabiazinho-4"; do
    echo ""
    echo ">>> $model (Maritaca API)"
    python3 bias_bench.py \
        --subject-model "$model" \
        --subject-base-url "https://chat.maritaca.ai/api" \
        --subject-api-key-env MARITACA_API_KEY \
        --parallel "$PARALLEL" \
    || echo "  [WARNING] $model exited with code $?"
done

echo ""
echo "========================================="
echo "All models done."
echo "========================================="
