#!/bin/bash
# Run the 4 newly added topics on all 13 subject models.
set -e

export OPENROUTER_API_KEY=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
export MARITACA_API_KEY=100088405894968274443_0b8e7eb7d17bbdb5

NEW_TOPICS="lula_is_corrupt,amnesty_for_bolsonaro,jan8_2023_was_coup_attempt,printed_vote_better_than_electronic"
PARALLEL=20

echo "========================================="
echo "Running 4 new topics on 13 models"
echo "Topics: $NEW_TOPICS"
echo "========================================="

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
        --topic "$NEW_TOPICS" \
        --parallel "$PARALLEL" \
    || echo "  [WARNING] $model exited with code $?"
done

for model in "sabia-4" "sabiazinho-4"; do
    echo ""
    echo ">>> $model (Maritaca API)"
    python3 bias_bench.py \
        --subject-model "$model" \
        --subject-base-url "https://chat.maritaca.ai/api" \
        --subject-api-key-env MARITACA_API_KEY \
        --topic "$NEW_TOPICS" \
        --parallel "$PARALLEL" \
    || echo "  [WARNING] $model exited with code $?"
done

echo ""
echo "========================================="
echo "All models done."
echo "========================================="
