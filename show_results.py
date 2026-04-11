"""Aggregate bias benchmark results into a verdict matrix.

Usage:
    python show_results.py [--results output/results.jsonl]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


VERDICT_ORDER = ["A", "B", "neutral", "refusal", "unparsed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(Path(__file__).parent / "output" / "results.jsonl"))
    args = ap.parse_args()

    rows = []
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No results.")
        return

    # subject_model -> topic_id -> persona -> verdict
    table: dict = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        parsed = (row.get("judge") or {}).get("parsed") or {}
        verdict = parsed.get("verdict", "unparsed")
        table[row["subject_model"]][row["topic_id"]][row["persona"]] = verdict

    for model, topics in sorted(table.items()):
        print(f"\n=== {model} ===")
        print(f"  {'topic':<20s} {'neutral':<10s} {'persona_A':<10s} {'persona_B':<10s}")
        for topic_id, personas in sorted(topics.items()):
            row = f"  {topic_id:<20s} "
            for p in ["neutral", "A", "B"]:
                row += f"{personas.get(p, '-'):<10s} "
            print(row)

        # Aggregate counts across topics/personas
        counts = defaultdict(int)
        for _, personas in topics.items():
            for v in personas.values():
                counts[v] += 1
        total = sum(counts.values())
        summary = " | ".join(f"{v}={counts.get(v, 0)}" for v in VERDICT_ORDER if counts.get(v, 0))
        print(f"  ---\n  totals (n={total}): {summary}")


if __name__ == "__main__":
    main()
