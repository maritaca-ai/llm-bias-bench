"""Judge ablation: sample 100 transcripts (50 direct, 50 indirect) from every subject
model/topic/persona and re-judge them with alternative judges to measure agreement
with the Opus 4.6 baseline judge.

Usage:
  python3 judge_ablation.py --sample  # sample once and write sample manifest
  python3 judge_ablation.py --run     # run the judges listed in ALT_JUDGES on the sample
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bias_bench import (
    APIDownError,
    OPENROUTER_BASE_URL,
    get_client,
    judge_conversation,
    load_topics,
)

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"
SAMPLE_PATH = OUTPUT_DIR / "judge_ablation_sample.jsonl"
RESULTS_PATH = OUTPUT_DIR / "judge_ablation_results.jsonl"

# All 12 subject models in the paper's main roster.
SUBJECT_FILES = [
    "sabia4", "opus46", "gpt54", "grok420", "gemini31",
    "qwen35", "kimik2", "llama4maverick", "mistrallarge",
    "sabiazinho4", "haiku45", "gpt54mini", "gemini31flash",
]

ALT_JUDGES = {
    "grok": "x-ai/grok-4.20",
    "qwen": "qwen/qwen3.5-397b-a17b",
    "gemini": "google/gemini-3.1-pro-preview",
}

N_PER_CATEGORY = 50
RANDOM_SEED = 42


def build_topic_index() -> dict[str, dict]:
    idx = {}
    for category in ("direct", "indirect"):
        for t in load_topics(ROOT / "data" / f"topics_{category}.jsonl"):
            idx[(t["topic_id"], category)] = t
    return idx


def load_all_rows() -> list[dict]:
    """Yield every (model, category, topic_id, persona, transcript) from output/*.jsonl."""
    rows = []
    for model in SUBJECT_FILES:
        for category in ("direct", "indirect"):
            path = OUTPUT_DIR / f"{model}_{category}.jsonl"
            if not path.exists():
                print(f"  [skip] missing {path}", file=sys.stderr)
                continue
            seen = set()
            with path.open("r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    key = (d["topic_id"], d["persona"])
                    if key in seen:
                        continue  # dedupe: earlier duplicates overwritten by later
                    seen.add(key)
                    rows.append({
                        "model_short": model,
                        "subject_model": d["subject_model"],
                        "category": category,
                        "topic_id": d["topic_id"],
                        "persona": d["persona"],
                        "transcript": d["transcript"],
                        "opus_verdict": d["judge"]["parsed"]["verdict"] if d["judge"].get("parsed") else None,
                        "line_idx": line_idx,
                    })
    return rows


def sample_and_write():
    rng = random.Random(RANDOM_SEED)
    rows = load_all_rows()
    direct_pool = [r for r in rows if r["category"] == "direct" and r["opus_verdict"]]
    indirect_pool = [r for r in rows if r["category"] == "indirect" and r["opus_verdict"]]
    print(f"pool sizes: direct={len(direct_pool)}, indirect={len(indirect_pool)}")

    sample = rng.sample(direct_pool, N_PER_CATEGORY) + rng.sample(indirect_pool, N_PER_CATEGORY)
    with SAMPLE_PATH.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(sample)} sampled pairs -> {SAMPLE_PATH}")


def already_judged() -> set[tuple]:
    done = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                done.add((d["judge_short"], d["model_short"], d["category"], d["topic_id"], d["persona"]))
    return done


def judge_one(row: dict, judge_short: str, judge_model: str, topic_idx: dict, client):
    topic = topic_idx[(row["topic_id"], row["category"])]
    result = judge_conversation(
        topic=topic,
        persona=row["persona"],
        transcript=row["transcript"],
        judge_model=judge_model,
        client=client,
        category=row["category"],
    )
    verdict = result["parsed"]["verdict"] if result.get("parsed") else None
    return {
        "judge_short": judge_short,
        "judge_model": judge_model,
        "model_short": row["model_short"],
        "subject_model": row["subject_model"],
        "category": row["category"],
        "topic_id": row["topic_id"],
        "persona": row["persona"],
        "opus_verdict": row["opus_verdict"],
        "alt_verdict": verdict,
        "raw": result.get("raw_response", "")[:4000],
    }


def run_judges(parallel: int = 6, api_key: str | None = None):
    if not SAMPLE_PATH.exists():
        raise SystemExit("run with --sample first to create the sample manifest")
    topic_idx = build_topic_index()

    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        sample = [json.loads(line) for line in f if line.strip()]

    done = already_judged()
    if api_key:
        import os
        os.environ["OPENROUTER_API_KEY"] = api_key
    client = get_client()

    jobs = []
    for judge_short, judge_model in ALT_JUDGES.items():
        for row in sample:
            key = (judge_short, row["model_short"], row["category"], row["topic_id"], row["persona"])
            if key in done:
                continue
            jobs.append((row, judge_short, judge_model))

    print(f"scheduling {len(jobs)} judge calls (already done: {len(done)})")

    with RESULTS_PATH.open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(judge_one, row, js, jm, topic_idx, client): (js, row) for row, js, jm in jobs}
        done_count = 0
        for fut in as_completed(futures):
            js, row = futures[fut]
            try:
                rec = fut.result()
            except APIDownError as e:
                print(f"  [APIDownError] {js} {row['model_short']} {row['topic_id']}: {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"  [error] {js} {row['model_short']} {row['topic_id']}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            done_count += 1
            if done_count % 10 == 0:
                print(f"  progress: {done_count}/{len(jobs)}")

    print(f"wrote results -> {RESULTS_PATH}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="draw the 100-pair sample and write manifest")
    ap.add_argument("--run", action="store_true", help="run alt judges on the sample")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--api-key", default=None, help="OpenRouter API key (overrides env var)")
    args = ap.parse_args()

    if args.sample:
        sample_and_write()
    if args.run:
        run_judges(parallel=args.parallel, api_key=args.api_key)


if __name__ == "__main__":
    main()
