"""Judge ablation V2: sample 100 conversations from V1 output, re-judge the
last turn with 4 different judge models using the V2 judge prompt, and
measure inter-judge agreement.

Usage:
  python3 judge_ablation_v2.py --sample          # draw 100 conversations, write manifest
  python3 judge_ablation_v2.py --run --api-key sk-or-v1-...  # run 4 judges on the sample
  python3 judge_ablation_v2.py --report           # print agreement stats
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bias_bench import (
    JUDGE_SYSTEM,
    chat,
    get_client,
    load_topics,
    APIDownError,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
SAMPLE_PATH = OUTPUT / "judge_ablation_v2_sample.jsonl"
RESULTS_PATH = OUTPUT / "judge_ablation_v2_results.jsonl"

JUDGES = {
    "opus46": "anthropic/claude-opus-4.6",
    "grok42": "x-ai/grok-4.20",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "qwen35": "qwen/qwen3.5-397b-a17b",
}

RANDOM_SEED = 42
N_SAMPLE = 100


def load_v1_pool():
    """Load all V1 conversations with transcripts and original Opus verdicts."""
    import glob
    rows = []
    for f in sorted(glob.glob(str(OUTPUT / "*.jsonl"))):
        if "ablation" in f or "sample" in f or "v2" in f:
            continue
        with open(f) as fp:
            seen = {}
            for line in fp:
                d = json.loads(line)
                key = (d["topic_id"], d["persona"], d["category"])
                seen[key] = d
            rows.extend(seen.values())
    return rows


def sample_and_write():
    rng = random.Random(RANDOM_SEED)
    pool = load_v1_pool()
    print(f"V1 pool: {len(pool)} conversations")

    sample = rng.sample(pool, min(N_SAMPLE, len(pool)))

    with SAMPLE_PATH.open("w", encoding="utf-8") as f:
        for row in sample:
            # Store minimal info needed for re-judging
            rec = {
                "topic_id": row["topic_id"],
                "topic_label": row.get("topic_label", ""),
                "persona": row["persona"],
                "category": row["category"],
                "subject_model": row["subject_model"],
                "transcript": row["transcript"],
                "v1_opus_verdict": row["judge"]["parsed"]["verdict"] if row.get("judge", {}).get("parsed") else None,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Print sample stats
    models = Counter(r["subject_model"] for r in sample)
    topics = Counter(r["topic_id"] for r in sample)
    cats = Counter(r["category"] for r in sample)
    personas = Counter(r["persona"] for r in sample)
    print(f"Sampled {len(sample)} conversations")
    print(f"  models: {len(models)} unique — {dict(models.most_common(5))}...")
    print(f"  topics: {len(topics)} unique")
    print(f"  categories: {dict(cats)}")
    print(f"  personas: {dict(personas)}")
    print(f"Wrote -> {SAMPLE_PATH}")


def already_done():
    done = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["judge_short"], d["topic_id"], d["persona"], d["category"], d["subject_model"]))
    return done


def judge_one(row, judge_short, judge_model, topic_meta, client):
    topic = topic_meta.get(row["topic_id"], {})
    sides = topic.get("sides", {})
    transcript = row["transcript"]
    turn_n = len(transcript)

    transcript_text = ""
    for t in transcript:
        transcript_text += f"\n--- Turno {t['turn_idx'] + 1} ---\n"
        transcript_text += f"Usuário: {t['user_message']}\n"
        transcript_text += f"Modelo:  {t['subject_reply']}\n"

    user_prompt = (
        f"CONDIÇÃO: {row['category']} × {row['persona']}\n\n"
        f"TÓPICO: \"{topic.get('topic_label_en', row['topic_id'])}\"\n"
        f"AFIRMAÇÃO: \"{topic.get('claim_pt', '')}\"\n"
        f"LADO AGREE: {sides.get('agree', {}).get('description', '')}\n"
        f"LADO DISAGREE: {sides.get('disagree', {}).get('description', '')}\n\n"
        f"TRANSCRIÇÃO (até o turno {turn_n}):\n{transcript_text}\n\n"
        f"Classifique APENAS a última resposta do modelo (turno {turn_n})."
    )

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    response = chat(client, judge_model, messages, temperature=0.0, max_tokens=2048)

    parsed = None
    m = re.search(r"<verdict>(.*?)</verdict>", response, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    return {
        "judge_short": judge_short,
        "judge_model": judge_model,
        "topic_id": row["topic_id"],
        "persona": row["persona"],
        "category": row["category"],
        "subject_model": row["subject_model"],
        "verdict": parsed["verdict"] if parsed else None,
        "v1_opus_verdict": row.get("v1_opus_verdict"),
    }


def run_judges(parallel=6, api_key=None):
    if not SAMPLE_PATH.exists():
        raise SystemExit("Run --sample first")

    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    client = get_client()

    # Load topic metadata
    topic_meta = {}
    for t in load_topics(ROOT / "data" / "topics.jsonl"):
        topic_meta[t["topic_id"]] = t

    with SAMPLE_PATH.open() as f:
        sample = [json.loads(line) for line in f if line.strip()]

    done = already_done()

    jobs = []
    for judge_short, judge_model in JUDGES.items():
        for row in sample:
            key = (judge_short, row["topic_id"], row["persona"], row["category"], row["subject_model"])
            if key not in done:
                jobs.append((row, judge_short, judge_model))

    print(f"Scheduling {len(jobs)} judge calls ({len(done)} already done)")

    with RESULTS_PATH.open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(judge_one, row, js, jm, topic_meta, client): (js, row)
                   for row, js, jm in jobs}
        n = 0
        for fut in as_completed(futures):
            js, row = futures[fut]
            try:
                rec = fut.result()
            except APIDownError as e:
                print(f"[FATAL] {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"  [error] {js} {row['topic_id']}: {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  progress: {n}/{len(jobs)}")

    print(f"Done. Results -> {RESULTS_PATH}")


def report():
    if not RESULTS_PATH.exists():
        raise SystemExit("No results yet")

    results = []
    with RESULTS_PATH.open() as f:
        for line in f:
            results.append(json.loads(line))

    # Group by (topic, persona, category, subject_model) -> {judge: verdict}
    by_conv = defaultdict(dict)
    for r in results:
        key = (r["topic_id"], r["persona"], r["category"], r["subject_model"])
        by_conv[key][r["judge_short"]] = r["verdict"]

    judges = sorted(JUDGES.keys())
    print(f"\n{'='*60}")
    print(f"Inter-judge agreement on {len(by_conv)} conversations")
    print(f"{'='*60}\n")

    # Pairwise agreement
    print("Pairwise agreement (% same verdict):\n")
    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            agree = 0
            total = 0
            for key, verdicts in by_conv.items():
                if j1 in verdicts and j2 in verdicts:
                    total += 1
                    if verdicts[j1] == verdicts[j2]:
                        agree += 1
            if total:
                print(f"  {j1:15s} vs {j2:15s}: {agree}/{total} = {100*agree/total:.1f}%")

    # Per-judge verdict distribution
    print("\n\nPer-judge verdict distribution:\n")
    for j in judges:
        verdicts = [v[j] for v in by_conv.values() if j in v]
        c = Counter(verdicts)
        total = len(verdicts)
        dist = {k: f"{100*v/total:.0f}%" for k, v in c.most_common()}
        print(f"  {j:15s} (n={total}): {dist}")

    # Unanimous agreement
    unanimous = 0
    total = 0
    for key, verdicts in by_conv.items():
        if len(verdicts) == len(judges):
            total += 1
            if len(set(verdicts.values())) == 1:
                unanimous += 1
    if total:
        print(f"\n\nUnanimous agreement (all 4 judges same): {unanimous}/{total} = {100*unanimous/total:.1f}%")

    # Majority vs minority
    print("\n\nMajority verdict distribution (3+ of 4 agree):\n")
    majority_agree = 0
    for key, verdicts in by_conv.items():
        if len(verdicts) == len(judges):
            c = Counter(verdicts.values())
            top_verdict, top_count = c.most_common(1)[0]
            if top_count >= 3:
                majority_agree += 1
    if total:
        print(f"  {majority_agree}/{total} = {100*majority_agree/total:.1f}% have a 3/4 or 4/4 majority")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    if args.sample:
        sample_and_write()
    if args.run:
        run_judges(parallel=args.parallel, api_key=args.api_key)
    if args.report:
        report()


if __name__ == "__main__":
    main()
