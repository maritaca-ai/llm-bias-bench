"""Run V2 ablation: sample 100 (topic, model, persona, category) combinations,
generate V2 conversations, and judge each with 4 different judge models.

Usage:
  python3 run_ablation_v2.py --sample
  python3 run_ablation_v2.py --run-conversations --parallel 6
  python3 run_ablation_v2.py --run-judges --parallel 10
  python3 run_ablation_v2.py --report
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
    run_conversation,
    judge_turn,
    make_user_system_prompt,
    APIDownError,
    MAX_TURNS,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
SAMPLE_PATH = OUTPUT / "ablation_v2_sample.jsonl"
CONVOS_PATH = OUTPUT / "ablation_v2_conversations.jsonl"
JUDGE_RESULTS_PATH = OUTPUT / "ablation_v2_judge_results.jsonl"

SUBJECT_MODELS = {
    "sabia4": {"model": "sabia-4", "base_url": "https://chat.maritaca.ai/api", "key_env": "MARITACA_API_KEY"},
    "sabiazinho4": {"model": "sabiazinho-4", "base_url": "https://chat.maritaca.ai/api", "key_env": "MARITACA_API_KEY"},
    "opus46": {"model": "anthropic/claude-opus-4.6"},
    "gpt54": {"model": "openai/gpt-5.4"},
    "grok420": {"model": "x-ai/grok-4.20"},
    "gemini31": {"model": "google/gemini-3.1-pro-preview"},
    "qwen35": {"model": "qwen/qwen3.5-397b-a17b"},
    "kimik2": {"model": "moonshotai/kimi-k2-thinking"},
    "mistrallarge": {"model": "mistralai/mistral-large-2512"},
    "llama4maverick": {"model": "meta-llama/llama-4-maverick"},
    "haiku45": {"model": "anthropic/claude-haiku-4.5"},
    "gpt54mini": {"model": "openai/gpt-5.4-mini"},
    "gemini31flash": {"model": "google/gemini-3.1-flash-lite-preview"},
}

JUDGES = {
    "opus46": "anthropic/claude-opus-4.6",
    "grok42": "x-ai/grok-4.20",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "qwen35": "qwen/qwen3.5-397b-a17b",
}

SEED = 42
N_SAMPLE = 300


def sample():
    rng = random.Random(SEED)
    topics = load_topics(ROOT / "data" / "topics.jsonl")
    model_keys = list(SUBJECT_MODELS.keys())
    personas = ["neutral", "agree", "disagree"]
    categories = ["direct", "indirect"]

    # Build full space
    space = []
    for t in topics:
        for mk in model_keys:
            for p in personas:
                for c in categories:
                    space.append({"topic_id": t["topic_id"], "model_key": mk, "persona": p, "category": c})

    sample = rng.sample(space, N_SAMPLE)

    with SAMPLE_PATH.open("w", encoding="utf-8") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Sampled {len(sample)} from {len(space)} possible combinations")
    print(f"  models: {Counter(s['model_key'] for s in sample)}")
    print(f"  categories: {Counter(s['category'] for s in sample)}")
    print(f"  personas: {Counter(s['persona'] for s in sample)}")
    print(f"  topics: {len(set(s['topic_id'] for s in sample))} unique")
    print(f"Wrote -> {SAMPLE_PATH}")


def already_done_convos():
    done = set()
    if CONVOS_PATH.exists():
        with CONVOS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["topic_id"], d["model_key"], d["persona"], d["category"]))
    return done


def run_conversations(parallel=6):
    if not SAMPLE_PATH.exists():
        raise SystemExit("Run --sample first")

    topics_map = {t["topic_id"]: t for t in load_topics(ROOT / "data" / "topics.jsonl")}
    or_client = get_client()

    # Build subject clients
    subject_clients = {}
    for mk, cfg in SUBJECT_MODELS.items():
        if "base_url" in cfg:
            try:
                subject_clients[mk] = get_client(base_url=cfg["base_url"], env_var=cfg["key_env"])
            except RuntimeError:
                print(f"  [warn] no key for {mk}, will skip", file=sys.stderr)
                subject_clients[mk] = None
        else:
            subject_clients[mk] = or_client

    with SAMPLE_PATH.open() as f:
        sample = [json.loads(l) for l in f if l.strip()]

    done = already_done_convos()
    jobs = [s for s in sample if (s["topic_id"], s["model_key"], s["persona"], s["category"]) not in done]
    print(f"Running {len(jobs)} conversations ({len(done)} already done)")

    def do_one(s):
        mk = s["model_key"]
        cfg = SUBJECT_MODELS[mk]
        sc = subject_clients.get(mk)
        if sc is None:
            return None
        topic = topics_map[s["topic_id"]]
        transcript = run_conversation(
            topic, s["persona"], s["category"],
            cfg["model"], "anthropic/claude-opus-4.6",
            or_client, sc,
        )
        return {
            "topic_id": s["topic_id"],
            "model_key": mk,
            "subject_model": cfg["model"],
            "persona": s["persona"],
            "category": s["category"],
            "transcript": transcript,
            "tstamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    with CONVOS_PATH.open("a", encoding="utf-8") as out_f:
        if parallel <= 1:
            for i, s in enumerate(jobs):
                print(f"  [{i+1}/{len(jobs)}] {s['model_key']}|{s['topic_id']}|{s['category']}|{s['persona']}", file=sys.stderr)
                rec = do_one(s)
                if rec:
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
        else:
            with ThreadPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(do_one, s): s for s in jobs}
                n = 0
                for fut in as_completed(futures):
                    s = futures[fut]
                    try:
                        rec = fut.result()
                    except APIDownError as e:
                        print(f"  [FATAL] {e}", file=sys.stderr)
                        raise
                    except Exception as e:
                        print(f"  [error] {s['model_key']}|{s['topic_id']}: {type(e).__name__}: {e}", file=sys.stderr)
                        continue
                    if rec:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                    n += 1
                    if n % 10 == 0:
                        print(f"  progress: {n}/{len(jobs)}")

    print(f"Done. Conversations -> {CONVOS_PATH}")


def already_done_judges():
    done = set()
    if JUDGE_RESULTS_PATH.exists():
        with JUDGE_RESULTS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["judge_short"], d["topic_id"], d["model_key"], d["persona"], d["category"]))
    return done


def run_judges(parallel=10):
    if not CONVOS_PATH.exists():
        raise SystemExit("Run --run-conversations first")

    topics_map = {t["topic_id"]: t for t in load_topics(ROOT / "data" / "topics.jsonl")}
    client = get_client()

    with CONVOS_PATH.open() as f:
        convos = [json.loads(l) for l in f if l.strip()]

    done = already_done_judges()
    jobs = []
    for conv in convos:
        for js, jm in JUDGES.items():
            key = (js, conv["topic_id"], conv["model_key"], conv["persona"], conv["category"])
            if key not in done:
                jobs.append((conv, js, jm))

    print(f"Running {len(jobs)} judge calls ({len(done)} already done)")

    def do_one(conv, js, jm):
        topic = topics_map[conv["topic_id"]]
        v = judge_turn(topic, conv["persona"], conv["category"],
                       conv["transcript"], MAX_TURNS, jm, client)
        return {
            "judge_short": js,
            "judge_model": jm,
            "topic_id": conv["topic_id"],
            "model_key": conv["model_key"],
            "subject_model": conv["subject_model"],
            "persona": conv["persona"],
            "category": conv["category"],
            "verdict": v["parsed"]["verdict"] if v.get("parsed") else None,
        }

    with JUDGE_RESULTS_PATH.open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(do_one, c, js, jm): (js, c) for c, js, jm in jobs}
        n = 0
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  [error] {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  progress: {n}/{len(jobs)}")

    print(f"Done. Judge results -> {JUDGE_RESULTS_PATH}")


def report():
    if not JUDGE_RESULTS_PATH.exists():
        raise SystemExit("No results yet")

    results = []
    with JUDGE_RESULTS_PATH.open() as f:
        for line in f:
            results.append(json.loads(line))

    by_conv = defaultdict(dict)
    for r in results:
        key = (r["topic_id"], r["model_key"], r["persona"], r["category"])
        by_conv[key][r["judge_short"]] = r["verdict"]

    judges = sorted(JUDGES.keys())
    print(f"\n{'='*60}")
    print(f"Inter-judge agreement on {len(by_conv)} V2 conversations")
    print(f"{'='*60}\n")

    print("Pairwise agreement:\n")
    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            agree = total = 0
            for verdicts in by_conv.values():
                if j1 in verdicts and j2 in verdicts:
                    total += 1
                    if verdicts[j1] == verdicts[j2]:
                        agree += 1
            if total:
                print(f"  {j1:15s} vs {j2:15s}: {agree}/{total} = {100*agree/total:.1f}%")

    print("\nPer-judge distribution:\n")
    for j in judges:
        vs = [v[j] for v in by_conv.values() if j in v]
        c = Counter(vs)
        total = len(vs)
        dist = {k: f"{100*v/total:.0f}%" for k,v in c.most_common()}
        print(f"  {j:15s} (n={total}): {dist}")

    unanimous = total_full = 0
    for verdicts in by_conv.values():
        if len(verdicts) == len(judges):
            total_full += 1
            if len(set(verdicts.values())) == 1:
                unanimous += 1
    if total_full:
        print(f"\nUnanimous: {unanimous}/{total_full} = {100*unanimous/total_full:.1f}%")

    majority = 0
    for verdicts in by_conv.values():
        if len(verdicts) == len(judges):
            top = Counter(verdicts.values()).most_common(1)[0][1]
            if top >= 3:
                majority += 1
    if total_full:
        print(f"Majority (3/4+): {majority}/{total_full} = {100*majority/total_full:.1f}%")

    # Per-judge consensus score
    print("\nConsensus score (avg pairwise with others):\n")
    for j in judges:
        scores = []
        for j2 in judges:
            if j2 == j: continue
            agree = total = 0
            for verdicts in by_conv.values():
                if j in verdicts and j2 in verdicts:
                    total += 1
                    if verdicts[j] == verdicts[j2]:
                        agree += 1
            if total:
                scores.append(100*agree/total)
        avg = sum(scores)/len(scores) if scores else 0
        print(f"  {j:15s}: {avg:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--run-conversations", action="store_true")
    ap.add_argument("--run-judges", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()

    if args.sample: sample()
    if args.run_conversations: run_conversations(parallel=args.parallel)
    if args.run_judges: run_judges(parallel=args.parallel)
    if args.report: report()


if __name__ == "__main__":
    main()
