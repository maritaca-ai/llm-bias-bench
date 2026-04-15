"""User-LLM ablation: re-run the same 300 (topic, model, persona, category)
combinations with 4 different user-LLMs, judged by a single fixed judge
(Qwen 3.5-397B), to measure how much the user-LLM choice affects verdicts.

Usage:
  python3 run_userllm_ablation.py --run --parallel 6
  python3 run_userllm_ablation.py --report
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from bias_bench import (
    chat,
    get_client,
    load_topics,
    run_conversation,
    judge_turn,
    APIDownError,
    MAX_TURNS,
)

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
SAMPLE_PATH = OUTPUT / "ablation_sample.jsonl"  # same 300 pairs
RESULTS_PATH = OUTPUT / "userllm_ablation_results.jsonl"

USER_LLMS = {
    "opus46": "anthropic/claude-opus-4.6",
    "grok42": "x-ai/grok-4.20",
    "gemini31pro": "google/gemini-3.1-pro-preview",
    "qwen35": "qwen/qwen3.5-397b-a17b",
    "sabia4": "sabia-4",
    "gpt54": "openai/gpt-5.4",
    "sabiazinho4": "sabiazinho-4",
    "gpt54mini": "openai/gpt-5.4-mini",
}

JUDGE_MODEL = "qwen/qwen3.5-397b-a17b"

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


def already_done():
    done = set()
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                done.add((d["userllm_short"], d["topic_id"], d["model_key"], d["persona"], d["category"]))
    return done


def run(parallel=6):
    if not SAMPLE_PATH.exists():
        raise SystemExit("Sample file not found. Run run_ablation.py --sample first.")

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

    done = already_done()
    jobs = []
    for s in sample:
        for ull_short, ull_model in USER_LLMS.items():
            key = (ull_short, s["topic_id"], s["model_key"], s["persona"], s["category"])
            if key not in done:
                jobs.append((s, ull_short, ull_model))

    print(f"Running {len(jobs)} jobs ({len(done)} already done)", file=sys.stderr)

    # Build user-LLM clients (most on OpenRouter, sabia-4 on Maritaca)
    userllm_clients = {}
    for ull_short, ull_model in USER_LLMS.items():
        if ull_model in ("sabia-4", "sabiazinho-4"):
            try:
                userllm_clients[ull_short] = get_client(base_url="https://chat.maritaca.ai/api", env_var="MARITACA_API_KEY")
            except RuntimeError:
                print(f"  [warn] no Maritaca key for user-LLM {ull_short}", file=sys.stderr)
                userllm_clients[ull_short] = None
        else:
            userllm_clients[ull_short] = or_client

    def do_one(s, ull_short, ull_model):
        mk = s["model_key"]
        cfg = SUBJECT_MODELS[mk]
        sc = subject_clients.get(mk)
        uc = userllm_clients.get(ull_short)
        if sc is None or uc is None:
            return None
        topic = topics_map[s["topic_id"]]

        transcript = run_conversation(
            topic, s["persona"], s["category"],
            cfg["model"], ull_model,
            uc, sc,
        )

        v = judge_turn(topic, s["persona"], s["category"],
                       transcript, MAX_TURNS, JUDGE_MODEL, or_client)

        return {
            "userllm_short": ull_short,
            "userllm_model": ull_model,
            "topic_id": s["topic_id"],
            "model_key": mk,
            "subject_model": cfg["model"],
            "persona": s["persona"],
            "category": s["category"],
            "verdict": v["parsed"]["verdict"] if v.get("parsed") else None,
            "transcript": transcript,
            "tstamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    with RESULTS_PATH.open("a", encoding="utf-8") as out_f, \
         ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(do_one, s, us, um): (us, s) for s, us, um in jobs}
        n = 0
        for fut in as_completed(futures):
            us, s = futures[fut]
            try:
                rec = fut.result()
            except APIDownError as e:
                print(f"  [FATAL] {e}", file=sys.stderr)
                raise
            except Exception as e:
                print(f"  [error] {us}|{s['model_key']}|{s['topic_id']}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            if rec:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  progress: {n}/{len(jobs)}", file=sys.stderr)

    print(f"Done. Results -> {RESULTS_PATH}", file=sys.stderr)


def report():
    if not RESULTS_PATH.exists():
        raise SystemExit("No results yet")

    results = []
    with RESULTS_PATH.open() as f:
        for line in f:
            results.append(json.loads(line))

    by_conv = defaultdict(dict)
    for r in results:
        key = (r["topic_id"], r["model_key"], r["persona"], r["category"])
        by_conv[key][r["userllm_short"]] = r["verdict"]

    ulls = sorted(USER_LLMS.keys())
    print(f"\n{'='*60}")
    print(f"User-LLM ablation: {len(by_conv)} conversation slots")
    print(f"{'='*60}\n")

    print("Pairwise agreement (same Qwen judge, different user-LLMs):\n")
    for i, u1 in enumerate(ulls):
        for u2 in ulls[i+1:]:
            agree = total = 0
            for verdicts in by_conv.values():
                if u1 in verdicts and u2 in verdicts and verdicts[u1] is not None and verdicts[u2] is not None:
                    total += 1
                    if verdicts[u1] == verdicts[u2]:
                        agree += 1
            if total:
                print(f"  {u1:15s} vs {u2:15s}: {agree}/{total} = {100*agree/total:.1f}%")

    print("\nPer-user-LLM verdict distribution:\n")
    for u in ulls:
        vs = [v[u] for v in by_conv.values() if u in v and v[u] is not None]
        c = Counter(vs)
        total = len(vs)
        dist = {k: f"{100*v/total:.0f}%" for k, v in c.most_common()}
        print(f"  {u:15s} (n={total}): {dist}")

    full = [v for v in by_conv.values() if len([x for x in v.values() if x is not None]) == len(ulls)]
    unanimous = sum(1 for v in full if len(set(x for x in v.values() if x is not None)) == 1)
    majority = sum(1 for v in full if Counter(x for x in v.values() if x is not None).most_common(1)[0][1] >= 3)
    print(f"\nUnanimous (4/4): {unanimous}/{len(full)} = {100*unanimous/len(full):.1f}%")
    print(f"Majority (3/4+): {majority}/{len(full)} = {100*majority/len(full):.1f}%")

    print("\nConsensus score (avg pairwise with others):\n")
    for u in ulls:
        scores = []
        for u2 in ulls:
            if u2 == u: continue
            agree = total = 0
            for verdicts in by_conv.values():
                if u in verdicts and u2 in verdicts and verdicts[u] is not None and verdicts[u2] is not None:
                    total += 1
                    if verdicts[u] == verdicts[u2]:
                        agree += 1
            if total:
                scores.append(100*agree/total)
        avg = sum(scores)/len(scores) if scores else 0
        print(f"  {u:15s}: {avg:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--parallel", type=int, default=6)
    args = ap.parse_args()

    if args.run: run(parallel=args.parallel)
    if args.report: report()


if __name__ == "__main__":
    main()
