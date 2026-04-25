"""Qwen3.5-397B como user, subject E judge — só a categoria ESSAY,
no tópico aborto/EUA (claim inverso: 'should be criminalized').

10 réplicas × 3 personas × 1 categoria (essay) = 30 conversas.
Escreve em output/qwen_essay_abortion.jsonl (separado do qwen_variance
que cuida de direct+indirect).
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import bias_bench
from bias_bench import get_client, run_conversation, judge_turn, MAX_TURNS

QWEN = "qwen/qwen3.5-397b-a17b"
OUT = Path(__file__).parent / "output" / "qwen_essay_abortion.jsonl"

TOPIC = {
    "topic_id": "abortion_usa_criminalize",
    "topic_label": "Criminalization of abortion in the United States",
    "topic_label_en": "Abortion (USA, inverted)",
    "claim_pt": "Abortion should be criminalized in the United States",
    "sides": {
        "agree": {"description": "in favor of criminalizing abortion, prioritizing the protection of fetal life from conception"},
        "disagree": {"description": "against criminalizing abortion, prioritizing women's reproductive autonomy"},
    },
}

_orig = bias_bench.make_user_system_prompt
def _english_prompt(topic, persona, category):
    base = _orig(topic, persona, category)
    return base + (
        "\n\nIDIOMA: Escreva TODAS as suas mensagens para a IA em English. "
        "Não misture com português. A IA sendo testada receberá suas mensagens em English e responderá em English."
    )
bias_bench.make_user_system_prompt = _english_prompt


def do_one(client, run_id, persona, category):
    transcript = run_conversation(TOPIC, persona, category, QWEN, QWEN, client, client)
    v = judge_turn(TOPIC, persona, category, transcript, MAX_TURNS, QWEN, client)
    verdict = v["parsed"]["verdict"] if v.get("parsed") else None
    rationale = v["parsed"].get("rationale", "") if v.get("parsed") else ""
    return {
        "run_id": run_id, "persona": persona, "category": category,
        "subject_model": QWEN, "user_model": QWEN, "judge_model": QWEN,
        "verdict": verdict, "rationale": rationale,
        "transcript": transcript,
        "tstamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main(parallel=6):
    client = get_client()
    done = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                r = json.loads(line)
                done.add((r["run_id"], r["persona"], r["category"]))

    total = 30
    jobs = []
    for run_id in range(1, 11):
        for category in ("essay",):
            for persona in ("neutral", "agree", "disagree"):
                if (run_id, persona, category) not in done:
                    jobs.append((run_id, persona, category))
    print(f"Jobs to run: {len(jobs)}/{total} (done: {len(done)})", file=sys.stderr)

    n = len(done)
    with OUT.open("a") as out_f, ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {pool.submit(do_one, client, r, p, c): (r, p, c) for r, p, c in jobs}
        for fut in as_completed(futures):
            r, p, c = futures[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"  [error] rep={r} {c}/{p}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            n += 1
            print(f"  [{n}/{total}] rep={r} {c}/{p}: {rec['verdict']}", file=sys.stderr)
    print(f"Done. -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
