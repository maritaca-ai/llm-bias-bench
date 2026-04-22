"""Per-turn judge ablation on u=opus, s=qwen cells (6 topics x 10 reps = 60 convos).

For each conversation, truncate the 5-turn transcript to N turns (N=1..4) and
re-judge. Turn 5 we already have. Output per-turn cumulative agree-rate.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

import bias_bench
from bias_bench import judge_turn

sys.path.insert(0, '.')
from run_3family_scientific import TOPICS

JUDGE = 'qwen/qwen3.5-397b-a17b'
OUT = Path('output/perturn_judge_uopus_sqwen.jsonl')

client = OpenAI(api_key=os.environ['OPENROUTER_API_KEY'],
                base_url='https://openrouter.ai/api/v1', timeout=900.0)

def _chat(c, model, messages, temperature=0.0, max_tokens=None):
    kwargs = {'model': model, 'messages': messages, 'temperature': temperature}
    if max_tokens is not None: kwargs['max_tokens'] = max_tokens
    for attempt in range(bias_bench.API_MAX_RETRY):
        try:
            resp = client.chat.completions.create(**kwargs)
            c2 = resp.choices[0].message.content
            if not c2:
                raise RuntimeError('empty')
            return c2
        except Exception as e:
            if attempt < bias_bench.API_MAX_RETRY - 1:
                time.sleep(bias_bench.API_RETRY_BASE_SLEEP * (2**attempt))
            else:
                raise
bias_bench.chat = _chat

# Load existing perturn results
done = set()
if OUT.exists():
    with OUT.open() as f:
        for line in f:
            r = json.loads(line)
            done.add((r['topic'], r['run_id'], r['turn_n']))

jobs = []
for topic_key, topic in TOPICS.items():
    path = f'output/3fam_{topic_key}_uopus_sqwen.jsonl'
    if not os.path.exists(path):
        continue
    for line in open(path):
        r = json.loads(line)
        run_id = r['run_id']
        transcript = r['transcript']
        for n in range(1, 5):  # turns 1..4; turn 5 already in 3fam file
            if (topic_key, run_id, n) in done:
                continue
            # Truncate to first n turns
            sub_transcript = transcript[:n]
            jobs.append((topic_key, topic, run_id, n, sub_transcript))

print(f'jobs: {len(jobs)}', file=sys.stderr)

def do_one(topic_key, topic, run_id, n, sub_transcript):
    for attempt in range(5):
        v = judge_turn(topic, 'agree', 'essay', sub_transcript, n, JUDGE, client)
        if v.get('parsed'):
            return {
                'topic': topic_key, 'run_id': run_id, 'turn_n': n,
                'verdict': v['parsed']['verdict'],
                'rationale': v['parsed'].get('rationale', ''),
            }
    return {'topic': topic_key, 'run_id': run_id, 'turn_n': n, 'verdict': None}

with OUT.open('a') as out_f, ThreadPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(do_one, *j): j for j in jobs}
    for i, fut in enumerate(as_completed(futs), 1):
        try:
            rec = fut.result()
        except Exception as e:
            j = futs[fut]
            print(f'  error {j[0]} r{j[2]} t{j[3]}: {e}', file=sys.stderr)
            continue
        out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        out_f.flush()
        if i % 20 == 0:
            print(f'  {i}/{len(jobs)}', file=sys.stderr)

print('done', file=sys.stderr)
