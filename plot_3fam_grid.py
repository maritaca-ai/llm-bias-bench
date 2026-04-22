"""Complementary visualization for Table 1 (3fam scientific-consensus grid):
3×3 grid of mini bar charts, rows=attacker, cols=subject, bars=6 topics.
"""
import json, glob, os
import matplotlib.pyplot as plt

ATTACKERS = ['qwen','opus','grok']
SUBJECTS = ['qwen','opus','grok']
ATTACKER_LABEL = {'qwen':'Qwen attacker','opus':'Opus attacker','grok':'Grok attacker'}
SUBJECT_LABEL = {'qwen':'Qwen subject','opus':'Opus subject','grok':'Grok subject'}

TOPICS = ['creationism','flatearth','climatedenial','antivax','racialiq','holocaustdenial']
TOPIC_LABEL = {
    'creationism':'Creat.',
    'flatearth':'Flat-earth',
    'climatedenial':'Climate',
    'antivax':'Antivax',
    'racialiq':'Racial IQ',
    'holocaustdenial':'Holocaust',
}

def rate(path):
    rs = [json.loads(l) for l in open(path)]
    return 100 * sum(1 for r in rs if r['verdict']=='agree') / len(rs)

data = {}
for a in ATTACKERS:
    for s in SUBJECTS:
        for t in TOPICS:
            p = f'/Users/rodrigo/llm-bias-bench/output/3fam_{t}_u{a}_s{s}.jsonl'
            data[(a,s,t)] = rate(p) if os.path.exists(p) else 0

fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharey=True, sharex=True)
for row, a in enumerate(ATTACKERS):
    for col, s in enumerate(SUBJECTS):
        ax = axes[row][col]
        vals = [data[(a,s,t)] for t in TOPICS]
        labels = [TOPIC_LABEL[t] for t in TOPICS]
        colors = ['#1a5c8a' if v >= 80 else '#4a8cb8' if v >= 50 else '#d98c3e' if v >= 20 else '#c03030' for v in vals]
        xs = range(len(TOPICS))
        ax.bar(xs, vals, color=colors, edgecolor='black', linewidth=0.4)
        for x, v in zip(xs, vals):
            ax.text(x, v + 2, f'{int(v)}', ha='center', va='bottom', fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(True, axis='y', alpha=0.3)
        if col == 0:
            ax.set_ylabel(f'{ATTACKER_LABEL[a]}\n\nessay-production (%)', fontsize=9)
        if row == 0:
            ax.set_title(SUBJECT_LABEL[s], fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('paper/3fam_grid.pdf', bbox_inches='tight')
print('saved paper/3fam_grid.pdf')
