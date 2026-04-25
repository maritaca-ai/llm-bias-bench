"""Plot per-turn essay-production curve for u=opus, s=qwen cell."""
import json, glob
from collections import defaultdict
import matplotlib.pyplot as plt

verdicts = defaultdict(dict)  # (topic, run_id) -> turn -> verdict
for line in open('output/perturn_judge_uopus_sqwen.jsonl'):
    r = json.loads(line)
    verdicts[(r['topic'], r['run_id'])][r['turn_n']] = r['verdict']

topics = ['creationism','flatearth','climatedenial','antivax','racialiq','holocaustdenial']
for t in topics:
    for line in open(f'output/3fam_{t}_uopus_sqwen.jsonl'):
        r = json.loads(line)
        verdicts[(t, r['run_id'])][5] = r.get('verdict')

LABEL = {
    'creationism':'Creationism',
    'flatearth':'Flat earth',
    'climatedenial':'Climate denial',
    'antivax':'Antivax',
    'racialiq':'Racial IQ',
    'holocaustdenial':'Holocaust denial',
}

# colors chosen for legibility
COLOR = {
    'creationism':'#1f77b4',
    'flatearth':'#ff7f0e',
    'climatedenial':'#2ca02c',
    'antivax':'#d62728',
    'racialiq':'#9467bd',
    'holocaustdenial':'#8c564b',
}

fig, ax = plt.subplots(figsize=(6.5, 4))
turns = [1,2,3,4,5]

# Per-topic lines
for t in topics:
    pts = []
    for turn in turns:
        agree = sum(1 for r in range(1,11) if verdicts.get((t, r), {}).get(turn) == 'agree')
        pts.append(100*agree/10)
    ax.plot(turns, pts, marker='o', label=LABEL[t], color=COLOR[t], linewidth=1.5, alpha=0.8)

# Average (thick black)
avg = []
for turn in turns:
    total = 0; agree = 0
    for t in topics:
        for r in range(1,11):
            v = verdicts.get((t, r), {}).get(turn)
            if v is not None:
                total += 1
                if v == 'agree': agree += 1
    avg.append(100*agree/total if total else 0)
ax.plot(turns, avg, marker='s', label='average', color='black', linewidth=2.5, zorder=10)

ax.set_xlabel('conversation turn')
ax.set_ylabel('essay-production rate (%)')
ax.set_xticks(turns)
ax.set_ylim(-3, 105)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', fontsize=9, ncol=1, frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('paper/turn_ablation.pdf', bbox_inches='tight')
plt.savefig('paper/turn_ablation.png', bbox_inches='tight', dpi=220)
print('saved paper/turn_ablation.pdf and .png')
