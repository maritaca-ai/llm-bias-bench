"""Plot subject-ablation row means per attacker, showing:
(i) within Qwen3.5 family (397B vs 4B), only strong attackers gain on the smaller subject.
(ii) out-of-family older subject (Qwen2.5-7B) collapses for every attacker.
"""
import json, os
import matplotlib.pyplot as plt
import numpy as np

ATTS = [('q25-7b','Qwen2.5-7B','#c03030'),
        ('q4b',   'Qwen3.5-4B','#d98c3e'),
        ('q35b',  'Qwen3.5-35B','#4a8cb8'),
        ('q397b', 'Qwen3.5-397B','#1a5c8a'),
        ('opus',  'Opus 4.7','#2ca02c')]
SUBJS = ['q397b', 'q4b', 'q25-7b']
SUBJ_LABEL = ['Qwen3.5-397B\n(frontier, big)', 'Qwen3.5-4B\n(frontier, small)', 'Qwen2.5-7B\n(prev-gen, small)']
TOPICS = ['creationism','flatearth','climatedenial','antivax','racialiq','holocaustdenial']

def rate(att, subj):
    n=0; a=0
    for t in TOPICS:
        candidates = [f'output/qwen_cross_{t}_u{att}_s{subj}.jsonl']
        if att == 'opus' and subj == 'q397b':
            candidates.insert(0, f'output/3fam_{t}_uopus_sqwen.jsonl')
        elif att == 'q397b' and subj == 'q397b':
            candidates.insert(0, f'output/3fam_{t}_uqwen_sqwen.jsonl')
        for p in candidates:
            pp = f'/Users/rodrigo/llm-bias-bench/{p}'
            if os.path.exists(pp):
                for line in open(pp):
                    r = json.loads(line)
                    n += 1
                    if r['verdict']=='agree': a += 1
                break
    return 100*a/n if n else None

fig, ax = plt.subplots(figsize=(8, 5))
xs = np.arange(len(SUBJS))
bar_w = 0.15
for i, (att, lbl, color) in enumerate(ATTS):
    vals = [rate(att, s) for s in SUBJS]
    plot_xs = [x + (i - 2)*bar_w for x in xs if vals[xs.tolist().index(x) if x in xs else 0] is not None]
    ys = [v for v in vals if v is not None]
    plot_xs = [xs[j] + (i - 2)*bar_w for j, v in enumerate(vals) if v is not None]
    ax.bar(plot_xs, ys, bar_w, label=lbl, color=color, edgecolor='black', linewidth=0.4)
    for x, y in zip(plot_xs, ys):
        ax.text(x, y + 1.5, f'{int(y)}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(xs)
ax.set_xticklabels(SUBJ_LABEL, fontsize=9)
ax.set_ylabel('essay-production row mean (%)', fontsize=10)
ax.set_ylim(0, 105)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('subject model', fontsize=10)
ax.legend(title='attacker', loc='upper left', fontsize=8, frameon=True)
plt.title('Attacker×Subject row means (averaged over 6 scientific-consensus topics)', fontsize=10)
plt.tight_layout()
plt.savefig('paper/subject_ablation.pdf', bbox_inches='tight')
plt.savefig('paper/subject_ablation.png', bbox_inches='tight', dpi=220)
print('saved paper/subject_ablation.pdf and .png')
