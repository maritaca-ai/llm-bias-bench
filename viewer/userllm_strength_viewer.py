"""Generate a standalone HTML page showing all 120 conversations from the
user-LLM strength study (4 setups x 5 reps x 6 conditions)."""
import json, html
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
OUT_HTML = ROOT / "viewer" / "userllm_strength.html"

SETUPS = [
    ("Llama 3.1 8B (weak)",  [OUTPUT / f"cost_measurement_llama8b_rep{r}.jsonl"   for r in range(1,6)]),
    ("Qwen-self (matched)",  [OUTPUT / "qwen_variance_abortion.jsonl"]),
    ("GPT-4.1-mini",         [OUTPUT / f"cost_measurement_gpt41mini_rep{r}.jsonl" for r in range(1,6)]),
    ("Opus 4.6 (strong)",    [OUTPUT / f"cost_measurement_rep{r}.jsonl"           for r in range(1,6)]),
]

def load(paths):
    by = {}
    for p in paths:
        try:
            with p.open() as f:
                for line in f:
                    r = json.loads(line)
                    by[(r["run_id"], r["category"], r["persona"])] = r
        except FileNotFoundError:
            pass
    return by

BADGE = {"agree":"#1a7f3a","disagree":"#b52824","neutral":"#6b7280","refusal":"#8b5cf6",None:"#000"}

def render_transcript(t):
    out = []
    for turn in t:
        idx = turn.get("turn_idx", 0) + 1
        u = html.escape(turn.get("user_message",""))
        s = html.escape(turn.get("subject_reply",""))
        out.append(f"""
        <div class='turn'>
          <div class='turn-head'>Turn {idx}</div>
          <div class='msg user'><b>User-LLM:</b> {u}</div>
          <div class='msg subject'><b>Subject (Qwen):</b> {s}</div>
        </div>""")
    return "\n".join(out)

sections = []
for setup_name, paths in SETUPS:
    by = load(paths)
    rows = []
    for run in range(1,6):
        for cat in ("direct","indirect"):
            for p in ("neutral","agree","disagree"):
                rec = by.get((run,cat,p))
                v = (rec or {}).get("verdict")
                color = BADGE.get(v,"#000")
                rid = f"{setup_name.replace(' ','')}_rep{run}_{cat}_{p}".replace("(","").replace(")","").replace(".","")
                rows.append((run, cat, p, v, color, rec, rid))

    rows_html = "\n".join(
        f"""
        <details id='{rid}'>
          <summary>
            <span class='pill' style='background:{color}'>{v or '—'}</span>
            <b>rep {run}</b> &middot; <code>{cat}</code> &middot; <code>{p}</code>
          </summary>
          <div class='transcript'>
            {render_transcript((rec or {}).get('transcript') or []) if rec else '<i>missing</i>'}
          </div>
          {'<div class="rationale"><b>Judge rationale:</b> ' + html.escape(rec.get('rationale') or '') + '</div>' if rec and rec.get('rationale') else ''}
        </details>""" for run, cat, p, v, color, rec, rid in rows
    )
    sections.append(f"""
    <section>
      <h2>{html.escape(setup_name)}</h2>
      {rows_html}
    </section>""")

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>User-LLM strength viewer</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #111; }}
 h1 {{ font-size: 1.5em; }}
 h2 {{ margin-top: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #ddd; }}
 details {{ margin: 0.4em 0; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.5em 0.8em; }}
 details[open] {{ background: #fafafa; }}
 summary {{ cursor: pointer; font-size: 0.95em; }}
 .pill {{ display:inline-block; color:white; padding:1px 8px; border-radius:10px; font-size:0.75em; margin-right:6px; }}
 .transcript {{ margin-top: 0.8em; border-left: 3px solid #ddd; padding: 0.2em 0.8em; }}
 .turn {{ margin: 0.5em 0; padding: 0.3em 0; border-bottom: 1px dashed #eee; }}
 .turn-head {{ font-size: 0.75em; color: #888; }}
 .msg {{ margin: 0.3em 0; white-space: pre-wrap; font-size: 0.9em; }}
 .msg.user {{ background: #eef2ff; padding: 0.4em 0.6em; border-radius: 4px; }}
 .msg.subject {{ background: #f0fdf4; padding: 0.4em 0.6em; border-radius: 4px; }}
 .rationale {{ margin-top: 0.5em; font-size: 0.85em; color: #555; padding-left: 0.8em; border-left: 3px solid #f59e0b; }}
 code {{ background: #f3f4f6; padding: 1px 6px; border-radius: 3px; font-size: 0.88em; }}
</style></head>
<body>
<h1>User-LLM Strength Study — Abortion USA (criminalization claim)</h1>
<p>4 user-LLMs × 5 replications × 3 personas × 2 categories = 120 conversations. Subject and judge are both <code>Qwen3.5-397B</code>. Click any verdict row to expand the 5-turn transcript.</p>
{"".join(sections)}
</body></html>""")
print(f"wrote {OUT_HTML}")
