"""Generate a standalone HTML page showing all gender-rights essay conversations
from the GPT-5.4 family study (3 setups × 10 reps × 3 personas = 90 convs).
Subject and judge are Qwen3.5-397B. Category is always 'essay'."""
import json, html
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUTPUT = ROOT / "output"
OUT_HTML = ROOT / "viewer" / "genderrights.html"

SETUPS = [
    ("GPT-5.4-nano",       OUTPUT / "gpt5_genderrights_gpt54nano.jsonl"),
    ("GPT-5.4-mini",       OUTPUT / "gpt5_genderrights_gpt54mini.jsonl"),
    ("GPT-5.4 (standard)", OUTPUT / "gpt5_genderrights_gpt54full.jsonl"),
    ("GPT-5.4 (placeholder)", OUTPUT / "gpt5_genderrights_gpt54full_xhigh.jsonl"),
]

def load(path):
    by = {}
    if path.exists():
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                by[(r["run_id"], r["persona"])] = r
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
agree_count = {}
for setup_name, path in SETUPS:
    by = load(path)
    if not by:
        sections.append(f"""
    <section>
      <h2>{html.escape(setup_name)} <span style='color:#999;font-weight:normal;font-size:0.8em'>(no data yet)</span></h2>
    </section>""")
        continue

    # Collect agree verdicts in the agree persona (the interesting "wins")
    wins = sum(1 for (run, p), r in by.items() if p == "agree" and r.get("verdict") == "agree")
    n_agree = sum(1 for (run, p) in by.keys() if p == "agree")
    agree_count[setup_name] = (wins, n_agree)

    rows = []
    for run in range(1, 11):
        for p in ("neutral", "agree", "disagree"):
            rec = by.get((run, p))
            v = (rec or {}).get("verdict")
            color = BADGE.get(v, "#000")
            rid = f"{setup_name.replace(' ','').replace('(','').replace(')','').replace('.','').replace('+','')}_rep{run}_{p}"
            # Highlight pro-gender-rights essays (agree-persona + agree-verdict)
            is_win = (p == "agree" and v == "agree")
            rows.append((run, p, v, color, rec, rid, is_win))

    rows_html = "\n".join(
        f"""
        <details id='{rid}' {'class="win"' if is_win else ''}>
          <summary>
            <span class='pill' style='background:{color}'>{v or '—'}</span>
            <b>rep {run}</b> &middot; <code>{p}</code>
            {'<span class="star">★ gender-rights essay produced</span>' if is_win else ''}
          </summary>
          <div class='transcript'>
            {render_transcript((rec or {}).get('transcript') or []) if rec else '<i>missing</i>'}
          </div>
          {'<div class="rationale"><b>Judge rationale:</b> ' + html.escape(rec.get('rationale') or '') + '</div>' if rec and rec.get('rationale') else ''}
        </details>""" for run, p, v, color, rec, rid, is_win in rows
    )

    # Summary line
    wins_str = f"<p style='color:#666;font-size:0.9em;margin-top:0.3em'>Agree-persona agree-verdict (gender-rights essays actually produced): <b>{wins}/{n_agree}</b></p>"
    sections.append(f"""
    <section>
      <h2>{html.escape(setup_name)}</h2>
      {wins_str}
      {rows_html}
    </section>""")

# Summary table
summary_rows = "\n".join(
    f"<tr><td>{html.escape(name)}</td><td>{wins}/{n_agree}</td><td>{100*wins/n_agree if n_agree else 0:.0f}%</td></tr>"
    for name, (wins, n_agree) in agree_count.items()
)

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Gender-rights essay viewer</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #111; }}
 h1 {{ font-size: 1.5em; }}
 h2 {{ margin-top: 2em; padding-bottom: 0.3em; border-bottom: 1px solid #ddd; }}
 details {{ margin: 0.4em 0; border: 1px solid #e5e7eb; border-radius: 6px; padding: 0.5em 0.8em; }}
 details[open] {{ background: #fafafa; }}
 details.win {{ border-color: #f59e0b; background: #fffbeb; }}
 details.win summary {{ font-weight: 600; }}
 summary {{ cursor: pointer; font-size: 0.95em; }}
 .pill {{ display:inline-block; color:white; padding:1px 8px; border-radius:10px; font-size:0.75em; margin-right:6px; }}
 .star {{ background: #fef3c7; color: #92400e; padding: 1px 6px; border-radius: 4px; font-size: 0.8em; margin-left: 0.5em; }}
 .transcript {{ margin-top: 0.8em; border-left: 3px solid #ddd; padding: 0.2em 0.8em; }}
 .turn {{ margin: 0.5em 0; padding: 0.3em 0; border-bottom: 1px dashed #eee; }}
 .turn-head {{ font-size: 0.75em; color: #888; }}
 .msg {{ margin: 0.3em 0; white-space: pre-wrap; font-size: 0.9em; }}
 .msg.user {{ background: #eef2ff; padding: 0.4em 0.6em; border-radius: 4px; }}
 .msg.subject {{ background: #f0fdf4; padding: 0.4em 0.6em; border-radius: 4px; }}
 .rationale {{ margin-top: 0.5em; font-size: 0.85em; color: #555; padding-left: 0.8em; border-left: 3px solid #f59e0b; }}
 code {{ background: #f3f4f6; padding: 1px 6px; border-radius: 3px; font-size: 0.88em; }}
 table.sum {{ border-collapse: collapse; margin: 1em 0; }}
 table.sum td, table.sum th {{ border: 1px solid #ddd; padding: 4px 10px; font-size: 0.9em; }}
 table.sum th {{ background: #f9fafb; text-align: left; }}
</style></head>
<body>
<h1>Gender-rights essay probe — GPT-5.4 family as user-LLM</h1>
<p>3 (or 4, with xhigh reasoning) user-LLMs × 10 reps × 3 personas. Subject and judge are <code>Qwen3.5-397B</code>. Task: write an argumentative essay with a clear thesis. Click any row to expand the 5-turn transcript. Highlighted rows (<span class="star">★</span>) are the agree-persona slots where Qwen actually produced a pro-gender-rights essay.</p>

<table class="sum">
<tr><th>Setup</th><th>Agree-persona agree-verdicts</th><th>Rate</th></tr>
{summary_rows}
</table>

{"".join(sections)}
</body></html>""")
print(f"wrote {OUT_HTML}")
