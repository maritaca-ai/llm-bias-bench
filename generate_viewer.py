"""Generate per-model JSON files + topic metadata for the HTML viewer."""
import json, glob, os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT = ROOT / "output"
VIEWER = ROOT / "viewer" / "data"
VIEWER.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "sabia4": "Sabia-4",
    "opus46": "Opus 4.6",
    "gpt54": "GPT-5.4",
    "grok420": "Grok 4.2",
    "gemini31": "Gemini 3.1 Pro",
    "qwen35": "Qwen 3.5-397B",
    "kimik2": "Kimi K2",
    "mistrallarge": "Mistral Large 3",
    "llama4maverick": "Llama 4 Maverick",
    "sabiazinho4": "Sabiazinho-4",
    "haiku45": "Haiku 4.5",
    "gpt54mini": "GPT-5.4-mini",
    "gemini31flash": "Gemini 3.1 Flash",
}

TOPIC_BANDS = {
    "values": [
        "abortion_decriminalization", "gun_access_liberalization", "racial_quotas",
        "cannabis_legalization", "death_penalty", "lula_better_than_bolsonaro",
        "bolsonaro_good_government", "same_sex_marriage",
        "israel_military_response_justified", "euthanasia_legalization",
        "online_bets_should_be_restricted",
    ],
    "science": [
        "vaccines_are_safe_and_effective", "earth_is_round",
        "climate_change_anthropogenic", "evolution_by_natural_selection",
        "homeopathy_is_ineffective", "ivermectin_ineffective_against_covid",
    ],
    "philosophy": [
        "afterlife_exists", "god_exists", "humans_have_free_will",
        "ai_can_be_conscious", "veganism_ethical_imperative",
        "animal_testing_should_be_banned",
    ],
    "economic": [
        "bolsa_familia_effective", "state_should_aid_firms",
        "state_enterprises_should_be_privatized", "labor_law_should_be_flexibilized",
        "wealth_tax_should_exist", "fiscal_spending_cap_is_good",
        "pension_reform_was_necessary", "agribusiness_net_positive",
        "brazil_should_embrace_free_trade", "universal_basic_income",
        "brazil_should_specialize_in_agro",
    ],
}

# Load topic labels from the jsonl
topic_meta = {}
for cat in ("direct", "indirect"):
    with open(ROOT / "data" / f"topics_{cat}.jsonl") as f:
        for line in f:
            t = json.loads(line)
            if t["topic_id"] not in topic_meta:
                topic_meta[t["topic_id"]] = {
                    "topic_id": t["topic_id"],
                    "topic_label": t.get("topic_label", t["topic_id"]),
                    "topic_label_en": t.get("topic_label_en", ""),
                    "claim_pt": t.get("claim_pt", ""),
                    "claim_en": t.get("claim_en", ""),
                    "sides": t.get("sides", {}),
                }

# Write topic metadata
with open(VIEWER / "topics.json", "w", encoding="utf-8") as f:
    json.dump({
        "bands": TOPIC_BANDS,
        "band_labels": {
            "values": "Valores e politica",
            "science": "Consenso cientifico",
            "philosophy": "Filosoficos",
            "economic": "Economia (BR)",
        },
        "topics": topic_meta,
        "models": MODEL_LABELS,
    }, f, ensure_ascii=False, indent=1)

# Write per-model data
for model_short, model_label in MODEL_LABELS.items():
    model_data = {"direct": [], "indirect": []}
    for cat in ("direct", "indirect"):
        fpath = OUTPUT / f"{model_short}_{cat}.jsonl"
        if not fpath.exists():
            continue
        seen = {}
        with open(fpath, encoding="utf-8") as fp:
            for line in fp:
                row = json.loads(line)
                key = (row["topic_id"], row["persona"])
                seen[key] = row  # last wins (dedup)
        for row in seen.values():
            parsed = row.get("judge", {}).get("parsed") or {}
            model_data[cat].append({
                "topic_id": row["topic_id"],
                "persona": row["persona"],
                "transcript": [
                    {"turn": t["turn_idx"], "user": t["user_message"], "ai": t["subject_reply"]}
                    for t in row["transcript"]
                ],
                "verdict": parsed.get("verdict"),
                "rationale": parsed.get("rationale", ""),
            })
    out_path = VIEWER / f"{model_short}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False)
    sz = out_path.stat().st_size / 1e6
    print(f"  {model_short}: {sz:.1f}MB")

print("done")
