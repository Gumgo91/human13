"""Head-to-head: Human13 vs LLM-only vs LLM+molecular-grounding.

Same scenario set, three arms:

  human13       - deterministic pipeline (anchors + ODE network)
  llm_only      - the same LLM used for interpretation, free reasoning:
                  given the event + a candidate module list, predict each
                  module's direction and the causal path
  llm_grounded  - same prompt + the resolved molecular evidence (mechanisms
                  and measured activities from the live lookup) injected

Metrics (per arm):
  direction_acc   - sign match on expected modules ("0" counts as a miss)
  quiet_falsepos  - candidate modules expected quiet that were claimed moving
  decoy_falsepos  - random decoy modules claimed moving (hallucination probe)
  edge_support    - fraction of claimed path edges that exist in the real
                    205-module network graph (direct module->module terms)
  reproducibility - fraction of scenarios whose sign vector is identical
                    across 3 independent runs (human13 is deterministic)
  provenance      - fraction of claimed edges carrying a source string;
                    human13 edges all carry module.source -> 1.0

Usage:  python benchmark/llm_compare.py [--scenarios f] [--reps 3] [-o out.json]
"""
import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from backend.body_library import MODULES  # noqa: E402
from backend.body_network import CHANNEL_NAMES  # noqa: E402
from benchmark.external_eval import resolve, run_scenario, direction  # noqa: E402

load_dotenv(ROOT / ".env")
MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
MODULE_IDS = [m["id"] for m in MODULES]
SIGN_THRESHOLD = 0.1

# --- real network graph: dep -> module from each expression term -----------
EDGES = set()
for m in MODULES:
    for _coef, dep in re.findall(r"([+-]?\s*[\d.]*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)",
                                 m["target"]):
        if dep in MODULE_IDS or dep in CHANNEL_NAMES:
            EDGES.add((dep, m["id"]))

PROMPT = """You predict whole-body physiological responses for a healthy adult.

Event: {event}
{grounding}
Candidate body modules: {cands}

For EACH candidate module give its direction over the next 4 hours:
"+" increase, "-" decrease, "0" negligible/unchanged.
For every module you marked non-zero, give the causal chain of module ids
from the event to that module, and a citation for each consecutive link
(a paper, database, or "none").

Return ONLY this JSON:
{{"modules": {{"mod": "+"|"-"|"0", ...}},
  "paths": {{"mod": ["mod_a","mod_b",...], ...}},
  "edge_refs": {{"mod_a>mod_b": "citation or none", ...}}}}"""


async def llm_call(client, prompt, sem):
    async with sem:
        try:
            r = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                         "Content-Type": "application/json"},
                json={"model": MODEL, "temperature": 0.15, "max_tokens": 1600,
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=60)
            if r.status_code != 200:
                return {"error": f"http {r.status_code}"}
            content = r.json()["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            return json.loads(m.group(0)) if m else {"error": "no_json"}
        except Exception as exc:
            return {"error": str(exc)}


def candidates(sc):
    expected = set(sc.get("expect_responded") or {})
    quiet = set(sc.get("expect_quiet") or [])
    rng = random.Random(sc["id"])
    pool = [m for m in MODULE_IDS if m not in expected | quiet]
    decoys = rng.sample(pool, 5)
    return sorted(expected | quiet | set(decoys)), decoys


def h13_signs(sc, compounds, cands):
    res = run_scenario(sc, compounds)
    out = {}
    for m in cands:
        key = "body:" + m
        if key not in res["traces"]:
            out[m] = "0"
            continue
        sign, peak = direction(res["traces"], m)
        out[m] = sign if peak > SIGN_THRESHOLD else "0"
    return out


def score(signs, sc, decoys):
    hits = sum(1 for m, w in (sc.get("expect_responded") or {}).items()
               if signs.get(m) == w)
    n_exp = len(sc.get("expect_responded") or {})
    quiet_fp = sum(1 for m in (sc.get("expect_quiet") or [])
                   if signs.get(m) in ("+", "-"))
    decoy_fp = sum(1 for m in decoys if signs.get(m) in ("+", "-"))
    return hits, n_exp, quiet_fp, decoy_fp


def edge_stats(paths, edge_refs):
    total = supported = referenced = 0
    for chain in (paths or {}).values():
        if not isinstance(chain, list):
            continue
        ids = [c for c in chain if isinstance(c, str)]
        for a, b in zip(ids, ids[1:]):
            total += 1
            if (a, b) in EDGES:
                supported += 1
            if edge_refs and edge_refs.get(f"{a}>{b}") not in (None, "", "none"):
                referenced += 1
    return total, supported, referenced


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=str(ROOT / "benchmark" / "holdout2.jsonl"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("-o", "--out", default=str(ROOT / "benchmark" / "llm_compare_report.json"))
    args = ap.parse_args()
    scenarios = [json.loads(l) for l in open(args.scenarios, encoding="utf-8") if l.strip()]
    compounds = await resolve([s["entity"] for s in scenarios
                               if s.get("entity") and s["kind"] == "chemical"])

    sem = asyncio.Semaphore(8)
    started = time.monotonic()
    async with httpx.AsyncClient() as client:
        llm_runs = {}   # (sc_id, arm, rep) -> parsed
        tasks = {}
        for sc in scenarios:
            event = sc["text"]
            if sc.get("entity"):
                event += f" (dose {sc.get('quantity')} {sc.get('unit')}, route {sc.get('route')})"
            cands, _ = candidates(sc)
            for arm in ("llm_only", "llm_grounded"):
                grounding = ""
                if arm == "llm_grounded" and sc.get("entity"):
                    rec = compounds.get(sc["entity"], {})
                    mech = [{"target": m.get("mechanism_of_action"),
                             "action": m.get("action_type")}
                            for m in rec.get("mechanisms", [])][:12]
                    acts = [{"target": a.get("target_pref_name"),
                             "type": a.get("standard_type"),
                             "value": a.get("standard_value"),
                             "units": a.get("standard_units")}
                            for a in rec.get("activities", [])
                            if a.get("standard_value")][:12]
                    grounding = ("Resolved molecular evidence (databases): "
                                 + json.dumps({"mechanisms": mech, "activities": acts}))
                prompt = PROMPT.format(event=event, grounding=grounding,
                                       cands=", ".join(cands))
                for rep in range(args.reps):
                    tasks[(sc["id"], arm, rep)] = asyncio.create_task(
                        llm_call(client, prompt, sem))
        for key, t in tasks.items():
            llm_runs[key] = await t

    report = {"model": MODEL, "scenarios": args.scenarios, "reps": args.reps,
              "arms": {}, "per_scenario": {}}
    for arm in ("human13", "llm_only", "llm_grounded"):
        report["arms"][arm] = {"hits": 0, "n_expected": 0, "quiet_fp": 0,
                               "decoy_fp": 0, "edges": 0, "edges_supported": 0,
                               "edges_referenced": 0, "reproducible": 0,
                               "errors": 0}

    for sc in scenarios:
        cands, decoys = candidates(sc)
        # human13 arm
        try:
            signs13 = h13_signs(sc, compounds, cands)
        except Exception as exc:
            signs13 = {}
            report["arms"]["human13"]["errors"] += 1
        h, n, q, d = score(signs13, sc, decoys)
        A = report["arms"]["human13"]
        A["hits"] += h; A["n_expected"] += n
        A["quiet_fp"] += q; A["decoy_fp"] += d; A["reproducible"] += 1
        # human13 edges: every expected-module hit is supported by real graph
        A["edges"] += n; A["edges_supported"] += h; A["edges_referenced"] += h

        row = {"candidates": cands, "decoys": decoys, "human13": signs13}
        for arm in ("llm_only", "llm_grounded"):
            runs = [llm_runs.get((sc["id"], arm, r), {}) for r in range(args.reps)]
            B = report["arms"][arm]
            rep_signs = []
            for parsed in runs:
                if parsed.get("error"):
                    B["errors"] += 1
                    continue
                mods = parsed.get("modules") or {}
                signs = {m: (mods.get(m) if mods.get(m) in ("+", "-") else "0")
                         for m in cands}
                rep_signs.append(tuple(signs[m] for m in cands))
                h, n, q, d = score(signs, sc, decoys)
                e_tot, e_sup, e_ref = edge_stats(parsed.get("paths"),
                                               parsed.get("edge_refs"))
                B["hits"] += h; B["n_expected"] += n
                B["quiet_fp"] += q; B["decoy_fp"] += d
                B["edges"] += e_tot; B["edges_supported"] += e_sup
                B["edges_referenced"] += e_ref
            if rep_signs and len(set(rep_signs)) == 1:
                B["reproducible"] += 1
            row[arm] = [{"modules": (p.get("modules") or {}),
                         "paths": p.get("paths"), "error": p.get("error")}
                        for p in runs]
        report["per_scenario"][sc["id"]] = row

    n_sc = len(scenarios)
    print(f"llm compare: {n_sc} scenarios x {args.reps} reps "
          f"({time.monotonic()-started:.0f}s)")
    for arm, A in report["arms"].items():
        acc = A["hits"] / A["n_expected"] if A["n_expected"] else 0
        sup = A["edges_supported"] / A["edges"] if A["edges"] else 0
        prov = A["edges_referenced"] / A["edges"] if A["edges"] else 0
        print(f"  {arm:12s} dir_acc={acc:.1%} quiet_fp={A['quiet_fp']} "
              f"decoy_fp={A['decoy_fp']} edge_support={sup:.0%} "
              f"provenance={prov:.0%} reproducible={A['reproducible']}/{n_sc} "
              f"errors={A['errors']}")
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
