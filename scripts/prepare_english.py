"""Seed reusable English presentation strings; no scientific state is edited."""
import asyncio
import json
from pathlib import Path
from backend.contracts import Interpretation, Intervention, RunSettings
from backend.simulation import simulate
from backend.localization import display_slots, translate_strings, STATIC, _cache


async def main():
    strings=[]
    for run_id in ("be99a22b0aa04e8b", "bc1c94452d874673"):
        path=Path("backend/data/runs")/(run_id+".json")
        if path.exists():
            run=json.loads(path.read_text(encoding="utf-8"))["result"]
            strings.extend(c[k] for c,k in display_slots(run))
    for species in ("sucrose", "lactose", "maltose", "starch", "glucose", "fructose", "galactose", "carbohydrate"):
        run=simulate(Interpretation(title="Nutrient ingestion",interventions=[Intervention(kind="nutrition",label="Nutrient ingestion",entity=species,quantity=10,unit="g",route="oral")]),RunSettings(),{})
        strings.extend(c[k] for c,k in display_slots(run))
    strings=list(dict.fromkeys(strings))
    print(f"Preparing {len(strings)} unique presentation strings",flush=True)
    for i in range(0,len(strings),48):
        await translate_strings(strings[i:i+48])
        print(f"Ready: {min(i+48,len(strings))}/{len(strings)}",flush=True)
    STATIC.parent.mkdir(parents=True,exist_ok=True)
    STATIC.write_text(json.dumps(_cache,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved {len(_cache)} English strings",flush=True)


if __name__=="__main__":asyncio.run(main())
