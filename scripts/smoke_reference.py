"""Exercise the full Korean-input -> English reference-scenario API path."""
import json
import re
from pathlib import Path
import httpx


def main():
    result=None
    with httpx.Client(timeout=300) as client:
        with client.stream("POST", "http://127.0.0.1:8000/api/simulate", json={
            "text":"방금 사탕 하나 먹었어.", "language":"en"}) as response:
            response.raise_for_status()
            kind=None
            for line in response.iter_lines():
                if line.startswith("event: "):kind=line[7:]
                if not line.startswith("data: "):continue
                data=json.loads(line[6:])
                if kind=="progress":print(f"Stage {data['stage']}: {data['fraction']}",flush=True)
                elif kind=="error":raise RuntimeError(data["message"])
                elif kind=="preview":
                    assert data["reference_scenarios"] and data["display_language"]=="en"
                    print("English preview with a conditional nutrient reference is ready",flush=True)
                elif kind=="result":result=data
    assert result and result["reference_scenarios"]
    assert result["interpretation"]["interventions"][0]["quantity"] is None
    assert result["event_coverage"][0]["native"]
    for key in ("enzyme:sucrase", "flux:sglt1", "flux:glut5", "flux:glut2"):
        assert max(result["traces"][key])>0, key
    for node in result["nodes"]:
        for text in [node["label"],node["group"],node["description"],*node["cell_types"],*node["limitations"],*node["equations"]]:
            assert not re.search(r"[가-힣]",text), (node["id"],text)
    assert not result.get("presentation_warning")
    path=Path(".runtime/candy-smoke.json")
    path.write_text(json.dumps(result,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"run_id":result["run_id"],"nodes":len(result["nodes"]),"generation":result["generation"],
                      "glucose_peak_delta":max(result["traces"]["glucose"])-result["traces"]["glucose"][0],
                      "all_balances_passed":all(b["passed"] for b in result["validation"]["balances"])},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
