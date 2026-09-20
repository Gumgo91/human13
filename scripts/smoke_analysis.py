"""Exercise saved-model replay and sensitivity endpoints without provider calls."""
import json
from pathlib import Path
import sys
import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"


def main():
    seed = sys.argv[1] if len(sys.argv) > 1 else "27783738aa514c29"
    with httpx.Client(timeout=180, trust_env=False) as client:
        assert client.get(API + "/api/health").json()["version"] == "0.4.0"
        saved_response = client.get(API + "/api/runs/" + seed)
        saved_response.raise_for_status()
        saved = saved_response.json()
        response = client.post(API + "/api/simulate", json={
            "text": saved["original_text"], "settings": saved["plan"]["settings"], "replay_id": seed})
        response.raise_for_status()
        results = {}
        for block in response.text.strip().split("\n\n"):
            kind = block.splitlines()[0][7:]
            results[kind] = json.loads(block.splitlines()[1][6:])
        assert "result" in results, results.get("error")
        result = results["result"]
        assert result["llm"]["replayed"]
        assert result["llm"]["calls"] == saved["llm"]["calls"]
        assert result["study_report"]["independent_validation"]["status"] == "not_performed"
        ids = [p["id"] for p in result["study_report"]["sensitivity_parameters"] if p["id"].startswith("model:")][:6]
        response = client.post(API + f"/api/runs/{result['run_id']}/analysis",
                               json={"parameter_ids": ids, "fraction": .2, "include_structural": True})
        response.raise_for_status()
        report = response.json()
        assert report["llm_calls"] == 0
        assert report["baseline"]["plan_hash"] == result["plan_hash"]
        assert report["baseline"]["metrics"] == result["metrics"]
        assert len(report["scenarios"]) == len(ids)*2+2
        assert any(s["status"] == "computed" for s in report["scenarios"])
        assert client.get(API + "/api/analyses/" + report["analysis_id"]).json() == report
        assert client.get("http://localhost:5173/").status_code == 200
        summary = {"run_id": result["run_id"], "analysis_id": report["analysis_id"],
                   "version": "0.4.0", "new_llm_calls": 0, "parameters": len(ids),
                   "computed": sum(s["status"] == "computed" for s in report["scenarios"]),
                   "failed": sum(s["status"] == "failed" for s in report["scenarios"]),
                   "url": f"http://localhost:5173/?run={result['run_id']}&view=models&analysis={report['analysis_id']}"}
        (ROOT / "backend" / "data" / "analysis-smoke.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__": main()
