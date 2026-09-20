"""Local, content-addressed model plans; reuse only exact interpreted inputs.

This deliberately does not generalize fitted/assumed gains between doses or
subjects. Replaying a run holds its plan fixed while changing parameters.
"""
import hashlib
import json
from pathlib import Path
from .contracts import Expansion

LIBRARY=Path(__file__).parent/"data"/"model_library"


def request_key(interpretation,engine_hash,model,conditions=None):
    return hashlib.sha256(json.dumps({"events":interpretation.model_dump(),"engine":engine_hash,"model":model,"conditions":conditions},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def get_plan(key):
    path=LIBRARY/(key+".json")
    if not path.exists():return None
    try:
        data=json.loads(path.read_text(encoding="utf-8"));return Expansion.model_validate(data["expansion"]),data
    except (ValueError,KeyError,OSError):return None


def save_plan(key,expansion,result):
    LIBRARY.mkdir(parents=True,exist_ok=True)
    data={"key":key,"title":result["title"],"created_at":result["created_at"],"engine_hash":result["plan"]["engine_hash"],"expansion":expansion.model_dump(),"model_hash":result["generated"]["hash"],"states":result["generated"]["states"],"processes":result["generated"]["processes"],"checks":result["generated"]["checks"],"event_coverage":result["event_coverage"]}
    path=LIBRARY/(key+".json");temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8");temp.replace(path)


def list_plans():
    rows=[]
    for path in sorted(LIBRARY.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)[:100]:
        try:
            data=json.loads(path.read_text(encoding="utf-8"));rows.append({k:data[k] for k in ("key","title","created_at","states","processes","model_hash","checks")})
        except (ValueError,OSError,KeyError):continue
    return rows
