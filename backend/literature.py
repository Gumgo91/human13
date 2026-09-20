"""Retrieve traceable primary-paper metadata/abstracts, never model coefficients.

Search results are evidence candidates. They do not confer quantitative model
validation and are passed to the planner as untrusted reference data.
"""
import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
import httpx

CACHE=Path(__file__).parent/"data"/"literature"


async def research_events(interpretation):
    async def search(index,event):
        term=(event.entity or event.label).strip()[:120]
        # Keep the API query a literal phrase, not user-controlled query syntax.
        term=re.sub(r'[^\w\s-]'," ",term).strip()
        if not term:return []
        key=hashlib.sha256(term.casefold().encode()).hexdigest()[:24];path=CACHE/(key+".json")
        records=None
        if path.exists() and time.time()-path.stat().st_mtime<7*86400:
            try:records=json.loads(path.read_text(encoding="utf-8"))
            except (ValueError,OSError):pass
        if records is None:
            try:
                async with httpx.AsyncClient(timeout=12) as client:
                    response=await client.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",params={"query":f'TITLE_ABS:"{term}" AND HAS_ABSTRACT:Y NOT PUB_TYPE:review',"format":"json","resultType":"core","pageSize":3})
                    response.raise_for_status();payload=response.json()
                records=[]
                for r in payload.get("resultList",{}).get("result",[]):
                    if r.get("source")!="MED" or not str(r.get("id","")).isdigit():continue
                    records.append({"id":"pmid:"+r["id"],"title":r.get("title","")[:500],"url":"https://europepmc.org/article/MED/"+r["id"],"year":r.get("pubYear"),"abstract":re.sub("<[^>]*>"," ",r.get("abstractText",""))[:4500],"query":term,"retrieved_at":time.time()})
                CACHE.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(records,ensure_ascii=False),encoding="utf-8")
            except (httpx.HTTPError,ValueError,OSError):records=[]
        return [{**r,"event_index":index} for r in records]
    batches=await asyncio.gather(*(search(i,e) for i,e in enumerate(interpretation.interventions[:8])))
    return [r for batch in batches for r in batch]
