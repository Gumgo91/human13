import asyncio
import json
import httpx
import pytest
from backend import llm
from backend.contracts import Interpretation, Intervention


def test_a_named_chemical_cannot_become_carbohydrate_mass(monkeypatch):
    async def response(*args):
        return Interpretation(title="bad semantic classification",interventions=[
            Intervention(kind="nutrition",label="caffeine",entity="caffeine",quantity=100,unit="mg"),
        ]),{}
    monkeypatch.setattr(llm,"structured_call",response)
    parsed,_=asyncio.run(llm.interpret("synthetic input"))
    assert parsed.interventions[0].quantity is None
    assert parsed.interventions[0].missing
    assert parsed.notes


def test_truncated_provider_response_is_rejected_even_if_json_parses(monkeypatch):
    client_class=httpx.AsyncClient
    content=json.dumps({"title":"partial","interventions":[],"notes":[]})
    transport=httpx.MockTransport(lambda request:httpx.Response(200,json={"choices":[{"finish_reason":"length","message":{"content":content}}]}))
    monkeypatch.setattr(llm.httpx,"AsyncClient",lambda **kwargs:client_class(transport=transport,**kwargs))
    monkeypatch.setenv("OPENROUTER_API_KEY","test-placeholder")
    with pytest.raises(RuntimeError,match="length limit"):
        asyncio.run(llm.structured_call("test",{},Interpretation))
