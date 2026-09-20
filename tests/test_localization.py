import asyncio
import copy
from backend import localization


def test_english_presentation_leaves_scientific_record_and_input_unchanged(monkeypatch):
    async def translate(strings):
        return {s: {"Blood glucose":"Blood glucose", "Pancreatic beta cells":"Pancreatic beta cell", "기존 근거":"Existing evidence"}.get(s,s) for s in strings}
    monkeypatch.setattr(localization, "translate_strings", translate)
    run = {"original_text":"방금 사탕 하나 먹었어", "plan":{"equation":"Blood glucose", "parameter":3.2},
           "traces":{"glucose":[100.,101.5]}, "nodes":[{"id":"glucose", "state_key":"glucose", "label":"Blood glucose",
           "unit":"mg/dL", "cell_types":["Pancreatic beta cells"], "equations":["dG/dt = R0 - EG0*G"],
           "sources":[{"id":"ref", "title":"기존 근거", "url":"https://example.org/evidence"}],
           "intervention":{"label":"원래 Input", "quantity":None}}]}
    original=copy.deepcopy(run)
    out=asyncio.run(localization.localize_run(run))
    assert run==original
    assert out["nodes"][0]["label"]=="Blood glucose"
    assert out["nodes"][0]["cell_types"]==["Pancreatic beta cell"]
    for key in ("original_text", "plan", "traces"): assert out[key]==original[key]
    for key in ("id", "state_key", "equations", "intervention"): assert out["nodes"][0][key]==original["nodes"][0][key]


def test_translation_failure_is_explicit_and_does_not_drop_numeric_data(monkeypatch):
    async def fail(strings): raise RuntimeError("offline")
    monkeypatch.setattr(localization,"translate_strings",fail)
    monkeypatch.setattr(localization,"_cache",{})
    result=asyncio.run(localization.localize_run({"nodes":[{"label":"미번역", "state_key":"x"}],"traces":{"x":[1,2]}}))
    assert result["presentation_warning"]
    assert result["nodes"][0]["label"]=="English description unavailable"
    assert result["traces"]["x"]==[1,2]
