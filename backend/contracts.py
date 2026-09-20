from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from .equation_contracts import EquationModel


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Intervention(StrictModel):
    kind: Literal["chemical", "nutrition", "activity", "environment", "hydration", "other"] = Field(description="Routing hint only: chemical, nutrition, activity, environment, hydration, or other general physiological events. Every category can create new equation models")
    label: str = Field(min_length=1, max_length=160)
    entity: str | None = Field(default=None, max_length=160)
    quantity: float | None = Field(default=None, ge=0, le=1e7)
    unit: str | None = Field(default=None, max_length=30)
    start_min: float = Field(default=0, ge=0, le=1440)
    duration_min: float | None = Field(default=None, gt=0, le=1440)
    route: Literal["oral", "iv", "inhaled", "other", "unspecified"] = "unspecified"
    intensity_met: float | None = Field(default=None, ge=1, le=20)
    missing: list[str] = Field(default_factory=list, max_length=12)
    # Initial body-network anchor channels decided by the LLM. None means not yet
    # interpreted; no keyword fallback exists. [] means the LLM decided there is
    # no applicable anchor, and it is never resurrected by a fallback.
    body_channels: list[str] | None = Field(default=None, max_length=10)
    # Outcome nodes (body-module ids) the user claims to have observed. As with
    # initial anchors the LLM selects them; the network backtraces the path. None means no claimed outcome.
    outcome_nodes: list[str] | None = Field(default=None, max_length=6)


class Interpretation(StrictModel):
    title: str = Field(max_length=160)
    interventions: list[Intervention] = Field(max_length=8)
    notes: list[str] = Field(default_factory=list, max_length=12)


class ExecutionProposal(StrictModel):
    mode: Literal["reuse", "surrogate", "unresolved"]
    module_id: str | None = Field(default=None, max_length=80)
    driver: Literal["event_exposure", "glucose_deviation", "insulin_deviation", "intestinal_uptake", "activity_load"] = "event_exposure"
    effect_port: str | None = Field(default=None, max_length=80)
    gain: float = Field(default=0, ge=-0.8, le=0.8)
    gain_low: float = Field(default=-0.2, ge=-0.8, le=0.8)
    gain_high: float = Field(default=0.2, ge=-0.8, le=0.8)
    tau_min: float = Field(default=15, ge=0.5, le=240)
    rationale: str = Field(default="", max_length=400)


class Hypothesis(StrictModel):
    event_index: int = Field(ge=0, le=7)
    organ: str = Field(max_length=80)
    cell_type: str = Field(max_length=100)
    target: str = Field(max_length=100)
    mechanism: str = Field(max_length=600)
    direction: Literal["increase", "decrease", "mixed", "unknown"]
    source_ids: list[str] = Field(default_factory=list, max_length=8)
    limitation: str = Field(max_length=300)
    execution: ExecutionProposal | None = None


class EvidenceSource(StrictModel):
    id: str = Field(max_length=120)
    title: str = Field(max_length=500)
    url: str = Field(max_length=600)


class Expansion(StrictModel):
    summary: str = Field(max_length=1000)
    hypotheses: list[Hypothesis] = Field(max_length=128)
    model: EquationModel | None = None
    sources: list[EvidenceSource] = Field(default_factory=list, max_length=24)


class RunSettings(StrictModel):
    horizon_min: float = Field(default=240, ge=10, le=1440)
    body_mass_kg: float = Field(default=70, ge=30, le=200)
    activity_met: float = Field(default=6, ge=1, le=16)
    allow_assumptions: bool = True
    gastric_half_min: float = Field(default=20, ge=1, le=300)
    absorption_half_min: float = Field(default=25, ge=1, le=300)
    elimination_half_min: float = Field(default=180, ge=1, le=2880)
    enable_llm_coupling: bool = True
    enable_body_network: bool = True
    llm_scale: float = Field(default=1, ge=0, le=2)
    sglt1_scale: float = Field(default=1, ge=0, le=3)
    glut2_scale: float = Field(default=1, ge=0, le=3)
    glut5_scale: float = Field(default=1, ge=0, le=3)
    sucrase_scale: float = Field(default=1, ge=0, le=3)
    glut4_scale: float = Field(default=1, ge=0, le=3)
    insulin_secretion_scale: float = Field(default=1, ge=0, le=3)
    model_parameters: dict[str, float] = Field(default_factory=dict, max_length=96)


class SimulationRequest(StrictModel):
    language: Literal["en"] | None = None
    text: str = Field(min_length=1, max_length=2000)
    settings: RunSettings = Field(default_factory=RunSettings)
    interpretation: Interpretation | None = None
    replay_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{16}$")
    rebuild_model: bool = False


class StateSpec(StrictModel):
    key: str
    label: str
    unit: str
    initial: float
    owner: str
    nonnegative: bool = True


class Parameter(StrictModel):
    name: str
    value: float
    unit: str
    source: str
    status: Literal["source_model", "assumption", "user_input", "derived"]
