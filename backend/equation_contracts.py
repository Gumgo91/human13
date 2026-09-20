"""Declarative extensions: new biology is model data, not a Python module."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class ModelObject(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ModelState(ModelObject):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    label: str = Field(max_length=100)
    entity: str = Field(max_length=100, description="One biological quantity; reuse this state across all events affecting it")
    compartment: str = Field(max_length=100)
    level: Literal["molecule", "cell", "tissue", "organ", "system"]
    cell_types: list[str] = Field(default_factory=list,max_length=8,description="Actual involved cell types, not names of signals or state variables; empty if unspecified")
    unit: str = Field(max_length=60, description="Pint unit. Use dimensionless for uncalibrated relative activity, not invented concentration")
    initial: float = Field(ge=-1e9, le=1e9)
    lower: float = Field(ge=-1e9, le=1e9)
    upper: float = Field(ge=-1e9, le=1e9)
    description: str = Field(max_length=400)


class ModelParameter(ModelObject):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    label: str = Field(max_length=100)
    value: float = Field(ge=-1e9, le=1e9)
    unit: str = Field(max_length=60)
    low: float = Field(ge=-1e9, le=1e9)
    high: float = Field(ge=-1e9, le=1e9)
    rationale: str = Field(max_length=300, description="Measured evidence vs exploratory prior; ranges are NOT confidence intervals")


class ModelBinding(ModelObject):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    source: str = Field(max_length=100, description="Exact native state/flux key from available_model. New states and event_N are already in scope")


class Stoichiometry(ModelObject):
    state: str = Field(max_length=48)
    coefficient: float = Field(ge=-100, le=100)


class ModelProcess(ModelObject):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    label: str = Field(max_length=100)
    event_indices: list[int] = Field(min_length=1, max_length=8)
    rate: str = Field(min_length=1, max_length=500, description="Arithmetic expression with names, + - * / **, min/max/exp/log/tanh/abs. Parameters carry units. Example (gain*event_0-x)/tau. No code")
    changes: list[Stoichiometry] = Field(min_length=1, max_length=12, description="d(state)/dt += coefficient*rate; negative for consumed, positive for produced")
    description: str = Field(max_length=500)
    source_ids: list[str] = Field(default_factory=list, max_length=8)


class ModelObservation(ModelObject):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    label: str = Field(max_length=100)
    expression: str = Field(min_length=1, max_length=500)
    unit: str = Field(max_length=60)
    description: str = Field(max_length=400)


class ModelCoupling(ModelObject):
    port: str = Field(max_length=80)
    expression: str = Field(min_length=1, max_length=500, description="Dimensionless additive shift for additive_target ports, log multiplier otherwise. Match the biological port label; use only new missing mechanisms")
    description: str = Field(max_length=400)
    signal_direction: Literal["increase", "decrease", "mixed"] | None = Field(default=None,description="Sign of this contribution over the simulated trajectory: increase is nonnegative, decrease nonpositive, mixed may change sign. Check coefficient and expression together")


class ModelImpulse(ModelObject):
    event_index: int = Field(ge=0, le=7)
    state: str = Field(max_length=48)
    amount: str = Field(max_length=200, description="Constant expression of parameters or dose_N with matching amount units")


class ModelInvariant(ModelObject):
    label: str = Field(max_length=100)
    weights: list[Stoichiometry] = Field(min_length=2, max_length=32)


class EquationModel(ModelObject):
    states: list[ModelState] = Field(default_factory=list, max_length=32)
    parameters: list[ModelParameter] = Field(default_factory=list, max_length=96)
    bindings: list[ModelBinding] = Field(default_factory=list, max_length=32)
    processes: list[ModelProcess] = Field(default_factory=list, max_length=48)
    observations: list[ModelObservation] = Field(default_factory=list, max_length=16)
    couplings: list[ModelCoupling] = Field(default_factory=list, max_length=16)
    impulses: list[ModelImpulse] = Field(default_factory=list, max_length=16)
    invariants: list[ModelInvariant] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=12)


class ModelSynthesis(ModelObject):
    summary: str = Field(max_length=1000)
    status: Literal["ok","failed"] = Field(default="ok", description="'failed' only when you cannot construct the requested model at all")
    model: EquationModel | None
    reuse: list[str] = Field(default_factory=list,max_length=128,description="IDs of existing native modules that already encode this event, no duplicate dynamics")
