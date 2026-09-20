"""Narrow, audited CellML/MathML adapter for the pinned Topp model.

This reproduces the PMR FILE, not the paper. The file explicitly reports that
published results are not reproduced. No arbitrary Python or expressions run.
Unknown MathML constructs are rejected. Unit semantics of this one model are
documented separately; this is not a general CellML importer.
"""
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import libsbml
import roadrunner

ROOT = Path(__file__).resolve().parent
SOURCE_URL = "https://models.physiomeproject.org/workspace/topp_promislow_devries_miura_finegood_2000/rawfile/0b82d5ca266a6748c3fbab3018862b111484811d/topp_promislow_devries_miura_finegood_2000.cellml"
CELL = "{http://www.cellml.org/cellml/1.0#}"
MATH = "{http://www.w3.org/1998/Math/MathML}"
STATE_NAMES = ["G", "I", "beta"]


def evaluate(node, context):
    tag = node.tag.split("}")[-1]
    if tag == "ci":
        return context[node.text.strip()]
    if tag == "cn":
        return float(node.text)
    if tag != "apply":
        raise ValueError(f"Unsupported MathML: {tag}")
    op = node[0].tag.split("}")[-1]
    v = [evaluate(c, context) for c in list(node)[1:]]
    if op == "plus": return sum(v)
    if op == "times": return float(np.prod(v))
    if op == "minus": return -v[0] if len(v) == 1 else v[0] - v[1]
    if op == "divide": return v[0] / v[1]
    if op == "power": return v[0] ** v[1]
    raise ValueError(f"Unsupported MathML operation: {op}")


class ToppModel:
    def __init__(self):
        self.path = ROOT / "models" / "topp_2000.cellml"
        self.raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(self.raw).hexdigest()
        tree = ET.fromstring(self.raw)
        self.constants = {}
        self.units = {}
        self.rhs = {}
        for component in tree.findall(CELL + "component"):
            for var in component.findall(CELL + "variable"):
                if "initial_value" in var.attrib:
                    name = var.attrib["name"]
                    if name in self.constants:
                        raise ValueError(f"Duplicate initial variable: {name}")
                    self.constants[name] = float(var.attrib["initial_value"])
                    self.units[name] = var.attrib["units"]
            for math in component.findall(MATH + "math"):
                for equation in math:
                    if equation[0].tag != MATH + "eq" or equation[1][0].tag != MATH + "diff":
                        raise ValueError("Only explicit ODEs are supported in this pinned adapter")
                    name = equation[1][-1].text.strip()
                    self.rhs[name] = equation[2]
        if set(self.rhs) != set(STATE_NAMES):
            raise ValueError("Unexpected model state set")

    def derivative(self, _t, y):
        context = self.constants | dict(zip(STATE_NAMES, y))
        return np.array([evaluate(self.rhs[n], context) for n in STATE_NAMES])

    def equilibrium(self):
        p = self.constants
        G = (p["r1"] - np.sqrt(p["r1"]**2 - 4*p["r2"]*p["d0"]))/(2*p["r2"])
        I = (p["R0"]/G-p["EG0"])/p["SI"]
        beta = I*p["k"]*(p["alpha"]+G*G)/(p["sigma"]*G*G)
        return [float(G), float(I), float(beta)]

    def to_sbml(self):
        doc = libsbml.SBMLDocument(3, 2)
        model = doc.createModel(); model.setId("topp_2000_pmr")
        for name, value in self.constants.items():
            p = model.createParameter(); p.setId(name); p.setValue(value)
            p.setConstant(name not in self.rhs)
        for name, expr in self.rhs.items():
            rule = model.createRateRule(); rule.setVariable(name)
            math = f'<math xmlns="http://www.w3.org/1998/Math/MathML">{ET.tostring(expr, encoding="unicode")}</math>'
            ast = libsbml.readMathMLFromString(math)
            if ast is None: raise ValueError("Unable to convert source MathML")
            rule.setMath(ast)
        if doc.checkInternalConsistency() and doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR):
            raise ValueError(doc.getErrorLog().toString())
        return libsbml.writeSBMLToString(doc)

    def reproduce(self):
        times = np.linspace(0, 10, 401)
        initial = [self.constants[n] for n in STATE_NAMES]
        solution = solve_ivp(self.derivative, (0, 10), initial, t_eval=times, method="Radau", rtol=1e-10, atol=1e-11)
        if not solution.success: raise ValueError(solution.message)
        rr = roadrunner.RoadRunner(self.to_sbml())
        rr.integrator.relative_tolerance = 1e-10
        rr.integrator.absolute_tolerance = 1e-11
        output = np.asarray(rr.simulate(0, 10, 401, selections=["time"]+STATE_NAMES))
        max_error = float(np.max(np.abs(output[:, 1:]-solution.y.T)))
        # With the source beta=I=0, G has an exact analytical solution.
        p = self.constants
        exact = p["R0"]/p["EG0"]+(initial[0]-p["R0"]/p["EG0"])*np.exp(-p["EG0"]*times)
        exact_error = float(np.max(np.abs(solution.y[0]-exact)))
        # A second nondegenerate initial condition exercises every RHS.
        perturbed = np.array(self.equilibrium()) * [1.5, .8, 1.1]
        ss = solve_ivp(self.derivative, (0, 10), perturbed, t_eval=times, method="Radau", rtol=1e-10, atol=1e-11)
        if not ss.success: raise ValueError(ss.message)
        rr.resetAll()
        for name, value in zip(STATE_NAMES, perturbed): rr[name] = value
        oo = np.asarray(rr.simulate(0, 10, 401, selections=["time"]+STATE_NAMES))
        nondegenerate_error = float(np.max(np.abs(oo[:, 1:]-ss.y.T)))
        report = {
            "model": "Topp et al. 2000 · pinned PMR CellML",
            "source_url": SOURCE_URL, "source_sha256": self.sha256,
            "scope": "Reproduces the source CellML file numerics. Not paper figures or clinical validation.",
            "source_warning": "The PMR source is annotated as failing to reproduce the paper results.",
            "engines": [f"SciPy Radau", f"libRoadRunner {roadrunner.__version__} / CVODE"],
            "source_initial_conditions": dict(zip(STATE_NAMES, initial)),
            "max_absolute_difference": max_error,
            "analytical_glucose_max_error": exact_error,
            "perturbed_initial_max_difference": nondegenerate_error,
            "acceptance_absolute_tolerance": 1e-4,
            "passed": max(max_error, exact_error, nondegenerate_error) < 1e-4,
            "time_unit": "day", "time": times.tolist(),
            "traces": {n: solution.y[i].tolist() for i, n in enumerate(STATE_NAMES)},
            "equilibrium": dict(zip(STATE_NAMES, self.equilibrium())),
        }
        return report


if __name__ == "__main__":
    model = ToppModel(); report = model.reproduce()
    (ROOT / "models" / "topp_2000.sbml").write_text(model.to_sbml(), encoding="utf-8")
    (ROOT / "models" / "reproduction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("time", "traces")}, ensure_ascii=False, indent=2))
