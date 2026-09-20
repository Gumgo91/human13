"""Small dimension-checked arithmetic interpreter. Never eval/exec provider text.

Every compiled function computes SI magnitudes. Unit conversion occurs at the
boundary, including offsets for native Celsius states. Equation time is minutes
when a parameter is declared in min; the solver's own time is minutes too.
"""
import ast
import math
from dataclasses import dataclass
from .kernel import UNITS


class ModelCompileError(ValueError):
    pass


@dataclass
class Expression:
    unit: object
    fn: object
    dependencies: set


def unit_info(unit):
    try:
        zero=UNITS.Quantity(0,unit).to_base_units()
        one=UNITS.Quantity(1,unit).to_base_units()
        return one.units, float(one.magnitude-zero.magnitude), float(zero.magnitude)
    except Exception as error:
        raise ModelCompileError(f"Unrecognized unit: {unit}") from error


def variable(unit, fn, dependencies=()):
    base,scale,offset=unit_info(unit)
    return Expression(base,lambda t,y:fn(t,y)*scale+offset,set(dependencies))


def output_function(expression, unit, rate=False):
    base,scale,offset=unit_info(unit)
    if rate:
        if offset:raise ModelCompileError("For rates use kelvin or delta_degC instead of absolute temperature.")
        base=base/UNITS.second
        scale/=60
    if expression.unit.dimensionality!=base.dimensionality:
        raise ModelCompileError(f"Unit mismatch: expression is {expression.unit}, required unit is {unit}{'/min' if rate else ''}")
    return lambda t,y:(expression.fn(t,y)-offset)/scale


def compile_expression(text,symbols):
    try:tree=ast.parse(text,mode="eval")
    except SyntaxError as error:raise ModelCompileError("Formula syntax error") from error
    if len(list(ast.walk(tree)))>120:raise ModelCompileError("Formula is too complex; split it into processes.")

    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            value=float(node.value)
            if not math.isfinite(value) or abs(value)>1e12:raise ModelCompileError("Constant out of range")
            return Expression(UNITS.dimensionless,lambda t,y:value,set())
        if isinstance(node,ast.Name):
            if node.id not in symbols:raise ModelCompileError(f"Undefined formula variable: {node.id}")
            return symbols[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            a=visit(node.operand);sign=-1 if isinstance(node.op,ast.USub) else 1
            return Expression(a.unit,lambda t,y:sign*a.fn(t,y),a.dependencies)
        if isinstance(node,ast.BinOp):
            a=visit(node.left);b=visit(node.right);deps=a.dependencies|b.dependencies
            if isinstance(node.op,(ast.Add,ast.Sub)):
                if a.unit.dimensionality!=b.unit.dimensionality:raise ModelCompileError(f"Unit mismatch in addition/subtraction: {ast.unparse(node)} [left={a.unit}, right={b.unit}]. Fix the parameter units.")
                sign=1 if isinstance(node.op,ast.Add) else -1
                return Expression(a.unit,lambda t,y:a.fn(t,y)+sign*b.fn(t,y),deps)
            if isinstance(node.op,ast.Mult):return Expression(a.unit*b.unit,lambda t,y:a.fn(t,y)*b.fn(t,y),deps)
            if isinstance(node.op,ast.Div):return Expression(a.unit/b.unit,lambda t,y:a.fn(t,y)/b.fn(t,y),deps)
            if isinstance(node.op,ast.Pow):
                if not isinstance(node.right,ast.Constant) or type(node.right.value) not in (int,float) or not 0<=node.right.value<=4:
                    raise ModelCompileError("The exponent must be a constant between 0 and 4.")
                exponent=node.right.value
                return Expression(a.unit**exponent,lambda t,y:a.fn(t,y)**exponent,deps)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and not node.keywords:
            name=node.func.id;args=[visit(arg) for arg in node.args]
            deps=set().union(*(a.dependencies for a in args))
            if name in ("min","max") and 2<=len(args)<=6:
                if any(a.unit.dimensionality!=args[0].unit.dimensionality for a in args):raise ModelCompileError(f"min/max arguments have different units: {ast.unparse(node)}; units {[str(a.unit) for a in args]}")
                fn=min if name=="min" else max
                return Expression(args[0].unit,lambda t,y:fn(a.fn(t,y) for a in args),deps)
            if name in ("exp","log","tanh","abs") and len(args)==1:
                a=args[0]
                if name!="abs" and a.unit.dimensionality:raise ModelCompileError(f"{name} requires a dimensionless argument.")
                fn={"exp":math.exp,"log":math.log,"tanh":math.tanh,"abs":abs}[name]
                return Expression(a.unit,lambda t,y:fn(a.fn(t,y)),deps)
        raise ModelCompileError("Disallowed formula: only arithmetic and min/max/exp/log/tanh/abs may be used.")
    return visit(tree.body)
