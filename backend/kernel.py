"""Typed state registry, conservative transfers and event-aware ODE integration."""
from collections.abc import Callable
import math
import numpy as np
from scipy.integrate import solve_ivp
import pint
from .contracts import StateSpec

UNITS = pint.UnitRegistry()


class ModularSystem:
    def __init__(self):
        self.states: list[StateSpec] = []
        self.indices: dict[str, int] = {}
        self.processes: list[tuple[str, Callable]] = []
        self.impulses: list[tuple[float, str, float]] = []
        self.updates: list[tuple[float, int, float, float, str]] = []
        self.nudges: dict[int, tuple[float, float]] = {}
        self.boundaries: set[float] = {0.0}
        self.observables: dict[str, tuple[str, Callable]] = {}

    def observe(self, key, unit, callback):
        if key in self.indices or key in self.observables: raise ValueError(f"Duplicate observation key: {key}")
        self.observables[key]=(unit,callback)
        return key

    def value(self,key,t,y):
        if key in self.indices:return float(y[self.indices[key]])
        if key in self.observables:return float(self.observables[key][1](t,y))
        raise KeyError(key)

    def observation_traces(self,times,values):
        result={key:[float(fn(t,values[:,i])) for i,t in enumerate(times)] for key,(_,fn) in self.observables.items()}
        if any(not np.all(np.isfinite(v)) for v in result.values()):raise ValueError("Non-finite process observation")
        return result

    def state(self, key, label, unit, initial, owner, nonnegative=True):
        if key in self.indices: raise ValueError(f"Duplicate state ownership: {key}")
        if not math.isfinite(initial): raise ValueError("Invalid initial value")
        self.indices[key] = len(self.states)
        self.states.append(StateSpec(key=key,label=label,unit=unit,initial=initial,owner=owner,nonnegative=nonnegative))
        return key

    def process(self, name, callback):
        if name in {p[0] for p in self.processes}: raise ValueError(f"Duplicate process: {name}")
        self.processes.append((name, callback))

    def transfer(self, name, source, target, rate_per_min):
        a=self.indices[source]; b=self.indices[target]
        # A transfer must preserve the same quantity, including a unit conversion.
        scale=(1*UNITS(self.states[a].unit)).to(self.states[b].unit).magnitude
        if not math.isfinite(rate_per_min) or rate_per_min<0: raise ValueError("Invalid transfer coefficient")
        def flow(t,y,dy):
            value=rate_per_min*y[a]
            dy[a]-=value;dy[b]+=value*scale
        self.process(name, flow)

    def impulse(self, time, key, amount):
        if key not in self.indices or amount<0 or not math.isfinite(amount): raise ValueError("Invalid event")
        self.impulses.append((time,key,amount));self.boundaries.add(float(time))

    def assimilate(self, time, key, target, gain):
        """Sequential state update: at `time`, move state toward `target` by
        `gain` (0-1). The observation operator lives outside the kernel; this
        is the assimilation step proper."""
        if key not in self.indices: raise ValueError("Invalid observation state: "+key)
        if not math.isfinite(target) or not 0<gain<=1: raise ValueError("Invalid observation update")
        self.updates.append((float(time),self.indices[key],float(target),float(gain),"set"))
        self.boundaries.add(float(time))

    def nudge(self, time, key, target, gain):
        """Continuous nudging: from `time` on, add gain*(target - x) to the
        state's derivative. The latest observation persists until the next
        update (streaming last-observation-hold)."""
        if key not in self.indices: raise ValueError("Invalid observation state: "+key)
        if not math.isfinite(target) or not 0<=gain: raise ValueError("Invalid observation update")
        self.updates.append((float(time),self.indices[key],float(target),float(gain),"nudge"))
        self.boundaries.add(float(time))
        if not any(n=="assimilation_nudge" for n,_ in self.processes):
            def nudge_process(t,y,dy):
                for i,(z,g) in self.nudges.items():dy[i]+=g*(z-y[i])
            self.process("assimilation_nudge",nudge_process)

    def rhs(self,t,y):
        dy=np.zeros_like(y)
        for _,function in self.processes: function(t,y,dy)
        if not np.all(np.isfinite(dy)): raise ValueError("Non-finite derivative")
        return dy

    def integrate(self,horizon,rtol=1e-7,atol=1e-9):
        schedule=sorted({0.0,float(horizon)}|{t for t in self.boundaries if 0<=t<=horizon})
        early=[t+offset for t,_,_ in self.impulses for offset in (.02,.05,.1,.2,.5,1) if 0<=t+offset<=horizon]
        grid=np.unique(np.r_[np.linspace(0,horizon,481),schedule,early])
        output=np.empty((len(self.states),len(grid)))
        y=np.array([s.initial for s in self.states],dtype=float)
        impulses={}
        for t,key,amount in self.impulses:
            impulses.setdefault(t,[]).append((self.indices[key],amount))
        updates={}
        for t,index,target,gain,kind in self.updates:
            updates.setdefault(t,[]).append((index,target,gain,kind))
        nfev=0
        for start,end in zip(schedule[:-1],schedule[1:]):
            for index,amount in impulses.get(start,[]): y[index]+=amount
            for index,target,gain,kind in updates.get(start,[]):
                if kind=="nudge":self.nudges[index]=(target,gain)
                else:y[index]+=gain*(target-y[index])
            mask=(grid>=start)&(grid<end)
            sample=grid[mask]
            # Events use left-side forcing through the segment endpoint. The
            # next segment then evaluates the new boundary condition exactly.
            def rhs(t,state):
                return self.rhs(min(t,np.nextafter(end,start)),state)
            solution=solve_ivp(rhs,(start,end),y,method="LSODA",rtol=rtol,atol=atol,t_eval=np.r_[sample,end],max_step=2.0)
            if not solution.success: raise ValueError("Numerical integration failed: "+solution.message)
            output[:,mask]=solution.y[:,:-1];y=solution.y[:,-1];nfev+=solution.nfev
        for index,amount in impulses.get(horizon,[]):y[index]+=amount
        output[:,-1]=y
        if not np.all(np.isfinite(output)):raise ValueError("Invalid computation result")
        for i,state in enumerate(self.states):
            if state.nonnegative and output[i].min() < -1e-6:
                raise ValueError(f"Negative state encountered: {state.key}")
        return grid,output,{"solver":"SciPy LSODA","rtol":rtol,"atol":atol,"nfev":nfev,"event_boundaries":schedule,"nonnegative_check":True}
