import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FuzzyPriority:
    """Mamdani fuzzy inference: normalized grid stress, renewable surplus, EV urgency -> stakeholder weights."""
    def __init__(self):
        self.grid=ctrl.Antecedent(np.linspace(0,1,101),"grid")
        self.renew=ctrl.Antecedent(np.linspace(0,1,101),"renew")
        self.urgency=ctrl.Antecedent(np.linspace(0,1,101),"urgency")
        self.gout=ctrl.Consequent(np.linspace(0,1,101),"gout")
        self.sout=ctrl.Consequent(np.linspace(0,1,101),"sout")
        self.uout=ctrl.Consequent(np.linspace(0,1,101),"uout")
        for v in [self.grid,self.renew,self.urgency]:
            v["low"]=fuzz.trimf(v.universe,[0,0,.5])
            v["medium"]=fuzz.trimf(v.universe,[0,.5,1])
            v["high"]=fuzz.trimf(v.universe,[.5,1,1])
        for v in [self.gout,self.sout,self.uout]:
            v["low"]=fuzz.trimf(v.universe,[0,0,.5])
            v["medium"]=fuzz.trimf(v.universe,[0,.5,1])
            v["high"]=fuzz.trimf(v.universe,[.5,1,1])
        rules=[
          ctrl.Rule(self.grid["low"],self.gout["low"]),
          ctrl.Rule(self.grid["medium"],self.gout["medium"]),
          ctrl.Rule(self.grid["high"],self.gout["high"]),
          ctrl.Rule(self.renew["low"],self.sout["low"]),
          ctrl.Rule(self.renew["medium"],self.sout["medium"]),
          ctrl.Rule(self.renew["high"],self.sout["high"]),
          ctrl.Rule(self.urgency["low"],self.uout["low"]),
          ctrl.Rule(self.urgency["medium"],self.uout["medium"]),
          ctrl.Rule(self.urgency["high"],self.uout["high"]),
          ctrl.Rule(self.grid["high"] & self.urgency["high"],self.gout["high"]),
          ctrl.Rule(self.renew["high"] & self.urgency["low"],self.sout["high"]),
          ctrl.Rule(self.grid["low"] & self.urgency["high"],self.uout["high"]),
          ctrl.Rule(self.grid["high"] & self.renew["low"],self.sout["low"]),
        ]
        self.system=ctrl.ControlSystem(rules)
    def __call__(self,grid,renew,urgency):
        for k,v in {"grid":grid,"renew":renew,"urgency":urgency}.items():
            if not 0<=float(v)<=1: raise ValueError(f"{k} must be in [0,1]")
        sim=ctrl.ControlSystemSimulation(self.system,clip_to_bounds=True)
        sim.input["grid"]=float(grid); sim.input["renew"]=float(renew); sim.input["urgency"]=float(urgency)
        sim.compute()
        scores=np.maximum([sim.output["gout"],sim.output["sout"],sim.output["uout"]],1e-6)
        w=scores/np.sum(scores)
        return {"w_grid":float(w[0]),"w_station":float(w[1]),"w_user":float(w[2])}
