import numpy as np
import pandas as pd
from src.fuzzy import FuzzyPriority

def run_dispatch(evs,pv,grid_limit_kw=90,adaptive=True,fixed_weights=(.34,.33,.33)):
    """One-day urgency-first heuristic for a demonstration; not COA/NSGA-II."""
    if len(pv)!=24: raise ValueError("PV profile must have 24 hourly values")
    evs=evs.copy().reset_index(drop=True)
    remaining={str(r.ev_id):float(r.energy_kwh) for r in evs.itertuples()}
    fuzzy=FuzzyPriority()
    price=.16+.09*np.exp(-((np.arange(24)-19)/3.2)**2)+.035*np.exp(-((np.arange(24)-9)/2.5)**2)
    sched=[]; logs=[]
    for h in range(24):
        active=evs[(evs.arrival_hour<=h)&(evs.departure_hour>h)].copy()
        solar=float(max(0,pv[h])); total=sum(remaining[str(i)] for i in active.ev_id)
        grid_stress=float(np.clip(total/max(grid_limit_kw*2.5,1),0,1))
        surplus=float(np.clip((solar-min(total,grid_limit_kw))/max(grid_limit_kw,1),0,1))
        uvals=[np.clip(remaining[str(r.ev_id)]/(max(1,r.departure_hour-h)*r.max_power_kw),0,1) for r in active.itertuples()]
        urgency=float(np.mean(uvals)) if uvals else 0.
        if adaptive:
            w=fuzzy(grid_stress,surplus,urgency)
            wg,ws,wu=w["w_grid"],w["w_station"],w["w_user"]
        else: wg,ws,wu=map(float,fixed_weights)
        budget=solar+grid_limit_kw*np.clip(.45+.8*wu-.35*wg,.15,1)
        used=0.
        if not active.empty:
            active["rank"]=active.apply(lambda r: remaining[str(r.ev_id)]/
                max(1,(r.departure_hour-h)*r.max_power_kw),axis=1)
            active=active.sort_values("rank",ascending=False)
        for r in active.itertuples():
            key=str(r.ev_id); p=min(float(r.max_power_kw),remaining[key],max(0,budget-used))
            if p>0: remaining[key]=max(0,remaining[key]-p); used+=p
        pv_used=min(solar,used); grid=max(0,used-pv_used)
        sched.append({"hour":h,"pv_available_kw":solar,"pv_used_kw":pv_used,"ev_charge_kw":used,
            "grid_import_kw":grid,"price_per_kwh":float(price[h]),"unserved_energy_kwh":sum(remaining.values()),
            "grid_stress":grid_stress,"renewable_surplus":surplus,"ev_urgency":urgency})
        logs.append({"hour":h,"w_grid":wg,"w_station":ws,"w_user":wu})
    out=pd.DataFrame(sched); weights=pd.DataFrame(logs)
    metrics={"grid_energy_kwh":out.grid_import_kw.sum(),"peak_grid_import_kw":out.grid_import_kw.max(),
      "renewable_used_kwh":out.pv_used_kw.sum(),"renewable_available_kwh":np.sum(pv),
      "renewable_utilization_pct":100*out.pv_used_kw.sum()/max(np.sum(pv),1e-9),
      "energy_delivered_kwh":out.ev_charge_kw.sum(),
      "unserved_energy_kwh":out.unserved_energy_kwh.iloc[-1],
      "user_energy_cost_proxy":(out.ev_charge_kw*out.price_per_kwh).sum(),
      "degradation_cost_proxy":out.ev_charge_kw.sum()*.02,
      "station_profit_proxy":(out.ev_charge_kw*out.price_per_kwh*1.25).sum()-(out.grid_import_kw*out.price_per_kwh).sum()}
    return out,weights,metrics
