from src.fuzzy import FuzzyPriority
from src.data_sources import make_evs, synthetic_pv
from src.engine import run_dispatch

def test_weights_sum():
    w=FuzzyPriority()(0.6,0.4,0.8)
    assert abs(w["w_grid"]+w["w_station"]+w["w_user"]-1)<1e-6

def test_dispatch_length():
    out,weights,metrics=run_dispatch(make_evs(10,3),synthetic_pv(40))
    assert len(out)==24 and len(weights)==24
    assert metrics["renewable_utilization_pct"]>=0
