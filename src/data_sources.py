import requests
import numpy as np
import pandas as pd

NASA_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

def fetch_nasa_power(lat, lon, start, end, timeout=45):
    params = {"parameters":"ALLSKY_SFC_SW_DWN,T2M", "community":"RE",
              "longitude":lon, "latitude":lat, "start":start, "end":end,
              "format":"JSON", "time-standard":"UTC"}
    r = requests.get(NASA_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json().get("properties", {}).get("parameter")
    if not data:
        raise ValueError("Unexpected or empty NASA POWER response")
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d%H", errors="coerce")
    df.index.name = "timestamp"
    df = df.rename(columns={"ALLSKY_SFC_SW_DWN":"solar_radiation", "T2M":"temperature_c"})
    return df.replace(-999, np.nan).apply(pd.to_numeric, errors="coerce").dropna(how="all").sort_index()

def solar_to_pv_proxy(weather, capacity_kw=50):
    # Demo-only proxy; verify API units and replace with a validated PV model for research.
    return (weather["solar_radiation"].clip(lower=0).fillna(0)/1000).clip(0,1)*float(capacity_kw)

def synthetic_pv(capacity_kw=50):
    h=np.arange(24)
    return float(capacity_kw)*np.maximum(0,np.sin(np.pi*(h-6)/12))*0.78

def make_evs(n=28, seed=12):
    rng=np.random.default_rng(seed)
    arrival=rng.integers(6,19,size=n)
    departure=np.minimum(arrival+rng.integers(3,9,size=n),24)
    return pd.DataFrame({"ev_id":[f"EV-{i+1:03d}" for i in range(n)],
        "arrival_hour":arrival,"departure_hour":departure,
        "energy_kwh":rng.uniform(7,25,size=n).round(2),
        "max_power_kw":rng.choice([7.2,11.0],size=n,p=[.72,.28])}).sort_values("arrival_hour").reset_index(drop=True)

def normalize_evs(df):
    required={"arrival_hour","departure_hour","energy_kwh"}
    if not required.issubset(df.columns): raise ValueError(f"CSV must include {sorted(required)}")
    out=df.copy()
    for c in required: out[c]=pd.to_numeric(out[c],errors="raise")
    if "max_power_kw" not in out: out["max_power_kw"]=7.2
    out["max_power_kw"]=pd.to_numeric(out["max_power_kw"],errors="raise")
    if "ev_id" not in out: out["ev_id"]=[f"EV-{i+1:03d}" for i in range(len(out))]
    if ((out.arrival_hour<0)|(out.arrival_hour>23)).any(): raise ValueError("arrival_hour must be 0..23")
    if ((out.departure_hour<=out.arrival_hour)|(out.departure_hour>24)).any(): raise ValueError("departure_hour must be greater than arrival and <=24")
    if (out.energy_kwh<=0).any() or (out.max_power_kw<=0).any(): raise ValueError("Energy/power must be positive")
    return out.reset_index(drop=True)
