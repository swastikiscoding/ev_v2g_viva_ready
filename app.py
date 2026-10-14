from datetime import date, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_sources import fetch_nasa_power, solar_to_pv_proxy, synthetic_pv, make_evs, normalize_evs
from src.engine import run_dispatch

st.set_page_config(page_title="EV–V2G Fuzzy-Adaptive Demo", layout="wide")
st.title("Fuzzy-Adaptive EV Charging & V2G")
st.caption("Viva-ready simulation | Renewable integration · Fuzzy priorities · Charging schedule")

with st.sidebar:
    st.header("Scenario controls")
    lat=st.number_input("Latitude",value=28.7041,format="%.4f")
    lon=st.number_input("Longitude",value=77.1025,format="%.4f")
    day=st.date_input("NASA POWER date",value=date(2024,5,15),max_value=date.today()-timedelta(days=2))
    pv_cap=st.slider("PV capacity (kW)",10,150,50,10)
    grid_cap=st.slider("Grid import limit (kW)",30,180,90,10)
    n=st.slider("Synthetic EV sessions",5,70,28)
    seed=st.number_input("Random seed",0,99999,12)
    up=st.file_uploader("Optional normalized EV CSV",type=["csv"])
    api=st.checkbox("Use NASA POWER API",False)
    fetch=st.button("Fetch solar/weather")

if "weather" not in st.session_state: st.session_state.weather=None
if fetch and api:
    try:
        with st.spinner("Fetching hourly NASA POWER data..."):
            st.session_state.weather=fetch_nasa_power(lat,lon,day.strftime("%Y%m%d"),(day+timedelta(days=1)).strftime("%Y%m%d"))
        st.success("NASA POWER request succeeded.")
    except Exception as e:
        st.session_state.weather=None
        st.error(f"API failed: {e}. Use synthetic fallback.")
weather=st.session_state.weather if api else None
if weather is not None:
    ps=solar_to_pv_proxy(weather,pv_cap)
    pv=ps.groupby(ps.index.hour).mean().reindex(range(24),fill_value=0).to_numpy()
    source="NASA POWER weather → simplified PV proxy"
else:
    pv=synthetic_pv(pv_cap); source="synthetic PV fallback"
try:
    evs=normalize_evs(pd.read_csv(up)) if up is not None else make_evs(n,int(seed))
except Exception as e:
    st.error(f"CSV import error: {e}")
    evs=make_evs(n,int(seed))

adaptive,aw,am=run_dispatch(evs,pv,grid_cap,True)
fixed,fw,fm=run_dispatch(evs,pv,grid_cap,False)
st.info(f"Data source: **{source}**. EV sessions are synthetic unless a CSV was imported.")
st.warning("Honesty note: local results below come from a heuristic simulation, not the paper's COA or IEEE feeder power-flow model. Paper-reported results are shown separately as context, not direct head-to-head measurements.")

t1,t2,t3,t4,t5=st.tabs(["Overview","Fuzzy controller","Local comparison","Reference paper","Sources & assumptions"])
with t1:
    st.subheader("System overview")
    c=st.columns(4)
    c[0].metric("EV sessions",len(evs))
    c[1].metric("Energy scheduled",f"{am['energy_delivered_kwh']:.1f} kWh")
    c[2].metric("Peak grid import",f"{am['peak_grid_import_kw']:.1f} kW")
    c[3].metric("Unserved request proxy",f"{am['unserved_energy_kwh']:.1f} kWh")
    df=adaptive[["hour","pv_available_kw","pv_used_kw","ev_charge_kw","grid_import_kw"]].melt("hour",var_name="Series",value_name="Power (kW)")
    st.plotly_chart(px.line(df,x="hour",y="Power (kW)",color="Series",markers=True,title="Hourly renewable availability/use, EV charging and grid import"),use_container_width=True)
    st.plotly_chart(px.line(adaptive,x="hour",y="price_per_kwh",markers=True,title="Illustrative time-varying electricity tariff"),use_container_width=True)
with t2:
    st.subheader("Fuzzy inference system")
    st.write("Inputs: normalized grid stress, renewable surplus, EV urgency. Outputs: grid, station, and user weights. Each hour's weights sum to 1.")
    long=aw.melt("hour",var_name="Stakeholder",value_name="Weight")
    st.plotly_chart(px.line(long,x="hour",y="Weight",color="Stakeholder",markers=True,title="Fuzzy-adaptive stakeholder weights"),use_container_width=True)
    il=adaptive[["hour","grid_stress","renewable_surplus","ev_urgency"]].melt("hour",var_name="Input",value_name="Normalized value")
    st.plotly_chart(px.line(il,x="hour",y="Normalized value",color="Input",title="Fuzzy-controller inputs"),use_container_width=True)
    st.dataframe(aw.round(3),hide_index=True,use_container_width=True)
with t3:
    st.subheader("Fixed weights vs fuzzy-adaptive weights")
    metrics=["grid_energy_kwh","peak_grid_import_kw","renewable_used_kwh","energy_delivered_kwh","unserved_energy_kwh","user_energy_cost_proxy","degradation_cost_proxy","station_profit_proxy"]
    labels=["Grid energy (kWh)","Peak import (kW)","Renewable used (kWh)","Energy scheduled (kWh)","Unserved request proxy (kWh)","User cost proxy","Degradation cost proxy","Station profit proxy"]
    comp=pd.DataFrame({"Metric":labels,"Fixed-weight heuristic":[fm[m] for m in metrics],"Fuzzy-adaptive heuristic":[am[m] for m in metrics]})
    st.dataframe(comp.round(3),hide_index=True,use_container_width=True)
    d=pd.DataFrame({"hour":adaptive.hour,"Adaptive charging":adaptive.ev_charge_kw,"Fixed charging":fixed.ev_charge_kw,"Adaptive grid import":adaptive.grid_import_kw,"Fixed grid import":fixed.grid_import_kw}).melt("hour",var_name="Series",value_name="kW")
    st.plotly_chart(px.line(d,x="hour",y="kW",color="Series",title="Dispatch comparison"),use_container_width=True)
    st.caption("Economic and degradation measures are rough proxies, not validated economic/electrochemical models.")
with t4:
    st.subheader("Reference-paper results (reported ranges)")
    st.markdown("""**M. Thirumalai et al.**, “Multi-objective coordinated optimization of renewable-based EV charging stations with user-centric scheduling and grid support in distribution networks,” *e-Prime*, 14 (2025), 101111.

The paper describes COA-based multi-objective scheduling with PV, wind, BESS, EV charging/V2G, Monte Carlo uncertainty, and IEEE 33-bus/118-bus distribution networks. Its objectives cover grid losses/voltage deviation, station economics, and user charging/degradation cost.""")
    ref=pd.DataFrame({"Outcome":["Network losses","Voltage profile","EVCS profitability","EV operating costs"],
      "Reported midpoint (%)":[16,42.5,57.5,42.5],"Low (%)":[15,40,55,40],"High (%)":[17,45,60,45]})
    st.plotly_chart(px.bar(ref,x="Outcome",y="Reported midpoint (%)",error_y=ref["High (%)"]-ref["Reported midpoint (%)"],error_y_minus=ref["Reported midpoint (%)"]-ref["Low (%)"],title="Paper-reported improvement ranges (midpoint and range)"),use_container_width=True)
    st.caption("The paper reports 15–17% lower network losses, 40–45% improved voltage profiles, 55–60% higher EVCS profitability, and 40–45% lower EV operating costs for its coordinated Scenario 1 relative to stressed uncoordinated Scenario 2. These are not measured by this prototype.")
    st.markdown("**Objective intent:** grid losses + voltage deviation; EVCS net economics; user charging cost + degradation cost. The exact paper equations and sign conventions must be cited from the PDF, not inferred from this chart.")
    st.latex(r"J(t)=w_g(t)\\widehat F_g(t)+w_s(t)\\widehat F_s(t)+w_u(t)\\widehat F_u(t)")
    st.write("The time-varying normalized-weight scalarization above is the prototype's proposed extension, not a verbatim claim that the paper uses fuzzy weights.")
with t5:
    st.subheader("Sources and limitations")
    st.markdown("""
- NASA POWER hourly API: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- Endpoint: `https://power.larc.nasa.gov/api/temporal/hourly/point`
- ACN-Data: https://ev.caltech.edu/dataset

**Current limitations**
1. Synthetic EV data by default; imported CSV must be normalized to arrival_hour, departure_hour, energy_kwh, and optionally max_power_kw.
2. Solar-to-PV conversion is a proxy, not a validated PV performance model.
3. Dispatch is urgency-first heuristic, not COA or NSGA-II.
4. No IEEE bus power flow, voltage calculation, or line-loss calculation.
5. Battery degradation is only a simple throughput-cost proxy.
6. Bidirectional V2G and full EV SoC accounting are not implemented in this viva build.
""")
    st.download_button("Download adaptive schedule CSV",adaptive.to_csv(index=False).encode(),"adaptive_schedule.csv","text/csv")
    st.download_button("Download fixed schedule CSV",fixed.to_csv(index=False).encode(),"fixed_schedule.csv","text/csv")
st.subheader("EV sessions in this run")
st.dataframe(evs,hide_index=True,use_container_width=True)
