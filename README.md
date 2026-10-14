# EV–V2G Fuzzy-Adaptive Viva Demo

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Sources:
- NASA POWER hourly API docs: https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
- NASA POWER point endpoint: https://power.larc.nasa.gov/api/temporal/hourly/point
- ACN-Data: https://ev.caltech.edu/dataset

Research boundary: this is a viva prototype using a transparent heuristic, not a faithful reproduction of the reference paper's COA or IEEE 33/118-bus power-flow model. Synthetic scenario results are kept separate from paper-reported improvement ranges. The paper's raw experimental data was not available for direct replication.

Reference paper: M. Thirumalai et al., “Multi-objective coordinated optimization of renewable-based EV charging stations with user-centric scheduling and grid support in distribution networks,” e-Prime, 14 (2025), 101111.
