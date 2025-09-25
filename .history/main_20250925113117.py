import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(page_title="Spend This — Lump Sum", layout="wide")
st.title("Spend This — Lump Sum Opportunity Cost")

# ---------- Helpers ----------
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def _canonical_alloc_name(col: str) -> str | None:
    s_raw = str(col).strip()
    s = s_raw.lower().replace("_","").replace(" ","")
    m_exact = re.fullmatch(r"(\d{1,3})e", s)
    if m_exact:
        pct = max(0, min(100, int(m_exact.group(1))))
        return f"{pct}E"
    if s.endswith("100f"):
        return "0E"
    m = re.search(r"(\d{1,3})e$", s)
    if m:
        pct = max(0, min(100, int(m.group(1))))
        return f"{pct}E"
    m2 = re.search(r"(\d{1,3})$", s)
    if m2 and any(prefix in s for prefix in ("lbm","spx","glob","global")):
        pct = max(0, min(100, int(m2.group(1))))
        return f"{pct}E"
    return None

def load_factors(path: str) -> pd.DataFrame:
    df = load_csv(path).apply(pd.to_numeric, errors="coerce")
    canon_cols = {}
    for c in df.columns:
        canon = _canonical_alloc_name(c)
        if canon and canon not in canon_cols:
            canon_cols[canon] = df[c]
    out = pd.DataFrame(canon_cols).dropna(axis=1, how="all")
    if out.empty:
        out = df.dropna(axis=1, how="all")
    def _k(n):
        try:
            return int(str(n).replace("E",""))
        except Exception:
            return 999
    return out.reindex(sorted(out.columns, key=_k), axis=1)

def build_windows(df: pd.DataFrame, alloc_col: str, years: int, step: int = 12, fee_mult_per_step: float = 1.0) -> pd.DataFrame:
    arr = df[alloc_col].values.astype(float)
    max_start = len(arr) - years * step
    rows = []
    if max_start < 0:
        return pd.DataFrame(columns=["start_index","factors","fv_multiple"])
    for s in range(0, max_start + 1):
        idxs = s + np.arange(years) * step
        if idxs[-1] >= len(arr):
            break
        window = arr[idxs]
        if np.isnan(window).any():
            continue
        fv = float(np.prod(window * fee_mult_per_step))
        rows.append((s, window * fee_mult_per_step, fv))
    return pd.DataFrame(rows, columns=["start_index","factors","fv_multiple"])

@st.cache_data
def load_all_data():
    df_glob = load_factors("data/global_factors.csv")
    df_spx  = load_factors("data/spx_factors.csv")
    try:
        df_w = load_csv("data/withdrawals.csv")
        df_w["Median"] = pd.to_numeric(df_w["Median"], errors="coerce")
    except Exception:
        df_w = None
    try:
        df_wx = load_csv("data/withdrawals_spx.csv")
        df_wx["Median"] = pd.to_numeric(df_wx["Median"], errors="coerce")
    except Exception:
        df_wx = None
    return df_glob, df_spx, df_w, df_wx

def lookup_withdrawal(df_w: pd.DataFrame|None, yrs: int):
    if df_w is None:
        return None
    try:
        return float(df_w.loc[df_w["Years"] == yrs, "Median"].iloc[0])
    except Exception:
        return None

# ---------- Inputs ----------
with st.sidebar:
    st.header("Inputs")
    current_age = st.number_input("Current Age", 18, 100, 30)
    retirement_age = st.number_input("Retirement Age", 30, 100, 65)
    retirement_years = st.slider("Years in Retirement", 20, 35, 30, 1)
    st.markdown("---")
    thinking = st.number_input("Thinking of Spending ($)", 0, value=15000, step=500)
    whatif   = st.number_input("What if I Spend This Instead ($)", 0, value=5000, step=500)
    st.caption(f"Difference to invest: **${max(0, thinking-whatif):,.0f}**")

years_to_retire = retirement_age - current_age
if years_to_retire <= 0:
    st.error("Retirement age must be greater than current age.")
    st.stop()

diff = max(0, thinking - whatif)
if diff == 0:
    st.info("No opportunity cost yet — increase the 'Thinking of Spending' amount.")
    st.stop()

# ---------- Data ----------
df_glob, df_spx, df_w, df_wx = load_all_data()
fee_mult_glob = (1.0 - 0.0020) ** (12/12)  # Global 20 bps/yr
fee_mult_spx  = (1.0 - 0.0005) ** (12/12)  # SPX 5 bps/yr

allocs = sorted(set(df_glob.columns).intersection(df_spx.columns),
                key=lambda x: int(str(x).replace("E","")), reverse=True)
if not allocs:
    st.error("No common allocation columns between global and SPX factors.")
    st.stop()

# ---------- Table: Min & Median by Allocation ----------
rows = []
raw_rows = []
for a in allocs:
    g = build_windows(df_glob, a, years_to_retire, 12, fee_mult_glob)
    s = build_windows(df_spx,  a, years_to_retire, 12, fee_mult_spx)
    gmin = float((g["fv_multiple"]*diff).min()) if not g.empty else np.nan
    gmed = float((g["fv_multiple"]*diff).median()) if not g.empty else np.nan
    smin = float((s["fv_multiple"]*diff).min()) if not s.empty else np.nan
    smed = float((s["fv_multiple"]*diff).median()) if not s.empty else np.nan
    rows.append({
        "Allocation": a,
        "Global Minimum Ending Value": (None if np.isnan(gmin) else f"${gmin:,.0f}"),
        "SPX Minimum Ending Value":    (None if np.isnan(smin) else f"${smin:,.0f}"),
        "Global Median Ending Value":  (None if np.isnan(gmed) else f"${gmed:,.0f}"),
        "SPX Median Ending Value":     (None if np.isnan(smed) else f"${smed:,.0f}"),
    })
    raw_rows.append({
        "Allocation": a,
        "Global Minimum Ending Value": gmin,
        "SPX Minimum Ending Value":    smin,
        "Global Median Ending Value":  gmed,
        "SPX Median Ending Value":     smed,
    })

st.subheader("Opportunity Cost of Lump Sum (Min & Median by Allocation)")
st.caption("Assumes annual expense ratios: Global 0.20%, SP500 0.05%.")
disp = pd.DataFrame(rows)[["Allocation","Global Minimum Ending Value","SPX Minimum Ending Value",
                           "Global Median Ending Value","SPX Median Ending Value"]]
st.dataframe(disp, use_container_width=True)

# ---------- Median Withdrawal (60/40 lookup) ----------
raw_df = pd.DataFrame(raw_rows)
w_g = lookup_withdrawal(df_w, retirement_years)
w_s = lookup_withdrawal(df_wx, retirement_years)

def _best(col_med, col_min):
    if raw_df.empty:
        return 0.0, 0.0
    idx = int(np.nanargmax(pd.to_numeric(raw_df[col_med], errors="coerce").values))
    return float(raw_df[col_med].iloc[idx] or 0.0), float(raw_df[col_min].iloc[idx] or 0.0)

g_med_ev, _ = _best("Global Median Ending Value","Global Minimum Ending Value")
s_med_ev, _ = _best("SPX Median Ending Value","SPX Minimum Ending Value")

rows2 = []
if w_g is not None and np.isfinite(g_med_ev):
    ann_g = w_g * g_med_ev if w_g <= 1.0 else w_g
    rows2.append({"Portfolio":"Global","Years":retirement_years,
                  "Median Ending Value": g_med_ev,
                  "Annual Retirement Income (Historically)":ann_g,
                  "Total Median Retirement Income":ann_g*retirement_years})
if w_s is not None and np.isfinite(s_med_ev):
    ann_s = w_s * s_med_ev if w_s <= 1.0 else w_s
    rows2.append({"Portfolio":"SPX","Years":retirement_years,
                  "Median Ending Value": s_med_ev,
                  "Annual Retirement Income (Historically)":ann_s,
                  "Total Median Retirement Income":ann_s*retirement_years})

if rows2:
    st.subheader("Median Withdrawal — Lump Sum")
    out = pd.DataFrame(rows2)
    fmt = out.copy()
    # Format currency with commas and no decimals
    fmt["Annual Retirement Income (Historically)"] = fmt["Annual Retirement Income (Historically)"].map(lambda v: f"${v:,.0f}")
    fmt["Total Median Lifetime Retirement Income"] = fmt["Total Median Retirement Income"].map(lambda v: f"${v:,.0f}")
    fmt["Median Ending Value"] = fmt["Median Ending Value"].map(lambda v: f"${v:,.0f}")
    # Drop the original numeric lifetime column to avoid duplication
    fmt = fmt.drop(columns=["Total Median Retirement Income"], errors="ignore")
    # Enforce a tidy column order
    fmt = fmt[[
        "Portfolio",
        "Years",
        "Median Ending Value",
        "Annual Retirement Income (Historically)",
        "Total Median Lifetime Retirement Income",
    ]]
    st.dataframe(fmt, use_container_width=True)