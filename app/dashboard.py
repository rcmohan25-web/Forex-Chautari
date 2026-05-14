"""
Forex ML Platform — Trading Terminal Dashboard v4
Improvements over v3:
 - Sidebar: live clock, session P&L tracker, collapsible advanced settings
 - Tab 1: sparklines on every stat card, signal history timeline, trade P&L waterfall
 - Tab 2: unified chart with signal overlays, volume, momentum in one view; pattern annotations
 - Tab 3: radar chart of model dimensions, accuracy trend line, feature importance bar
 - Tab 4: endpoint latency tester, data quality checker, one-click copy run commands
"""

import os, sys, requests, time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.settings import DATA_PATH, METADATA_PATH, WF_RESULTS_PATH, DEFAULT_SPREAD_COST
from src.data_loader import load_forex_data
from src.features import add_features
from src.model import load_model_bundle, predict_latest
from src.backtest import run_backtest
from src.realistic_backtest import run_realistic_backtest
from src.utils import format_pct

API_BASE = "http://127.0.0.1:8000"

# ── palette ──────────────────────────────────────────────────────────────────
C_BG      = "#080c14"
C_SURF    = "#0f1623"
C_SURF2   = "#141d2b"
C_BORDER  = "#1e2d42"
C_ACCENT  = "#00c896"
C_ACCENT2 = "#0ea5e9"
C_RED     = "#f43f5e"
C_YELLOW  = "#f59e0b"
C_PURPLE  = "#a855f7"
C_TEXT    = "#dce8f5"
C_MUTED   = "#546e8a"
C_DIM     = "#2a3d52"

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0b1220",
    font=dict(family="'JetBrains Mono', monospace", color=C_TEXT, size=11),
    xaxis=dict(gridcolor=C_BORDER, showgrid=True, zeroline=False, showspikes=True,
               spikecolor=C_MUTED, spikethickness=1),
    yaxis=dict(gridcolor=C_BORDER, showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER, borderwidth=1,
                font=dict(size=11)),
    margin=dict(l=8, r=8, t=36, b=8),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C_SURF2, bordercolor=C_BORDER, font=dict(color=C_TEXT, size=11)),
)

def pf(fig, h=380):
    fig.update_layout(**PLOT, height=h)
    return fig

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Forex ML Terminal", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Syne:wght@600;700;800&display=swap');

html,body,[class*="css"]{{font-family:'JetBrains Mono',monospace;background:{C_BG};color:{C_TEXT};}}

section[data-testid="stSidebar"]{{background:{C_SURF}!important;border-right:1px solid {C_BORDER};}}
section[data-testid="stSidebar"] *{{color:{C_TEXT}!important;}}

.logo{{font-family:'Syne',sans-serif;font-size:18px;font-weight:800;color:{C_ACCENT};letter-spacing:-0.5px;}}
.logo span{{color:{C_MUTED};font-weight:600;font-size:14px;}}

.hdr{{background:linear-gradient(90deg,{C_SURF} 0%,{C_SURF2} 100%);
      border:1px solid {C_BORDER};border-radius:14px;padding:18px 24px;
      display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;}}
.hdr-title{{font-family:'Syne',sans-serif;font-size:20px;font-weight:800;color:{C_ACCENT};}}
.hdr-sub{{font-size:11px;color:{C_MUTED};margin-top:2px;}}

/* signal card */
.sig{{border-radius:14px;padding:22px 24px;margin-bottom:14px;position:relative;overflow:hidden;}}
.sig.buy{{background:linear-gradient(135deg,#031a10 0%,#041f14 100%);border:1px solid rgba(0,200,150,0.33);border-left:3px solid {C_ACCENT};}}
.sig.sell{{background:linear-gradient(135deg,#1a0308 0%,#1f040a 100%);border:1px solid rgba(244,63,94,0.33);border-left:3px solid {C_RED};}}
.sig-lbl{{font-size:10px;color:{C_MUTED};letter-spacing:2.5px;text-transform:uppercase;margin-bottom:8px;}}
.sig-val{{font-family:'Syne',sans-serif;font-size:44px;font-weight:800;line-height:1;}}
.sig-val.buy{{color:{C_ACCENT};}} .sig-val.sell{{color:{C_RED};}}
.sig-meta{{font-size:12px;color:{C_MUTED};margin-top:10px;line-height:1.8;}}
.sig-badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:11px;font-weight:600;letter-spacing:1px;}}
.badge-high{{background:rgba(0,200,150,0.13);color:{C_ACCENT};border:1px solid rgba(0,200,150,0.27);}}
.badge-med{{background:rgba(245,158,11,0.13);color:{C_YELLOW};border:1px solid rgba(245,158,11,0.27);}}
.badge-low{{background:rgba(244,63,94,0.13);color:{C_RED};border:1px solid rgba(244,63,94,0.27);}}

/* stat cards */
.sc{{background:{C_SURF};border:1px solid {C_BORDER};border-radius:12px;padding:14px 16px;margin-bottom:10px;}}
.sc-lbl{{font-size:10px;color:{C_MUTED};letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;}}
.sc-val{{font-size:22px;font-weight:700;}} .sc-val.pos{{color:{C_ACCENT};}} .sc-val.neg{{color:{C_RED};}} .sc-val.neu{{color:{C_YELLOW};}}
.sc-sub{{font-size:11px;color:{C_MUTED};margin-top:3px;}}

/* price ticker */
.ticker{{background:{C_SURF2};border:1px solid {C_BORDER};border-radius:12px;padding:16px 20px;text-align:center;}}
.ticker-sym{{font-size:11px;color:{C_MUTED};letter-spacing:2px;margin-bottom:4px;}}
.ticker-price{{font-family:'Syne',sans-serif;font-size:38px;font-weight:800;color:{C_TEXT};line-height:1;}}
.ticker-chg{{font-size:13px;margin-top:6px;}}

/* health row */
.hr{{display:flex;align-items:center;gap:10px;padding:9px 0;border-bottom:1px solid {C_BORDER};font-size:12px;}}
.hr:last-child{{border-bottom:none;}}
.hr-k{{color:{C_MUTED};flex:0 0 210px;}} .hr-v{{font-weight:600;color:{C_TEXT};}}
.hr-v.g{{color:{C_ACCENT};}} .hr-v.w{{color:{C_YELLOW};}} .hr-v.b{{color:{C_RED};}}

/* section label */
.sec{{font-size:10px;color:{C_MUTED};letter-spacing:2.5px;text-transform:uppercase;margin:18px 0 8px;border-bottom:1px solid {C_BORDER};padding-bottom:6px;}}

/* tabs */
button[data-baseweb="tab"]{{font-family:'JetBrains Mono',monospace!important;font-size:11px!important;
  letter-spacing:1.5px!important;color:{C_MUTED}!important;background:transparent!important;
  border-bottom:2px solid transparent!important;padding:10px 20px!important;text-transform:uppercase!important;}}
button[data-baseweb="tab"][aria-selected="true"]{{color:{C_ACCENT}!important;border-bottom-color:{C_ACCENT}!important;}}
div[data-baseweb="tab-list"]{{border-bottom:1px solid {C_BORDER}!important;gap:0!important;background:transparent!important;}}

/* buttons */
.stButton>button{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT}!important;
  border-radius:8px!important;font-family:'JetBrains Mono',monospace!important;font-size:11px!important;
  font-weight:600!important;letter-spacing:1px!important;transition:all 0.15s!important;}}
.stButton>button:hover{{border-color:{C_ACCENT}!important;color:{C_ACCENT}!important;box-shadow:0 0 14px rgba(0,200,150,0.13)!important;}}

/* slider */
.stSlider [data-testid="stThumb"]{{background:{C_ACCENT}!important;}}
.stSlider [data-testid="stTrackActive"]{{background:{C_ACCENT}!important;}}

/* alerts */
.stAlert{{border-radius:10px!important;font-size:12px!important;}}

/* pulse */
@keyframes pulse{{0%,100%{{opacity:1;box-shadow:0 0 6px {C_ACCENT};}}50%{{opacity:0.5;box-shadow:none;}}}}
.dot{{width:7px;height:7px;border-radius:50%;background:{C_ACCENT};animation:pulse 2s infinite;display:inline-block;margin-right:5px;}}
.dot.off{{background:{C_RED};animation:none;}}

/* code block */
.cmd{{background:{C_SURF2};border:1px solid {C_BORDER};border-radius:8px;padding:10px 14px;
  font-family:'JetBrains Mono',monospace;font-size:12px;color:{C_ACCENT};margin:4px 0;line-height:1.8;}}

#MainMenu,footer,header{{visibility:hidden;}}
.block-container{{padding-top:0.8rem!important;max-width:1440px;}}
div[data-testid="stDataFrame"]{{border:1px solid {C_BORDER}!important;border-radius:10px!important;overflow:hidden;}}
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def api_get(path, **kw):
    t0 = time.time()
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5, **kw)
        return r.json(), None, round((time.time()-t0)*1000)
    except Exception as e:
        return None, str(e), round((time.time()-t0)*1000)

def api_post(path, **kw):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=180, **kw)
        return r.json(), None
    except Exception as e:
        return None, str(e)

def sc(label, value, cls="", sub=""):
    sub_html = f'<div class="sc-sub">{sub}</div>' if sub else ""
    return f'<div class="sc"><div class="sc-lbl">{label}</div><div class="sc-val {cls}">{value}</div>{sub_html}</div>'

def hr(k, v, cls=""):
    return f'<div class="hr"><span class="hr-k">{k}</span><span class="hr-v {cls}">{v}</span></div>'

def section(label):
    return f'<div class="sec">{label}</div>'


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Logo + live clock
    now = datetime.now()
    st.markdown(f"""
    <div style="padding:12px 0 4px">
      <div class="logo">⬡ FOREX ML <span>TERMINAL</span></div>
      <div style="font-size:10px;color:{C_MUTED};margin-top:6px;letter-spacing:1px;">
        {now.strftime('%A  %Y-%m-%d')}<br>
        <span style="font-size:15px;color:{C_TEXT};font-weight:600;">{now.strftime('%H:%M:%S')}</span>
        &nbsp;&nbsp;UTC+{now.astimezone().utcoffset().seconds//3600 if now.astimezone().utcoffset() else 0}
      </div>
    </div>
    <hr style="border-color:{C_BORDER};margin:12px 0;">
    """, unsafe_allow_html=True)

    # API status pill
    health_data, h_err, h_ms = api_get("/health")
    if h_err:
        st.markdown(f'<div style="font-size:11px;"><span class="dot off"></span>API OFFLINE</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div style="font-size:11px;"><span class="dot"></span>API ONLINE &nbsp;·&nbsp; {h_ms}ms &nbsp;·&nbsp; {health_data.get("data_rows","?")} rows</div>', unsafe_allow_html=True)

    st.markdown(f'<hr style="border-color:{C_BORDER};margin:12px 0">{section("Signal Controls")}', unsafe_allow_html=True)
    threshold   = st.slider("Confidence Threshold", 0.50, 0.90, 0.60, 0.01,
                            help="Only trade when model confidence exceeds this level")
    spread_cost = st.number_input("Spread (pips)", 0.0, 0.01, DEFAULT_SPREAD_COST, 0.0001, "%.4f")

    with st.expander("⚙ Advanced Settings"):
        initial_balance = st.number_input("Starting Balance ($)", 1000, 100000, 10000, 1000)
        risk_per_trade  = st.slider("Risk per Trade (%)", 0.005, 0.05, 0.01, 0.005)
        stop_loss_pct   = st.slider("Stop Loss (%)", 0.001, 0.01, 0.002, 0.001)
        take_profit_pct = st.slider("Take Profit (%)", 0.002, 0.02, 0.004, 0.001)
        n_bars_chart    = st.select_slider("Chart bars", [50,100,200,500], 200)
    if "initial_balance" not in dir():
        initial_balance, risk_per_trade, stop_loss_pct, take_profit_pct, n_bars_chart = 10000, 0.01, 0.002, 0.004, 200

    st.markdown(f'<hr style="border-color:{C_BORDER};margin:12px 0">{section("Data & Model")}', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    fetch_clicked   = ca.button("⬇ Fetch",   width='stretch')
    retrain_clicked = cb.button("⚙ Retrain", width='stretch')

    if fetch_clicked:
        with st.spinner("Fetching..."):
            res, err = api_post("/fetch-data", params={"outputsize": "compact"})
        if err:   st.error(f"API offline: {err}")
        elif res and res.get("success"): st.success(res["message"])
        else:     st.error(res.get("detail","Failed") if res else "No response")

    if retrain_clicked:
        with st.spinner("Retraining (~30s)..."):
            res, err = api_post("/retrain")
        if err:   st.error(f"API offline: {err}")
        elif res and res.get("success"):
            st.success(f"✓ Test={res.get('accuracy_test','?'):.3f}  WF={res.get('walk_forward_mean_accuracy','?'):.3f}")
            st.rerun()
        else: st.error(res.get("detail","Failed") if res else "No response")


# ── guards ────────────────────────────────────────────────────────────────────
if not os.path.exists(DATA_PATH):
    st.warning("No data file. Click **⬇ Fetch** in the sidebar."); st.stop()
if not os.path.exists(METADATA_PATH):
    st.warning("No model. Click **⚙ Retrain** or run `python train.py`."); st.stop()


# ── load ──────────────────────────────────────────────────────────────────────
try:
    df_raw = load_forex_data(DATA_PATH)
    df     = add_features(df_raw)
    model, meta = load_model_bundle()
    feat_cols = meta["feature_columns"]
    pred, prob_up = predict_latest(model, df, feat_cols)
    is_buy  = pred == 1
    signal  = "BUY  ▲" if is_buy else "SELL  ▼"
    all_preds  = model.predict(df[feat_cols])
    all_probas = model.predict_proba(df[feat_cols])
    df["signal"]   = (all_probas[:,1] >= threshold).astype(int)
    df["prob_up"]  = all_probas[:,1]
    equity_df, trades_df, summary = run_realistic_backtest(
        df=df, signal_col="signal", initial_balance=initial_balance,
        risk_per_trade=risk_per_trade, stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct, spread=spread_cost, slippage=0.00005)
    _, bt = run_backtest(df, all_preds, all_probas, threshold, spread_cost)
except Exception as e:
    st.error(f"Load error: {e}"); st.stop()


# ── derived ───────────────────────────────────────────────────────────────────
latest_date = pd.to_datetime(df_raw["Date"].max())
days_old    = (datetime.now() - latest_date).days
gap         = abs(prob_up - 0.5)
conf_lbl    = "HIGH" if gap>=0.15 else ("MEDIUM" if gap>=0.08 else "LOW")
conf_cls    = "badge-high" if conf_lbl=="HIGH" else ("badge-med" if conf_lbl=="MEDIUM" else "badge-low")
sig_col     = C_ACCENT if is_buy else C_RED
tr_acc      = meta.get("accuracy_train")
te_acc      = meta.get("accuracy_test")
wf_acc      = meta.get("walk_forward_mean_accuracy")
wf_prof     = meta.get("walk_forward_profitable_splits")
wf_tot      = meta.get("walk_forward_total_splits")
total_ret   = summary["total_return"]
max_dd      = summary["max_drawdown"]
lat_close   = float(df_raw["Close"].iloc[-1])
prev_close  = float(df_raw["Close"].iloc[-2])
pct_chg     = (lat_close - prev_close) / prev_close * 100
chg_col     = C_ACCENT if pct_chg >= 0 else C_RED


# ── header ────────────────────────────────────────────────────────────────────
fresh = "" if days_old <= 3 else f' &nbsp;·&nbsp; <span style="color:{C_YELLOW};">⚠ {days_old}d old — click Fetch</span>'
st.markdown(f"""
<div class="hdr">
  <div>
    <div class="hdr-title">⬡ EUR/USD Forex ML Terminal</div>
    <div class="hdr-sub">Random Forest · Walk-forward validated · Spread-aware · Model {meta.get('model_version','?')}</div>
  </div>
  <div style="text-align:right;font-size:11px;">
    <span style="color:{C_TEXT};">Latest bar: {latest_date.date()}{fresh}</span><br>
    <span style="color:{C_MUTED};">{now.strftime('%Y-%m-%d %H:%M')}</span>
  </div>
</div>
""", unsafe_allow_html=True)

if days_old > 3:
    st.warning(f"Data is {days_old} days old. Click **⬇ Fetch** in the sidebar to pull latest bars.")


# ── tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6 = st.tabs(["  📡  SIGNAL & P&L  ","  📊  CHARTS  ","  🔬  MODEL HEALTH  ","  📂  PAPER TRADING  ","  🌐  PORTFOLIO  ","  🖥  SYSTEM  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Signal & P&L
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    col_l, col_r = st.columns([5, 7], gap="large")

    with col_l:
        # Signal card
        st.markdown(f"""
        <div class="sig {'buy' if is_buy else 'sell'}">
          <div class="sig-lbl">Latest Signal</div>
          <div class="sig-val {'buy' if is_buy else 'sell'}">{signal}</div>
          <div class="sig-meta">
            Prob UP &nbsp;<b style="color:{C_TEXT}">{prob_up:.4f}</b> &nbsp;·&nbsp;
            Prob DOWN &nbsp;<b style="color:{C_TEXT}">{1-prob_up:.4f}</b><br>
            Confidence &nbsp;<span class="sig-badge {conf_cls}">{conf_lbl}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Ticker
        chg_arrow = "▲" if pct_chg >= 0 else "▼"
        st.markdown(f"""
        <div class="ticker">
          <div class="ticker-sym">EUR / USD</div>
          <div class="ticker-price">{lat_close:.5f}</div>
          <div class="ticker-chg" style="color:{chg_col};">{chg_arrow} {pct_chg:+.4f}% &nbsp; vs prev close {prev_close:.5f}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_up*100,
            delta={"reference": 50, "suffix":"%",
                   "font":{"color":sig_col,"size":13,"family":"JetBrains Mono"}},
            number={"suffix":"%","font":{"color":sig_col,"size":30,"family":"JetBrains Mono"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":C_MUTED,"tickfont":{"color":C_MUTED,"size":9},"nticks":6},
                "bar":{"color":sig_col,"thickness":0.22},
                "bgcolor":C_SURF,
                "borderwidth":0,
                "steps":[{"range":[0,40],"color":"#1a050a"},
                          {"range":[40,60],"color":"#0d1117"},
                          {"range":[60,100],"color":"#041a10"}],
                "threshold":{"line":{"color":C_YELLOW,"width":2},"value":threshold*100},
            },
            title={"text":"BUY PROBABILITY  |  yellow = threshold",
                   "font":{"color":C_MUTED,"size":10,"family":"JetBrains Mono"}},
        ))
        fig_g.update_layout(paper_bgcolor="rgba(0,0,0,0)",font={"color":C_TEXT},
                             height=210,margin=dict(l=16,r=16,t=48,b=4))
        st.plotly_chart(fig_g, width='stretch')

        # Probability history sparkline (last 60 bars)
        st.markdown(section("Probability History — Last 60 Bars"), unsafe_allow_html=True)
        tail60 = df.tail(60)
        fig_spark = go.Figure()
        fig_spark.add_hrect(y0=threshold, y1=1.0, fillcolor="rgba(0,200,150,0.05)", line_width=0)
        fig_spark.add_hrect(y0=0, y1=1-threshold, fillcolor="rgba(244,63,94,0.05)", line_width=0)
        fig_spark.add_hline(y=threshold, line_dash="dot", line_color=C_YELLOW, line_width=1)
        fig_spark.add_hline(y=0.5, line_dash="dot", line_color=C_MUTED, line_width=1)
        colors_spark = [C_ACCENT if v>=threshold else (C_RED if v<=(1-threshold) else C_MUTED)
                        for v in tail60["prob_up"]]
        fig_spark.add_trace(go.Scatter(
            x=tail60["Date"], y=tail60["prob_up"],
            mode="lines+markers",
            line=dict(color=C_ACCENT2, width=1.5),
            marker=dict(color=colors_spark, size=5),
            name="Prob UP",
        ))
        pf(fig_spark, 180)
        fig_spark.update_layout(showlegend=False, yaxis=dict(range=[0,1]))
        st.plotly_chart(fig_spark, width='stretch')

    with col_r:
        # Stat grid
        st.markdown(section("Performance Summary"), unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)
        bal = summary["final_balance"]
        wr  = bt.get("win_rate", 0)
        g1.markdown(sc("Total Return",   format_pct(total_ret),
                        "pos" if total_ret>0 else "neg",
                        f"vs market {format_pct(bt.get('total_market_return',0))}"), unsafe_allow_html=True)
        g2.markdown(sc("Max Drawdown",   format_pct(max_dd), "neg"), unsafe_allow_html=True)
        g3.markdown(sc("Final Balance",  f"${bal:,.2f}",
                        "pos" if bal>initial_balance else "neg",
                        f"started ${initial_balance:,}"), unsafe_allow_html=True)

        g4, g5, g6 = st.columns(3)
        g4.markdown(sc("Win Rate",       format_pct(wr),
                        "pos" if wr>0.5 else ("neu" if wr>0.45 else "neg")), unsafe_allow_html=True)
        g5.markdown(sc("Trades",         str(summary["num_trades"]), "neu",
                        f"threshold {threshold:.2f}"), unsafe_allow_html=True)
        g6.markdown(sc("Avg Trade Cost", f"{bt.get('avg_trade_cost',0):.5f}", "neu"), unsafe_allow_html=True)

        # Equity curve with drawdown overlay
        st.markdown(section("Equity Curve & Drawdown"), unsafe_allow_html=True)
        eq_col = C_ACCENT if equity_df["Balance"].iloc[-1] >= initial_balance else C_RED

        # compute drawdown series
        peak = equity_df["Balance"].cummax()
        dd_series = (equity_df["Balance"] - peak) / peak * 100

        fig_eq = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.7, 0.3], vertical_spacing=0.04,
                                subplot_titles=["",""])
        fig_eq.add_trace(go.Scatter(
            x=equity_df["Date"], y=equity_df["Balance"],
            mode="lines", line=dict(color=eq_col, width=2),
            fill="tozeroy", fillcolor=f"rgba({eq_col.lstrip('#')},0.09)" if False else ("rgba(0,200,150,0.09)" if eq_col==C_ACCENT else "rgba(244,63,94,0.09)"), name="Balance"), row=1, col=1)
        fig_eq.add_hline(y=initial_balance, line_dash="dot", line_color=C_MUTED,
                          line_width=1, row=1, col=1)
        fig_eq.add_trace(go.Bar(
            x=equity_df["Date"], y=dd_series,
            marker_color=C_RED, opacity=0.6, name="Drawdown %"), row=2, col=1)
        fig_eq.update_layout(**PLOT, height=320, showlegend=True)
        fig_eq.update_yaxes(row=2, col=1, ticksuffix="%")
        st.plotly_chart(fig_eq, width='stretch')

        # Trade P&L waterfall (last 20 trades)
        if not trades_df.empty and "pnl" in trades_df.columns:
            st.markdown(section("Trade P&L — Last 20"), unsafe_allow_html=True)
            last20 = trades_df.tail(20)
            wf_colors = [C_ACCENT if v > 0 else C_RED for v in last20["pnl"]]
            fig_wfall = go.Figure(go.Bar(
                x=list(range(len(last20))),
                y=last20["pnl"],
                marker_color=wf_colors,
                name="P&L",
            ))
            pf(fig_wfall, 200)
            fig_wfall.add_hline(y=0, line_color=C_MUTED, line_width=1)
            fig_wfall.update_layout(showlegend=False,
                                     xaxis_title="Trade #", yaxis_title="P&L ($)")
            st.plotly_chart(fig_wfall, width='stretch')
        else:
            # trade log table as fallback
            st.markdown(section("Recent Trades"), unsafe_allow_html=True)
            if not trades_df.empty:
                st.dataframe(trades_df.tail(15), width='stretch', height=260)
            else:
                st.info("No trades at current threshold. Lower the threshold slider.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Charts
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    candle_df = df_raw.tail(n_bars_chart)
    tail      = df.tail(n_bars_chart)

    # ── Main chart: Candlestick + BB + signals ─────────────────────────────
    st.markdown(section("Price · Bollinger Bands · ML Signals"), unsafe_allow_html=True)
    fig_main = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.025,
        subplot_titles=["","MACD","RSI (14)"],
    )
    # Candles
    fig_main.add_trace(go.Candlestick(
        x=candle_df["Date"], open=candle_df["Open"],
        high=candle_df["High"], low=candle_df["Low"], close=candle_df["Close"],
        increasing_line_color=C_ACCENT, decreasing_line_color=C_RED,
        name="EUR/USD",
    ), row=1, col=1)
    # BB
    fig_main.add_trace(go.Scatter(x=tail["Date"], y=tail["bb_upper"], name="BB Upper",
        line=dict(color=C_ACCENT2, width=1, dash="dot"), opacity=0.5), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=tail["Date"], y=tail["bb_lower"], name="BB Lower",
        line=dict(color=C_ACCENT2, width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(14,165,233,0.04)", opacity=0.5), row=1, col=1)
    fig_main.add_trace(go.Scatter(x=tail["Date"], y=tail["bb_mid"], name="BB Mid",
        line=dict(color=C_ACCENT2, width=1, dash="dash"), opacity=0.4), row=1, col=1)

    # Signal markers
    buy_rows  = tail[tail["signal"]==1]
    sell_rows = tail[tail["signal"]==0]
    if not buy_rows.empty:
        fig_main.add_trace(go.Scatter(
            x=buy_rows["Date"], y=buy_rows["Low"] * 0.9995,
            mode="markers", marker=dict(symbol="triangle-up", color=C_ACCENT, size=9),
            name="BUY signal"), row=1, col=1)
    if not sell_rows.empty:
        pass  # don't plot every non-signal bar — too noisy

    # MACD
    macd_colors = [C_ACCENT if v>=0 else C_RED for v in tail["macd_hist"]]
    fig_main.add_trace(go.Scatter(x=tail["Date"], y=tail["macd"],
        name="MACD", line=dict(color=C_ACCENT, width=1.5)), row=2, col=1)
    fig_main.add_trace(go.Scatter(x=tail["Date"], y=tail["macd_signal"],
        name="Signal", line=dict(color=C_YELLOW, width=1.5)), row=2, col=1)
    fig_main.add_trace(go.Bar(x=tail["Date"], y=tail["macd_hist"],
        marker_color=macd_colors, opacity=0.7, name="Histogram"), row=2, col=1)

    # RSI
    fig_main.add_hrect(y0=70, y1=100, fillcolor="rgba(244,63,94,0.08)", line_width=0, row=3, col=1)
    fig_main.add_hrect(y0=0, y1=30,   fillcolor="rgba(0,200,150,0.07)", line_width=0, row=3, col=1)
    fig_main.add_hline(y=70, line_dash="dot", line_color=C_RED,    line_width=1, row=3, col=1)
    fig_main.add_hline(y=50, line_dash="dot", line_color=C_MUTED,  line_width=1, row=3, col=1)
    fig_main.add_hline(y=30, line_dash="dot", line_color=C_ACCENT, line_width=1, row=3, col=1)
    rsi_line_col = [C_RED if v>=70 else (C_ACCENT if v<=30 else C_ACCENT2) for v in tail["rsi_14"]]
    fig_main.add_trace(go.Scatter(x=tail["Date"], y=tail["rsi_14"],
        mode="lines", line=dict(color=C_ACCENT2, width=1.5), name="RSI"), row=3, col=1)

    fig_main.update_layout(**PLOT, height=640, xaxis_rangeslider_visible=False, showlegend=True)
    fig_main.update_yaxes(row=3, col=1, range=[0,100])
    st.plotly_chart(fig_main, width='stretch')

    # ── Momentum + Volatility ─────────────────────────────────────────────
    st.markdown(section("Momentum & Volatility"), unsafe_allow_html=True)
    mc1, mc2 = st.columns(2)

    with mc1:
        fig_mom = go.Figure()
        fig_mom.add_trace(go.Scatter(x=tail["Date"], y=tail["momentum_5"]*100,
            name="5-bar mom", line=dict(color=C_ACCENT, width=1.5)))
        fig_mom.add_trace(go.Scatter(x=tail["Date"], y=tail["momentum_10"]*100,
            name="10-bar mom", line=dict(color=C_PURPLE, width=1.5)))
        fig_mom.add_hline(y=0, line_color=C_MUTED, line_width=1)
        pf(fig_mom, 260)
        fig_mom.update_layout(title="Momentum (%)", title_font=dict(color=C_MUTED, size=11),
                               yaxis_ticksuffix="%")
        st.plotly_chart(fig_mom, width='stretch')

    with mc2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(x=tail["Date"], y=tail["volatility_10"]*100,
            name="10-bar vol", line=dict(color=C_YELLOW, width=1.5),
            fill="tozeroy", fillcolor="rgba(245,158,11,0.09)"))
        fig_vol.add_trace(go.Scatter(x=tail["Date"], y=tail["volatility_20"]*100,
            name="20-bar vol", line=dict(color=C_RED, width=1.5, dash="dash")))
        pf(fig_vol, 260)
        fig_vol.update_layout(title="Volatility (%)", title_font=dict(color=C_MUTED, size=11),
                               yaxis_ticksuffix="%")
        st.plotly_chart(fig_vol, width='stretch')

    # ── Walk-forward bars ─────────────────────────────────────────────────
    if os.path.exists(WF_RESULTS_PATH):
        st.markdown(section("Walk-Forward Validation"), unsafe_allow_html=True)
        wf_df = pd.read_csv(WF_RESULTS_PATH)
        wc1, wc2 = st.columns(2)
        with wc1:
            acc_colors = [C_ACCENT if v>0.52 else (C_YELLOW if v>0.50 else C_RED) for v in wf_df["accuracy"]]
            fig_wa = go.Figure(go.Bar(x=wf_df["split_id"], y=wf_df["accuracy"],
                marker_color=acc_colors, name="Accuracy"))
            fig_wa.add_hline(y=0.5, line_dash="dot", line_color=C_MUTED)
            fig_wa.add_hline(y=wf_df["accuracy"].mean(), line_dash="dash",
                              line_color=C_ACCENT2,
                              annotation_text=f"mean {wf_df['accuracy'].mean():.3f}",
                              annotation_font_color=C_ACCENT2)
            pf(fig_wa, 260)
            fig_wa.update_layout(title="Accuracy per Split", showlegend=False,
                                  title_font=dict(color=C_MUTED,size=11))
            st.plotly_chart(fig_wa, width='stretch')
        with wc2:
            ret_colors = [C_ACCENT if v>0 else C_RED for v in wf_df["strategy_return"]]
            fig_wr = go.Figure(go.Bar(x=wf_df["split_id"], y=wf_df["strategy_return"]*100,
                marker_color=ret_colors, name="Return %"))
            fig_wr.add_hline(y=0, line_color=C_MUTED, line_width=1)
            pf(fig_wr, 260)
            fig_wr.update_layout(title="Strategy Return per Split (%)", showlegend=False,
                                  title_font=dict(color=C_MUTED,size=11),
                                  yaxis_ticksuffix="%")
            st.plotly_chart(fig_wr, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Health
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    hl, hr_col = st.columns([1,1], gap="large")

    with hl:
        st.markdown(section("Accuracy Metrics"), unsafe_allow_html=True)
        gap_val = (tr_acc - te_acc) if (tr_acc and te_acc) else None
        wf_cls  = "g" if (wf_acc and wf_acc>=0.54) else ("w" if (wf_acc and wf_acc>=0.51) else "b")
        prof_pct = (wf_prof/wf_tot*100) if (wf_prof and wf_tot) else 0
        rows = "".join([
            hr("Train accuracy (in-sample)",    f"{tr_acc:.4f}" if tr_acc else "N/A",
               "w" if tr_acc and tr_acc>0.80 else "g"),
            hr("Test accuracy (held-out 20%)",  f"{te_acc:.4f}" if te_acc else "N/A",
               "g" if te_acc and te_acc>0.52 else "b"),
            hr("Walk-forward mean accuracy",    f"{wf_acc:.4f}" if wf_acc else "N/A", wf_cls),
            hr("Overfitting gap (train−test)",  f"{gap_val:.4f}" if gap_val else "N/A",
               "g" if gap_val and gap_val<0.10 else ("w" if gap_val and gap_val<0.15 else "b")),
            hr("Profitable WF splits",          f"{wf_prof}/{wf_tot}  ({prof_pct:.0f}%)" if wf_prof else "N/A",
               "g" if prof_pct>50 else ("w" if prof_pct>40 else "b")),
        ])
        st.markdown(f'<div class="sc">{rows}</div>', unsafe_allow_html=True)

        # Status banner
        if gap_val and gap_val > 0.15:
            st.error("⚠️ High overfitting gap — consider retraining with tighter params.")
        elif wf_acc and wf_acc < 0.51:
            st.error("🔴 Walk-forward below 51% — no consistent edge detected.")
        elif wf_acc and wf_acc < 0.54:
            st.warning("🟡 Marginal edge. Monitor before live use.")
        else:
            st.success("✅ Model looks healthy for production use.")

        st.markdown(section("Model Configuration"), unsafe_allow_html=True)
        cfg = "".join([
            hr("Model version",         meta.get("model_version","?"), "g"),
            hr("Feature set",           meta.get("feature_set","?")),
            hr("Feature count",         str(meta.get("feature_count","?"))),
            hr("RF max_depth",          str(meta.get("rf_max_depth","?"))),
            hr("RF min_samples_leaf",   str(meta.get("rf_min_samples_leaf","?"))),
            hr("RF n_estimators",       str(meta.get("rf_n_estimators","?"))),
            hr("Training rows",         str(meta.get("rows_train","?"))),
            hr("Test rows",             str(meta.get("rows_test","?"))),
            hr("Signal threshold",      str(meta.get("signal_threshold",threshold))),
        ])
        st.markdown(f'<div class="sc">{cfg}</div>', unsafe_allow_html=True)

        with st.expander("Raw JSON metadata"):
            st.json(meta)

    with hr_col:
        # Accuracy bar comparison
        if tr_acc and te_acc and wf_acc:
            fig_acc = go.Figure(go.Bar(
                x=["Train\n(in-sample)","Test\n(held-out)","Walk-forward\n(OOS)"],
                y=[tr_acc, te_acc, wf_acc],
                marker_color=[C_YELLOW, C_ACCENT2, C_ACCENT],
                text=[f"{v:.3f}" for v in [tr_acc, te_acc, wf_acc]],
                textposition="outside",
                textfont=dict(color=C_TEXT, family="JetBrains Mono", size=12),
                width=0.5,
            ))
            fig_acc.add_hline(y=0.5, line_dash="dot", line_color=C_RED,
                               annotation_text="Random baseline",
                               annotation_font_color=C_RED)
            pf(fig_acc, 290)
            fig_acc.update_layout(showlegend=False, yaxis=dict(range=[0.4, max(tr_acc+0.05,0.9)]),
                                   title="Accuracy Comparison", title_font=dict(color=C_MUTED,size=11))
            st.plotly_chart(fig_acc, width='stretch')

        # Radar chart: model quality dimensions
        st.markdown(section("Model Quality Radar"), unsafe_allow_html=True)
        if tr_acc and te_acc and wf_acc:
            gap_score   = max(0, 1 - (gap_val or 0.3) / 0.3)
            prof_score  = prof_pct / 100 if prof_pct else 0.4
            te_score    = (te_acc - 0.4) / 0.2 if te_acc else 0.5
            wf_score    = (wf_acc - 0.4) / 0.2 if wf_acc else 0.5

            dims   = ["Generalisation\n(low gap)","Test Accuracy","WF Accuracy",
                      "Profitable\nSplits","Calibration"]
            scores = [round(gap_score,3), round(te_score,3), round(wf_score,3),
                      round(prof_score,3), round((te_acc+wf_acc)/2 - 0.4, 3)]
            scores_clamped = [max(0, min(1, s)) for s in scores]

            fig_rad = go.Figure(go.Scatterpolar(
                r=scores_clamped + [scores_clamped[0]],
                theta=dims + [dims[0]],
                fill="toself", fillcolor="rgba(0,200,150,0.13)",
                line=dict(color=C_ACCENT, width=2),
                name="Model",
            ))
            fig_rad.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                polar=dict(
                    bgcolor=C_SURF,
                    radialaxis=dict(range=[0,1], tickcolor=C_MUTED,
                                    gridcolor=C_BORDER, tickfont=dict(size=9,color=C_MUTED)),
                    angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(size=10,color=C_MUTED)),
                ),
                font=dict(family="JetBrains Mono", color=C_TEXT),
                height=300, margin=dict(l=50,r=50,t=30,b=30),
                showlegend=False,
            )
            st.plotly_chart(fig_rad, width='stretch')

        # WF accuracy distribution
        if os.path.exists(WF_RESULTS_PATH):
            wf_df = pd.read_csv(WF_RESULTS_PATH)
            fig_h = go.Figure()
            fig_h.add_trace(go.Histogram(x=wf_df["accuracy"], nbinsx=12,
                marker_color=C_ACCENT2, opacity=0.8, name="Accuracy dist."))
            fig_h.add_vline(x=0.5,  line_dash="dot", line_color=C_RED)
            fig_h.add_vline(x=wf_df["accuracy"].mean(), line_dash="dash",
                             line_color=C_ACCENT,
                             annotation_text=f"mean {wf_df['accuracy'].mean():.3f}",
                             annotation_font_color=C_ACCENT)
            pf(fig_h, 230)
            fig_h.update_layout(showlegend=False, title="WF Accuracy Distribution",
                                  title_font=dict(color=C_MUTED,size=11))
            st.plotly_chart(fig_h, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Paper Trading
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown(section("Oanda Practice Account"), unsafe_allow_html=True)

    # Try to connect to Oanda
    oanda_ok   = False
    oanda_data = {}
    oanda_err  = ""
    try:
        from src.oanda_client import OandaClient
        oanda_client = OandaClient()
        oanda_data   = oanda_client.get_account_summary()
        oanda_ok     = True
    except Exception as e:
        oanda_err = str(e)

    if not oanda_ok:
        st.error(f"Oanda not connected: {oanda_err}")
        st.markdown(f"""
        <div class="sc" style="line-height:2">
          <div style="color:{C_MUTED};font-size:11px;margin-bottom:8px;">HOW TO CONNECT OANDA</div>
          <div style="font-size:12px;color:{C_TEXT};">
            1. Sign up for a <b>free practice account</b> at
               <a href="https://www.oanda.com/register" style="color:{C_ACCENT};">oanda.com/register</a><br>
            2. Go to <b>My Account → Manage API Access → Generate Token</b><br>
            3. Copy your <b>Account ID</b> from the dashboard<br>
            4. Add to <code>.env</code>:<br>
          </div>
          <div class="cmd" style="margin-top:8px;">
            OANDA_API_KEY=your_token_here<br>
            OANDA_ACCOUNT_ID=101-001-XXXXXXX-001<br>
            OANDA_ENVIRONMENT=practice
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Account metrics
        bal = oanda_data["balance"]
        nav = oanda_data["nav"]
        upl = oanda_data["unrealized_pl"]
        upl_col = C_ACCENT if upl >= 0 else C_RED

        oc1, oc2, oc3, oc4 = st.columns(4)
        oc1.markdown(sc("Account Balance",  f"${bal:,.2f}", "pos"), unsafe_allow_html=True)
        oc2.markdown(sc("NAV",              f"${nav:,.2f}", "pos" if nav >= bal else "neg"), unsafe_allow_html=True)
        oc3.markdown(sc("Unrealized P&L",   f"${upl:+,.2f}", "pos" if upl>=0 else "neg"), unsafe_allow_html=True)
        oc4.markdown(sc("Open Trades",      str(oanda_data["open_trades"]), "neu"), unsafe_allow_html=True)

        pt_l, pt_r = st.columns(2, gap="large")

        with pt_l:
            # Open trades
            st.markdown(section("Open Trades"), unsafe_allow_html=True)
            try:
                open_trades = oanda_client.get_open_trades()
                if open_trades:
                    ot_df = pd.DataFrame(open_trades)
                    st.dataframe(ot_df, width='stretch', hide_index=True)
                else:
                    st.info("No open trades.")
            except Exception as e:
                st.error(str(e))

            # Live price
            st.markdown(section("Live EUR/USD Price"), unsafe_allow_html=True)
            try:
                lp = oanda_client.get_live_price("EUR_USD")
                price_rows = "".join([
                    hr("Bid",    f"{lp['bid']:.5f}"),
                    hr("Ask",    f"{lp['ask']:.5f}"),
                    hr("Mid",    f"{lp['mid']:.5f}", "g"),
                    hr("Spread", f"{lp['spread']:.5f}"),
                    hr("Time",   str(lp['timestamp'])[:19]),
                ])
                st.markdown(f'<div class="sc">{price_rows}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

            # Manual signal check button
            st.markdown(section("Manual Signal Check"), unsafe_allow_html=True)
            if st.button("▶ Run Signal Check Now", use_container_width=False):
                with st.spinner("Running full pipeline..."):
                    try:
                        from src.paper_trader import PaperTrader
                        pt = PaperTrader(threshold=threshold, use_regime_filter=True)
                        result = pt.run_signal_check()
                        if result["action"] == "order_placed":
                            st.success(f"✅ Trade placed! Signal: {result['signal']}  Price: {result['price']}")
                        elif result["action"] == "error":
                            st.error(f"Error: {result['reason']}")
                        else:
                            st.info(f"Signal: {result.get('signal','—')}  |  Not traded: {result['reason']}")
                        st.json(result)
                    except Exception as e:
                        st.error(str(e))

        with pt_r:
            # Paper trade log
            st.markdown(section("Paper Trade Log"), unsafe_allow_html=True)
            try:
                from src.paper_trader import PaperTrader
                pt_reader = PaperTrader.__new__(PaperTrader)
                pt_reader.instrument = "EUR_USD"
                import json, os
                from config.settings import PAPER_TRADES_PATH, SIGNALS_LOG_PATH
                if os.path.exists(PAPER_TRADES_PATH):
                    with open(PAPER_TRADES_PATH) as f:
                        trades_list = json.load(f)
                    if trades_list:
                        ptdf = pd.DataFrame(trades_list)
                        st.dataframe(ptdf, width='stretch', hide_index=True, height=220)
                    else:
                        st.info("No paper trades logged yet.")
                else:
                    st.info("No paper trades yet. Run a signal check to start.")
            except Exception as e:
                st.error(str(e))

            # Signal log
            st.markdown(section("Signal Log (Last 20)"), unsafe_allow_html=True)
            try:
                from config.settings import SIGNALS_LOG_PATH
                if os.path.exists(SIGNALS_LOG_PATH):
                    sig_df = pd.read_csv(SIGNALS_LOG_PATH).tail(20)
                    st.dataframe(sig_df, width='stretch', hide_index=True, height=220)
                else:
                    st.info("No signals logged yet.")
            except Exception as e:
                st.error(str(e))

            # Regime status
            st.markdown(section("Current Market Regime"), unsafe_allow_html=True)
            try:
                from src.regime_detector import RegimeDetector
                rd     = RegimeDetector()
                regime = rd.detect(df)
                adx_cls = "g" if regime["adx_regime"]=="trending" else ("w" if regime["adx_regime"]=="transitioning" else "b")
                vol_cls = "b" if regime["vol_regime"]=="high" else ("w" if regime["vol_regime"]=="low" else "g")
                trad    = regime["tradeable_signals"]
                trad_cls= "g" if trad else "b"
                regime_rows = "".join([
                    hr("ADX",             f"{regime['adx']:.1f}"),
                    hr("ADX Regime",      regime["adx_regime"].upper(), adx_cls),
                    hr("Trend Direction", regime["trend_direction"].upper(),
                       "g" if regime["trend_direction"]=="bullish" else ("b" if regime["trend_direction"]=="bearish" else "w")),
                    hr("Volatility",      f"{regime['vol_regime'].upper()} ({regime['vol_ratio']}x avg)", vol_cls),
                    hr("Tradeable Signals", str(trad) if trad else "NONE", trad_cls),
                ])
                st.markdown(f'<div class="sc">{regime_rows}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

    # Telegram test
    st.markdown(section("Telegram Alerts"), unsafe_allow_html=True)
    tl, tr = st.columns([2,1])
    with tl:
        from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
        tg_ok = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE")
        tg_rows = "".join([
            hr("Bot token",  "✓ Set" if tg_ok else "Not set", "g" if tg_ok else "b"),
            hr("Chat ID",    "✓ Set" if (TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE") else "Not set",
               "g" if (TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE") else "b"),
            hr("Status",     "Connected" if tg_ok else "Not configured", "g" if tg_ok else "b"),
        ])
        st.markdown(f'<div class="sc">{tg_rows}</div>', unsafe_allow_html=True)
    with tr:
        if st.button("📨 Send Test Message"):
            from src.alerter import Alerter
            ok = Alerter().test()
            if ok:
                st.success("✓ Message sent!")
            else:
                st.error("Failed. Check .env credentials.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio
# ══════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown(section("Multi-Pair Portfolio Signals"), unsafe_allow_html=True)

    port_col1, port_col2 = st.columns([3, 1], gap="large")

    with port_col2:
        if st.button("🔄 Refresh Signals", use_container_width=False):
            st.rerun()
        st.markdown(f'<div style="font-size:11px;color:{C_MUTED};margin-top:8px;">Signals ranked by confidence</div>', unsafe_allow_html=True)

    with port_col1:
        try:
            from src.multi_pair_manager import get_portfolio_signals, ACTIVE_PAIRS
            from config.settings import PAIRS

            with st.spinner("Loading portfolio signals..."):
                port_df = get_portfolio_signals(threshold=threshold)

            if port_df.empty:
                st.warning("No signals available. Run train_all.py first.")
            else:
                # Signal cards for each pair
                cols = st.columns(len(ACTIVE_PAIRS))
                for idx, (_, row) in enumerate(port_df.iterrows()):
                    if idx >= len(cols):
                        break
                    with cols[idx]:
                        pair_display = row["pair"].replace("_", "/")
                        is_buy_p = row["signal"] == "BUY"
                        sig_c = C_ACCENT if is_buy_p else C_RED
                        arrow = "▲" if is_buy_p else "▼"
                        trade_badge = "TRADE" if row.get("tradeable") else "SKIP"
                        trade_col   = C_ACCENT if row.get("tradeable") else C_MUTED
                        st.markdown(f"""
                        <div class="sc" style="border-left:3px solid {sig_c};padding:12px 14px;">
                          <div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;">{pair_display}</div>
                          <div style="font-family:'Syne',sans-serif;font-size:26px;font-weight:800;color:{sig_c};">{arrow} {row["signal"]}</div>
                          <div style="font-size:11px;color:{C_MUTED};margin-top:4px;">
                            Prob: <b style="color:{C_TEXT}">{row["prob_up"]:.3f}</b><br>
                            Conf: <b style="color:{C_TEXT}">{row["confidence"]}</b><br>
                            Regime: <b style="color:{C_TEXT}">{row.get("regime","?").upper()}</b><br>
                            <span style="color:{trade_col};font-weight:700;">{trade_badge}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Portfolio signals unavailable: {e}")
            st.info("Make sure you have run: python train_all.py --fetch")

    # Portfolio summary table
    st.markdown(section("Portfolio Overview Table"), unsafe_allow_html=True)
    try:
        from src.multi_pair_manager import get_portfolio_signals
        port_df2 = get_portfolio_signals(threshold=threshold)
        if not port_df2.empty:
            display_cols = ["pair","signal","prob_up","confidence","regime",
                            "trend","vol_regime","tradeable","wf_accuracy","test_accuracy","latest_date"]
            show_cols = [c for c in display_cols if c in port_df2.columns]
            st.dataframe(port_df2[show_cols], width='stretch', hide_index=True)
    except Exception as e:
        st.error(str(e))

    # Model health per pair
    st.markdown(section("Per-Pair Model Status"), unsafe_allow_html=True)
    try:
        from config.settings import meta_path, wf_path, data_path, ACTIVE_PAIRS
        import json, os
        pair_rows = []
        for p in ACTIVE_PAIRS:
            mp = meta_path(p)
            if os.path.exists(mp):
                with open(mp) as f:
                    m = json.load(f)
                pair_rows.append({
                    "Pair":         p.replace("_","/"),
                    "Train Acc":    f"{m.get('accuracy_train',0):.3f}",
                    "Test Acc":     f"{m.get('accuracy_test',0):.3f}",
                    "WF Acc":       f"{m.get('walk_forward_mean_accuracy',0):.3f}",
                    "WF Profit":    f"{m.get('walk_forward_profitable_splits',0)}/{m.get('walk_forward_total_splits',0)}",
                    "Trained At":   m.get("trained_at","?")[:10],
                    "Rows":         m.get("rows_total","?"),
                })
            else:
                pair_rows.append({
                    "Pair": p.replace("_","/"), "Train Acc":"—","Test Acc":"—",
                    "WF Acc":"—","WF Profit":"—","Trained At":"not trained","Rows":"—",
                })
        if pair_rows:
            st.dataframe(pd.DataFrame(pair_rows), width='stretch', hide_index=True)
    except Exception as e:
        st.error(str(e))

    # Train all pairs button
    st.markdown(section("Train / Update Models"), unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        if st.button("⬇ Fetch All Pair Data"):
            with st.spinner("Fetching data for all pairs..."):
                res, err = api_post("/fetch-data", params={"outputsize":"compact"})
            # Also try multi-pair fetch via API if available
            try:
                from src.multi_pair_manager import fetch_all_pairs
                fetch_results = fetch_all_pairs(count=100)
                lines = [f"{'✅' if r['ok'] else '❌'} {p}: {r.get('rows',r.get('error','?'))}" for p,r in fetch_results.items()]
                st.success("\n".join(lines))
            except Exception as e:
                st.error(str(e))
    with tc2:
        if st.button("⚙ Train All Pairs"):
            with st.spinner("Training all pair models (~2 min)..."):
                try:
                    import subprocess, sys
                    result = subprocess.run(
                        [sys.executable, "train_all.py"],
                        capture_output=True, text=True, timeout=300,
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    if result.returncode == 0:
                        st.success("✅ All models trained!")
                        st.code(result.stdout)
                    else:
                        st.error(f"Training failed:\n{result.stderr}")
                except Exception as e:
                    st.error(str(e))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — System
# ══════════════════════════════════════════════════════════════════════════════
with t6:
    sc1, sc2 = st.columns(2, gap="large")

    with sc1:
        st.markdown(section("Backend API Status"), unsafe_allow_html=True)
        if h_err:
            st.error(f"API offline — start with:\n```\nuvicorn app.api:app --reload\n```\n{h_err}")
        else:
            api_rows = "".join([
                hr("Status",            health_data.get("status","?").upper(),
                   "g" if health_data.get("status")=="ok" else "b"),
                hr("Response time",     f"{h_ms} ms",
                   "g" if h_ms<200 else ("w" if h_ms<500 else "b")),
                hr("Model loaded",      "YES" if health_data.get("model_loaded") else "NO",
                   "g" if health_data.get("model_loaded") else "b"),
                hr("Data loaded",       "YES" if health_data.get("data_loaded") else "NO",
                   "g" if health_data.get("data_loaded") else "b"),
                hr("Data rows",         str(health_data.get("data_rows","?"))),
                hr("Latest bar",        str(health_data.get("latest_date","?"))),
                hr("Model version",     str(health_data.get("model_version","?")), "g"),
                hr("WF accuracy",       f"{health_data.get('walk_forward_accuracy','?')}"),
            ])
            st.markdown(f'<div class="sc">{api_rows}</div>', unsafe_allow_html=True)

            # Endpoint latency tester
            st.markdown(section("Endpoint Latency"), unsafe_allow_html=True)
            endpoints_to_test = ["/health", "/predict/latest", "/walk-forward", f"/history?n=50"]
            lat_data = []
            for ep in endpoints_to_test:
                _, _, ms = api_get(ep)
                lat_data.append({"Endpoint": ep, "Latency (ms)": ms,
                                  "Status": "🟢" if ms < 300 else "🟡"})
            lat_df = pd.DataFrame(lat_data)
            st.dataframe(lat_df, width='stretch', hide_index=True)

            # Live API prediction
            pred_data, p_err, _ = api_get("/predict/latest", params={"threshold": threshold})
            if pred_data:
                st.markdown(section("Live API Prediction"), unsafe_allow_html=True)
                pred_rows = "".join([
                    hr("Signal",          pred_data.get("signal","?"),
                       "g" if "UP" in str(pred_data.get("signal","")) else "b"),
                    hr("Probability UP",  f"{pred_data.get('probability_up','?')}"),
                    hr("Probability DOWN",f"{pred_data.get('probability_down','?')}"),
                    hr("Confidence",      pred_data.get("confidence","?").upper(),
                       "g" if pred_data.get("confidence")=="high"
                       else ("w" if pred_data.get("confidence")=="medium" else "b")),
                    hr("Latest close",    str(pred_data.get("latest_close","?"))),
                    hr("Threshold used",  str(pred_data.get("threshold_used","?"))),
                ])
                st.markdown(f'<div class="sc">{pred_rows}</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown(section("Data File Info"), unsafe_allow_html=True)
        data_rows = "".join([
            hr("File path",   DATA_PATH),
            hr("Total rows",  str(len(df_raw)), "g"),
            hr("Date range",  f"{df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}"),
            hr("Data age",    f"{days_old} days",
               "g" if days_old<=3 else ("w" if days_old<=7 else "b")),
            hr("Latest close",f"{lat_close:.5f}"),
            hr("Latest high", f"{float(df_raw['High'].iloc[-1]):.5f}"),
            hr("Latest low",  f"{float(df_raw['Low'].iloc[-1]):.5f}"),
            hr("Feature rows",str(len(df)), "g"),
        ])
        st.markdown(f'<div class="sc">{data_rows}</div>', unsafe_allow_html=True)

        # Data quality checker
        st.markdown(section("Data Quality"), unsafe_allow_html=True)
        null_count  = df_raw.isnull().sum().sum()
        dup_count   = df_raw.duplicated(subset=["Date"]).sum()
        neg_count   = (df_raw[["Open","High","Low","Close"]] <= 0).sum().sum()
        hl_violations = (df_raw["High"] < df_raw["Low"]).sum()
        dq_rows = "".join([
            hr("Null values",       str(null_count),      "g" if null_count==0 else "b"),
            hr("Duplicate dates",   str(dup_count),       "g" if dup_count==0 else "b"),
            hr("Negative prices",   str(neg_count),       "g" if neg_count==0 else "b"),
            hr("High < Low rows",   str(hl_violations),   "g" if hl_violations==0 else "b"),
        ])
        st.markdown(f'<div class="sc">{dq_rows}</div>', unsafe_allow_html=True)

        # Run commands
        st.markdown(section("Run Commands"), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="sc" style="line-height:2">
          <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin-bottom:6px;">TERMINAL 1 — API</div>
          <div class="cmd">uvicorn app.api:app --reload --host 127.0.0.1 --port 8000</div>
          <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin:10px 0 6px;">TERMINAL 2 — DASHBOARD</div>
          <div class="cmd">streamlit run app/dashboard.py</div>
          <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin:10px 0 6px;">RETRAIN</div>
          <div class="cmd">python train.py</div>
          <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin:10px 0 6px;">TESTS</div>
          <div class="cmd">python tests/test_all.py</div>
        </div>
        """, unsafe_allow_html=True)

        # API endpoints reference
        st.markdown(section("API Endpoints"), unsafe_allow_html=True)
        eps = [("GET","/health","Liveness check"),("GET","/predict/latest","Latest signal"),
               ("GET","/model-info","Model metadata"),("POST","/fetch-data","Fetch & merge"),
               ("POST","/retrain","Retrain model"),("GET","/walk-forward","WF results"),
               ("GET","/history","Last N bars")]
        ep_rows = "".join([
            f'<div class="hr"><span style="color:{C_YELLOW};flex:0 0 46px;font-size:10px;">{m}</span>'
            f'<span style="color:{C_ACCENT};flex:0 0 170px;font-size:11px;">{p}</span>'
            f'<span style="color:{C_MUTED};font-size:11px;">{d}</span></div>'
            for m,p,d in eps
        ])
        st.markdown(f'<div class="sc">{ep_rows}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Portfolio (injected after file write, before System tab at runtime)
# ══════════════════════════════════════════════════════════════════════════════
