"""
ForexChautari — User Dashboard v2
Every backend feature exposed with proper plan-gating.

Tabs:
  📡 Signals      — all pairs, live prices, regime, signal history (all plans)
  📊 Analysis     — charts, indicators, walk-forward, model health per pair (all plans)
  ⚡ Trading      — account, place/close trades, positions, history (pro/enterprise)
  📂 My Trades    — trade log, P&L chart, stats (pro/enterprise)
  🔔 Alerts       — notification centre, telegram test (all plans)
  👤 Account      — profile, password, subscription, linked accounts

Task 3.2 changes (signal confidence calibration):
  - Model Health section now shows Brier score KPI row
    (brier_score_test, brier_score_wf_mean, calibration method)
  - Reliability diagram (calibration curve) rendered via Plotly from
    the held-out 20% test split when is_calibrated=True.
  - Prompt to retrain shown for pre-calibration model files.
"""

import os, sys, json, time, requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.auth import render_logout_button
from app.api_client import api_get, api_post, api_delete, ApiError
from src.database import (
    get_trading_accounts, add_trading_account, remove_trading_account,
    get_user_trades, get_trade_stats, log_signal, get_signals_log,
    update_user_profile, update_user_password, verify_password,
    get_notifications, mark_notifications_read, get_db,
    get_user_trading_settings, update_user_trading_settings,
)
from config.settings import (
    ACTIVE_PAIRS, PAIRS, DEFAULT_SIGNAL_THRESHOLD,
    meta_path, data_path, wf_path, APP_BRAND, APP_NAME, PLAN_LIMITS,
    TRADABLE_MIN_WF_ACCURACY, TRADABLE_MIN_NET_PROFIT_FACTOR,
    TRADABLE_MIN_PROFITABLE_SPLITS_PCT,
)

API_BASE = "http://127.0.0.1:8000"

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#070b13"
C_SURF   = "#0d1520"
C_SURF2  = "#111e2d"
C_CARD   = "#0f1a28"
C_BORDER = "#1a2d42"
C_ACCENT = "#00d4a8"
C_ACCENT2= "#0891b2"
C_RED    = "#f43f5e"
C_YELLOW = "#eab308"
C_GREEN  = "#22c55e"
C_ORANGE = "#f97316"
C_PURPLE = "#8b5cf6"
C_TEXT   = "#e2eaf4"
C_MUTED  = "#4a6580"
C_DIM    = "#1e3348"

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#08111c",
    font=dict(family="'IBM Plex Mono', monospace", color=C_TEXT, size=11),
    xaxis=dict(gridcolor=C_BORDER, showgrid=True, zeroline=False,
               showspikes=True, spikecolor=C_MUTED, spikethickness=1),
    yaxis=dict(gridcolor=C_BORDER, showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
    margin=dict(l=8, r=8, t=36, b=8),
    hovermode="x unified",
    hoverlabel=dict(bgcolor=C_SURF2, bordercolor=C_BORDER,
                    font=dict(color=C_TEXT, size=11)),
)

def pf(fig, h=340):
    fig.update_layout(**PLOT, height=h)
    return fig

# ── Plan config ───────────────────────────────────────────────────────────────
PLAN_UI = {
    "free":       {"pairs":1, "auto_trade":False, "charts":True,  "wf":False, "alerts":False, "color":C_MUTED,  "label":"Free"},
    "basic":      {"pairs":2, "auto_trade":False, "charts":True,  "wf":True,  "alerts":False, "color":C_ACCENT2,"label":"Basic"},
    "pro":        {"pairs":4, "auto_trade":True,  "charts":True,  "wf":True,  "alerts":True,  "color":C_YELLOW, "label":"Pro"},
    "enterprise": {"pairs":99,"auto_trade":True,  "charts":True,  "wf":True,  "alerts":True,  "color":C_ACCENT, "label":"Enterprise"},
}

def _plan_color(plan): return PLAN_UI.get(plan, PLAN_UI["free"])["color"]

def _badge(text, color=None):
    c = color or C_ACCENT
    return f'<span style="display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;letter-spacing:1.2px;background:{c}18;color:{c};border:1px solid {c}33;">{text}</span>'

def _locked(name, plan):
    st.markdown(f"""
    <div style="background:{C_SURF};border:1px dashed {C_BORDER};border-radius:12px;
         padding:28px;text-align:center;margin:8px 0;">
      <div style="font-size:32px;margin-bottom:10px;">🔒</div>
      <div style="font-size:15px;font-weight:600;color:{C_TEXT};margin-bottom:6px;">{name}</div>
      <div style="font-size:12px;color:{C_MUTED};">
        Available on <b style="color:{C_YELLOW};">{plan.title()}</b> plan and above.<br>
        Upgrade in the <b>Account</b> tab.
      </div>
    </div>""", unsafe_allow_html=True)

def _kpi(label, value, cls="", sub=""):
    sc = {"g":C_GREEN,"r":C_RED,"y":C_YELLOW,"a":C_ACCENT,"m":C_MUTED}.get(cls, C_TEXT)
    sub_html = f'<div style="font-size:10px;color:{C_MUTED};margin-top:3px;">{sub}</div>' if sub else ""
    return f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:12px;padding:14px 16px;margin-bottom:10px;">
      <div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;text-transform:uppercase;margin-bottom:5px;">{label}</div>
      <div style="font-size:22px;font-weight:700;color:{sc};font-family:'IBM Plex Mono',monospace;">{value}</div>
      {sub_html}</div>"""

def _section(title):
    st.markdown(f'<div style="font-size:10px;color:{C_MUTED};letter-spacing:3px;text-transform:uppercase;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid {C_BORDER};">{title}</div>', unsafe_allow_html=True)

def _card(content, border_color=None):
    bc = border_color or C_BORDER
    st.markdown(f'<div style="background:{C_CARD};border:1px solid {bc};border-radius:12px;padding:16px 18px;margin-bottom:12px;">{content}</div>', unsafe_allow_html=True)

def _plan_card(name: str, info: dict, current: bool):
    color = info["color"]
    border = color if current else C_BORDER
    current_html = (
        f'<span style="font-size:10px;color:{color};letter-spacing:1px;margin-left:8px;">CURRENT</span>'
        if current else ""
    )
    features = "".join([
        f'<span style="display:inline-block;margin:3px 8px 3px 0;color:{C_TEXT};font-size:11px;">{feat}</span>'
        for feat in info["features"]
    ])
    shadow = f"box-shadow:0 0 12px {color}22;" if current else ""
    st.markdown(f"""
    <div style="background:{C_CARD};border:1px solid {border};border-radius:10px;
         padding:16px 18px;margin:8px 0;{shadow}">
      <div style="display:flex;justify-content:space-between;gap:16px;align-items:center;margin-bottom:8px;">
        <div>
          <span style="font-size:14px;font-weight:700;color:{color};letter-spacing:.5px;">{name.upper()}</span>
          {current_html}
        </div>
        <span style="font-size:13px;color:{C_MUTED};font-family:'IBM Plex Mono',monospace;">{info["price"]}</span>
      </div>
      <div style="font-size:11px;color:{C_MUTED};margin-bottom:8px;">{info["desc"]}</div>
      <div>{features}</div>
    </div>
    """, unsafe_allow_html=True)

def _css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Syne+Mono&family=Space+Grotesk:wght@500;600;700&display=swap');
    html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;background:{C_BG};color:{C_TEXT};}}
    section[data-testid="stSidebar"]{{background:{C_SURF}!important;border-right:1px solid {C_BORDER};}}
    section[data-testid="stSidebar"] *{{color:{C_TEXT}!important;}}
    button[data-baseweb="tab"]{{font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;letter-spacing:1.5px!important;color:{C_MUTED}!important;text-transform:uppercase!important;padding:12px 18px!important;}}
    button[data-baseweb="tab"][aria-selected="true"]{{color:{C_ACCENT}!important;border-bottom:2px solid {C_ACCENT}!important;}}
    div[data-baseweb="tab-list"]{{background:transparent!important;border-bottom:1px solid {C_BORDER}!important;gap:0!important;}}
    .stButton>button{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT}!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;font-weight:500!important;letter-spacing:.5px!important;transition:all .15s!important;}}
    .stButton>button:hover{{border-color:{C_ACCENT}!important;color:{C_ACCENT}!important;box-shadow:0 0 12px {C_ACCENT}22!important;}}
    .stButton>button[kind="primary"]{{background:{C_ACCENT}!important;color:#000!important;border-color:{C_ACCENT}!important;font-weight:700!important;}}
    .stButton>button[kind="primary"]:hover{{opacity:.88!important;}}
    .stTextInput input,.stNumberInput input{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT}!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;font-size:13px!important;}}
    .stTextInput input:focus{{border-color:{C_ACCENT}!important;box-shadow:0 0 0 2px {C_ACCENT}22!important;}}
    .stSelectbox>div>div{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;border-radius:8px!important;}}
    .stSlider [data-testid="stThumb"]{{background:{C_ACCENT}!important;}}
    .stSlider [data-testid="stTrackActive"]{{background:{C_ACCENT}!important;}}
    div[data-testid="stDataFrame"]{{border:1px solid {C_BORDER}!important;border-radius:10px!important;overflow:hidden;}}
    .stAlert{{border-radius:10px!important;font-size:12px!important;}}
    #MainMenu,footer,header{{visibility:hidden;}}
    .block-container{{padding-top:0.6rem!important;max-width:1480px;}}
    </style>""", unsafe_allow_html=True)


# ── Oanda helper for user ──────────────────────────────────────────────────────

def _get_user_client(user_id: int, account_idx: int = 0):
    """Build OandaClient from user's stored credentials."""
    from src.trading_engine import get_client_for_user
    return get_client_for_user(user_id, account_idx)


# ── Reliability diagram helper ─────────────────────────────────────────────────

def _render_reliability_diagram(df, meta, sel_pair):
    """
    Render a Plotly reliability diagram (calibration curve) for sel_pair.

    Loads the saved model, runs predict_proba() on the held-out 20% test
    split, then plots fraction-of-positives vs mean-predicted-probability.

    Called only when meta["is_calibrated"] is True.
    """
    _section("Calibration · Reliability Diagram")
    try:
        import joblib as _jl
        from sklearn.calibration import calibration_curve as _cal_curve
        from sklearn.metrics import brier_score_loss as _brier_fn

        _mp_pkl = f"models/{sel_pair}_model.pkl"
        if not os.path.exists(_mp_pkl):
            st.info(f"Model file not found at `{_mp_pkl}`. Run: `python train_all.py --fetch`")
            return

        _model   = _jl.load(_mp_pkl)
        _sp_idx  = int(len(df) * 0.8)
        _te_df   = df.iloc[_sp_idx:]
        _feat    = meta.get("feature_columns", [])

        if len(_te_df) < 40 or not _feat:
            st.info("Not enough held-out rows to render a reliability diagram (need ≥ 40).")
            return

        _probas  = _model.predict_proba(_te_df[_feat])[:, 1]
        _y_true  = _te_df["target"]
        _pt, _pp = _cal_curve(_y_true, _probas, n_bins=8, strategy="quantile")
        _bs      = float(_brier_fn(_y_true, _probas))

        fig_rel = go.Figure()

        # Perfect-calibration diagonal
        fig_rel.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Perfect calibration",
            line=dict(color=C_MUTED, dash="dot", width=1.5),
            hoverinfo="skip",
        ))

        # Model calibration curve
        fig_rel.add_trace(go.Scatter(
            x=_pp.tolist(),
            y=_pt.tolist(),
            name=f"{sel_pair.replace('_', '/')}  ·  Brier = {_bs:.4f}",
            mode="lines+markers",
            line=dict(color=C_ACCENT, width=2.5),
            marker=dict(
                size=10,
                color=C_ACCENT,
                line=dict(color=C_SURF2, width=1.5),
                symbol="circle",
            ),
        ))

        pf(fig_rel, 320)
        fig_rel.update_layout(
            title="Reliability Diagram — fraction of positives vs mean predicted probability",
            title_font=dict(color=C_MUTED, size=11),
            xaxis=dict(
                title="Mean Predicted Probability",
                range=[-0.05, 1.05],
                tickformat=".0%",
                dtick=0.1,
            ),
            yaxis=dict(
                title="Fraction of Positives",
                range=[-0.05, 1.05],
                tickformat=".0%",
                dtick=0.1,
            ),
            showlegend=True,
        )
        st.plotly_chart(fig_rel, use_container_width=True)

        wf_acc = meta.get("walk_forward_mean_accuracy", 0.52) or 0.52
        st.caption(
            f"**How to read this chart:** A perfectly calibrated model follows the dotted diagonal — "
            f"when it predicts 60% probability, the outcome is positive 60% of the time. "
            f"Raw Random Forests cluster near the extremes (over-confident). "
            f"Platt scaling compresses probabilities toward 0.5, which is correct for a model "
            f"with ≈{wf_acc * 100:.0f}% walk-forward accuracy. "
            f"Fewer HIGH-confidence signals after calibration is **expected behaviour, not a regression**."
        )

    except Exception as _rel_err:
        st.info(f"Reliability diagram unavailable: {_rel_err}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN RENDER
# ══════════════════════════════════════════════════════════════════════════════

def render_user_dashboard(user: dict):
    _css()

    plan     = user.get("plan", "free")
    pui      = PLAN_UI.get(plan, PLAN_UI["free"])
    max_pairs= min(pui["pairs"], len(ACTIVE_PAIRS))
    can_trade= pui["auto_trade"]
    pairs    = ACTIVE_PAIRS[:max_pairs]

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        notifs = get_notifications(user["id"], unread_only=True)
        unread = len(notifs)

        st.markdown(f"""
        <div style="padding:14px 0 10px;">
          <div style="font-family:'Syne Mono',monospace;font-size:15px;font-weight:700;color:{C_ACCENT};letter-spacing:-0.5px;">⬡ {APP_BRAND}</div>
          <div style="font-size:10px;color:{C_MUTED};margin-top:3px;letter-spacing:1px;">FOREX MARKET PREDICTION & ANALYSIS</div>
          <div style="margin-top:10px;">{_badge(pui["label"].upper(), pui["color"])}</div>
          <div style="margin-top:8px;">
            <div style="font-size:13px;font-weight:600;color:{C_TEXT};">{user.get("full_name") or user["username"]}</div>
            <div style="font-size:11px;color:{C_MUTED};">{user.get("email","")}</div>
          </div>
          {"" if not unread else f'<div style="margin-top:8px;font-size:12px;color:{C_YELLOW};">🔔 {unread} new notification{"s" if unread>1 else ""}</div>'}
        </div>
        <hr style="border-color:{C_BORDER};margin:10px 0;">
        """, unsafe_allow_html=True)

        threshold = st.slider("Signal Threshold", 0.50, 0.90, 0.60, 0.01,
                              help="Minimum model confidence to show/act on a signal")
        sl_pips   = st.slider("Stop Loss (pips)", 5, 100, 20, 5)
        tp_pips   = st.slider("Take Profit (pips)", 5, 200, 40, 5)
        risk_pct  = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.5) / 100

        st.markdown(f'<hr style="border-color:{C_BORDER};margin:10px 0;">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;margin-bottom:8px;">YOUR PAIRS</div>', unsafe_allow_html=True)
        for p in pairs:
            st.markdown(f'<div style="font-size:12px;color:{C_ACCENT};padding:4px 0;font-family:IBM Plex Mono,monospace;">⬡ {p.replace("_","/")}</div>', unsafe_allow_html=True)
        if max_pairs < len(ACTIVE_PAIRS):
            locked_pairs = ACTIVE_PAIRS[max_pairs:]
            for p in locked_pairs:
                st.markdown(f'<div style="font-size:12px;color:{C_DIM};padding:4px 0;font-family:IBM Plex Mono,monospace;">🔒 {p.replace("_","/")}</div>', unsafe_allow_html=True)

        st.markdown(f'<hr style="border-color:{C_BORDER};margin:10px 0;">', unsafe_allow_html=True)
        render_logout_button()

    # ── Header ─────────────────────────────────────────────────────────────────
    fname = (user.get("full_name") or user["username"]).split()[0]
    now   = datetime.now().strftime("%Y-%m-%d  %H:%M")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C_SURF} 0%,{C_SURF2} 100%);
         border:1px solid {C_BORDER};border-radius:14px;padding:18px 26px;
         display:flex;align-items:center;justify-content:space-between;margin-bottom:18px;">
      <div>
        <div style="font-family:'Syne Mono',monospace;font-size:20px;color:{C_ACCENT};font-weight:700;">
          ⬡ {APP_BRAND}
        </div>
        <div style="font-size:12px;color:{C_MUTED};margin-top:3px;">
          Welcome back, {fname} · {APP_NAME}
        </div>
      </div>
      <div style="text-align:right;">
        <div style="font-size:12px;color:{C_MUTED};font-family:IBM Plex Mono,monospace;">{now}</div>
        <div style="margin-top:6px;">{_badge(pui["label"].upper(), pui["color"])}</div>
      </div>
    </div>""", unsafe_allow_html=True)
    _, logout_col = st.columns([8, 1])
    with logout_col:
        render_logout_button(sidebar=False, key="user_logout_top")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "  📡  SIGNALS  ",
        "  📊  ANALYSIS  ",
        "  ⚡  TRADING  ",
        "  📂  MY TRADES  ",
        "  🔔  ALERTS  ",
        "  👤  ACCOUNT  ",
    ])
    t_signals, t_analysis, t_trading, t_trades, t_alerts, t_account = tabs

    # ══ SIGNALS TAB ═══════════════════════════════════════════════════════════
    with t_signals:
        st.caption(
            "⚠️ ML signals are for informational purposes only and are **not financial advice**. "
            "Past model performance does not guarantee future returns."
        )
        try:
            from src.multi_pair_manager import get_portfolio_signals
            with st.spinner("Loading signals..."):
                port_df = get_portfolio_signals(threshold)

            ok_df = port_df[port_df["ok"]==True] if not port_df.empty else pd.DataFrame()
            user_ok_df = ok_df[ok_df["pair"].isin(pairs)] if not ok_df.empty else pd.DataFrame()

            # ── Live prices row ────────────────────────────────────────────────
            _section("Live Market Prices")
            accounts = get_trading_accounts(user["id"])
            if accounts:
                try:
                    client = _get_user_client(user["id"])
                    prices = client.get_all_prices(ACTIVE_PAIRS)
                    pc = st.columns(len(prices))
                    for i, p in enumerate(prices):
                        locked = p["instrument"] not in pairs
                        dim    = C_MUTED if locked else C_TEXT
                        tc     = C_ACCENT if p.get("tradeable") else C_RED
                        with pc[i]:
                            st.markdown(f"""
                            <div style="background:{C_CARD};border:1px solid {'#1a2d42' if locked else C_BORDER};
                                 border-radius:10px;padding:12px 14px;{'opacity:.45;' if locked else ''}">
                              <div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;margin-bottom:4px;">
                                {p["instrument"].replace("_","/")} {"🔒" if locked else ""}
                              </div>
                              <div style="font-size:20px;font-weight:700;color:{dim};font-family:IBM Plex Mono,monospace;">
                                {p["mid"]:.5f}
                              </div>
                              <div style="font-size:11px;color:{C_MUTED};margin-top:4px;">
                                Spread: <span style="color:{tc};">{p["spread_pips"]:.1f} pips</span>
                              </div>
                            </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.info(f"Connect a trading account to see live prices. ({e})")
            else:
                st.info("Connect an Oanda account in the **Trading** tab to see live prices.")

            # ── Signal cards ──────────────────────────────────────────────────
            _section("ML Signals")
            if user_ok_df.empty:
                st.info("No signals available. Make sure models are trained: `python train_all.py --fetch`")
            else:
                for _, row in user_ok_df.iterrows():
                    try:
                        from src.data_loader import load_forex_data
                        latest_price = float(load_forex_data(data_path(row["pair"]))["Close"].iloc[-1])
                        log_signal(row["pair"], row["signal"], float(row["prob_up"]),
                                   row["confidence"], str(row.get("regime","?")),
                                   bool(row.get("tradeable",False)), latest_price)
                    except Exception:
                        pass

                cols = st.columns(len(user_ok_df))
                for idx, (_, row) in enumerate(user_ok_df.iterrows()):
                    if idx >= len(cols): break
                    is_b  = row["signal"] == "BUY"
                    sc    = C_ACCENT if is_b else C_RED
                    arr   = "▲" if is_b else "▼"
                    conf  = row["confidence"]
                    cc    = C_GREEN if conf=="HIGH" else (C_YELLOW if conf=="MEDIUM" else C_RED)
                    trad  = bool(row.get("tradeable", False))
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background:{'linear-gradient(135deg,#031a12,#051f17)' if is_b else 'linear-gradient(135deg,#1a0308,#1f040c)'};
                             border:1px solid {sc}44;border-left:4px solid {sc};
                             border-radius:14px;padding:20px 18px;margin-bottom:10px;">
                          <div style="font-size:10px;color:{C_MUTED};letter-spacing:3px;margin-bottom:8px;">
                            {row["pair"].replace("_","/")}
                          </div>
                          <div style="font-family:'Syne Mono',monospace;font-size:32px;font-weight:700;color:{sc};line-height:1.1;">
                            {arr} {row["signal"]}
                          </div>
                          <div style="font-size:12px;margin-top:10px;line-height:2;">
                            <span style="color:{C_MUTED};">Prob UP</span> <b style="color:{C_TEXT};font-family:IBM Plex Mono,monospace;">{row["prob_up"]:.4f}</b><br>
                            <span style="color:{C_MUTED};">Confidence</span> <b style="color:{cc};">{conf}</b><br>
                            <span style="color:{C_MUTED};">Regime</span> <b style="color:{C_TEXT};">{row.get("regime","?").upper()}</b><br>
                            <span style="color:{C_MUTED};">Tradeable</span> <b style="color:{'#22c55e' if trad else '#f43f5e'};">{"YES" if trad else "NO"}</b>
                          </div>
                        </div>""", unsafe_allow_html=True)

            # ── Regime panel ──────────────────────────────────────────────────
            _section("Market Regime Analysis")
            if not user_ok_df.empty:
                rc = st.columns(len(user_ok_df))
                for idx, (_, row) in enumerate(user_ok_df.iterrows()):
                    if idx >= len(rc): break
                    with rc[idx]:
                        adx_c = C_GREEN if row.get("regime")=="trending" else (C_YELLOW if row.get("regime")=="transitioning" else C_ACCENT2)
                        vol_c = C_RED if row.get("vol_regime")=="high" else (C_YELLOW if row.get("vol_regime")=="low" else C_GREEN)
                        trend_c = C_GREEN if row.get("trend")=="bullish" else (C_RED if row.get("trend")=="bearish" else C_MUTED)
                        st.markdown(f"""
                        <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:14px;">
                          <div style="font-size:11px;color:{C_MUTED};margin-bottom:8px;font-family:IBM Plex Mono,monospace;">{row["pair"].replace("_","/")}</div>
                          <div style="font-size:12px;line-height:2.2;">
                            <div>ADX Regime: <b style="color:{adx_c};">{row.get("regime","?").upper()}</b></div>
                            <div>Trend: <b style="color:{trend_c};">{row.get("trend","?").upper()}</b></div>
                            <div>Volatility: <b style="color:{vol_c};">{row.get("vol_regime","?").upper()}</b></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

            # ── Signal history chart ──────────────────────────────────────────
            sigs = get_signals_log(limit=300)
            if sigs:
                sl_df = pd.DataFrame(sigs)
                sl_df = sl_df[sl_df["pair"].isin(pairs)]
                if not sl_df.empty:
                    _section("Signal Probability History")
                    fig = go.Figure()
                    colors = [C_ACCENT, C_YELLOW, C_PURPLE, C_ORANGE]
                    for i, p in enumerate(pairs):
                        p_df = sl_df[sl_df["pair"]==p]
                        if not p_df.empty:
                            fig.add_trace(go.Scatter(
                                x=p_df["created_at"], y=p_df["prob_up"],
                                name=p.replace("_","/"), mode="lines+markers",
                                line=dict(color=colors[i % len(colors)], width=1.5),
                                marker=dict(size=4),
                            ))
                    fig.add_hline(y=threshold, line_dash="dot", line_color=C_YELLOW,
                                  annotation_text=f"threshold {threshold:.2f}",
                                  annotation_font_color=C_YELLOW)
                    fig.add_hline(y=0.5, line_dash="dot", line_color=C_MUTED)
                    pf(fig, 280)
                    fig.update_layout(yaxis=dict(range=[0,1]))
                    st.plotly_chart(fig, use_container_width=True)

            # ── Locked pairs ──────────────────────────────────────────────────
            if max_pairs < len(ACTIVE_PAIRS):
                _section("Locked Pairs")
                lc = st.columns(len(ACTIVE_PAIRS) - max_pairs)
                for i, lp in enumerate(ACTIVE_PAIRS[max_pairs:]):
                    with lc[i]:
                        _locked(lp.replace("_","/"), "Pro")

        except Exception as e:
            st.error(f"Signal error: {e}")

    # ══ ANALYSIS TAB ═══════════════════════════════════════════════════════════
    with t_analysis:
        sel_pair = st.selectbox("Select Pair", pairs, key="analysis_pair")
        csv = data_path(sel_pair)

        if not os.path.exists(csv):
            st.warning(f"No data for {sel_pair}. Run: `python train_all.py --fetch`")
        else:
            try:
                from src.data_loader import load_forex_data
                from src.features import add_features
                from src.regime_detector import RegimeDetector

                df_raw = load_forex_data(csv)
                df     = add_features(df_raw)
                n_bars = st.select_slider("Bars to display", [50,100,200,365,500], 200)
                tail   = df_raw.tail(n_bars)
                df_tail= df.tail(n_bars)

                # ── Main chart + BB ───────────────────────────────────────────
                _section(f"{sel_pair.replace('_','/')} — Price & Bollinger Bands")
                fig_main = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.02,
                )
                fig_main.add_trace(go.Candlestick(
                    x=tail["Date"], open=tail["Open"], high=tail["High"],
                    low=tail["Low"], close=tail["Close"],
                    increasing_line_color=C_ACCENT, decreasing_line_color=C_RED,
                    name=sel_pair.replace("_","/"),
                ), row=1, col=1)
                fig_main.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["bb_upper"],
                    name="BB Upper", line=dict(color=C_ACCENT2, width=1, dash="dot"), opacity=0.6), row=1, col=1)
                fig_main.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["bb_lower"],
                    name="BB Lower", line=dict(color=C_ACCENT2, width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(8,145,178,0.06)", opacity=0.6), row=1, col=1)
                # MACD
                mc = [C_ACCENT if v>=0 else C_RED for v in df_tail["macd_hist"]]
                fig_main.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["macd"],
                    name="MACD", line=dict(color=C_ACCENT, width=1.5)), row=2, col=1)
                fig_main.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["macd_signal"],
                    name="Signal", line=dict(color=C_YELLOW, width=1.5)), row=2, col=1)
                fig_main.add_trace(go.Bar(x=df_tail["Date"], y=df_tail["macd_hist"],
                    marker_color=mc, opacity=0.7, name="Hist"), row=2, col=1)
                # RSI
                fig_main.add_hrect(y0=70, y1=100, fillcolor="rgba(244,63,94,0.08)", line_width=0, row=3, col=1)
                fig_main.add_hrect(y0=0, y1=30, fillcolor="rgba(0,212,168,0.08)", line_width=0, row=3, col=1)
                fig_main.add_hline(y=70, line_dash="dot", line_color=C_RED, line_width=1, row=3, col=1)
                fig_main.add_hline(y=30, line_dash="dot", line_color=C_ACCENT, line_width=1, row=3, col=1)
                fig_main.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["rsi_14"],
                    name="RSI", line=dict(color=C_PURPLE, width=1.5)), row=3, col=1)
                fig_main.update_layout(**PLOT, height=620, xaxis_rangeslider_visible=False)
                fig_main.update_yaxes(row=3, col=1, range=[0,100])
                st.plotly_chart(fig_main, use_container_width=True)

                # ── Momentum & Volatility ─────────────────────────────────────
                _section("Momentum & Volatility")
                m1, m2 = st.columns(2)
                with m1:
                    fig_m = go.Figure()
                    fig_m.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["momentum_5"]*100,
                        name="5-bar", line=dict(color=C_ACCENT, width=1.5)))
                    fig_m.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["momentum_10"]*100,
                        name="10-bar", line=dict(color=C_PURPLE, width=1.5)))
                    fig_m.add_hline(y=0, line_color=C_MUTED, line_width=1)
                    pf(fig_m, 240)
                    fig_m.update_layout(title="Momentum (%)", title_font=dict(color=C_MUTED,size=11),
                                         yaxis_ticksuffix="%")
                    st.plotly_chart(fig_m, use_container_width=True)
                with m2:
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["volatility_10"]*100,
                        name="10-bar vol", line=dict(color=C_YELLOW, width=1.5),
                        fill="tozeroy", fillcolor="rgba(234,179,8,0.1)"))
                    fig_v.add_trace(go.Scatter(x=df_tail["Date"], y=df_tail["volatility_20"]*100,
                        name="20-bar vol", line=dict(color=C_RED, width=1.5, dash="dash")))
                    pf(fig_v, 240)
                    fig_v.update_layout(title="Volatility (%)", title_font=dict(color=C_MUTED,size=11),
                                         yaxis_ticksuffix="%")
                    st.plotly_chart(fig_v, use_container_width=True)

                # ── ADX & Regime detail ───────────────────────────────────────
                _section("ADX & Regime Indicators")
                rd     = RegimeDetector()
                regime = rd.detect(df)
                rd_df  = rd.add_regime_features(df).tail(n_bars)

                r1, r2, r3, r4 = st.columns(4)
                adx_c = C_GREEN if regime["adx"]>25 else (C_YELLOW if regime["adx"]>20 else C_ORANGE)
                r1.markdown(_kpi("ADX", f"{regime['adx']:.1f}", sub="Trend strength"), unsafe_allow_html=True)
                r2.markdown(_kpi("+DI", f"{regime['plus_di']:.1f}", sub="Bullish pressure"), unsafe_allow_html=True)
                r3.markdown(_kpi("-DI", f"{regime['minus_di']:.1f}", sub="Bearish pressure"), unsafe_allow_html=True)
                r4.markdown(_kpi("Vol Ratio", f"{regime['vol_ratio']:.2f}x", sub="vs 90-day avg"), unsafe_allow_html=True)

                fig_adx = go.Figure()
                fig_adx.add_trace(go.Scatter(x=rd_df["Date"], y=rd_df["adx"],
                    name="ADX", line=dict(color=C_YELLOW, width=2)))
                fig_adx.add_trace(go.Scatter(x=rd_df["Date"], y=rd_df["plus_di"],
                    name="+DI", line=dict(color=C_GREEN, width=1.5)))
                fig_adx.add_trace(go.Scatter(x=rd_df["Date"], y=rd_df["minus_di"],
                    name="-DI", line=dict(color=C_RED, width=1.5)))
                fig_adx.add_hline(y=25, line_dash="dot", line_color=C_MUTED,
                                   annotation_text="Trend threshold (25)")
                fig_adx.add_hline(y=20, line_dash="dot", line_color=C_DIM)
                pf(fig_adx, 280)
                st.plotly_chart(fig_adx, use_container_width=True)

                # ── Walk-forward results ──────────────────────────────────────
                if pui["wf"]:
                    wf = wf_path(sel_pair)
                    if os.path.exists(wf):
                        _section("Walk-Forward Validation Results")
                        wf_df = pd.read_csv(wf)
                        w1, w2 = st.columns(2)
                        with w1:
                            wf_acc_mean = wf_df["accuracy"].mean()
                            wm1,wm2,wm3 = st.columns(3)
                            wm1.markdown(_kpi("Mean WF Acc", f"{wf_acc_mean:.3f}", "g" if wf_acc_mean>0.52 else "r"), unsafe_allow_html=True)
                            wm2.markdown(_kpi("Profitable Splits", f"{(wf_df['strategy_return']>0).sum()}/{len(wf_df)}", "y"), unsafe_allow_html=True)
                            wm3.markdown(_kpi("Mean Return", f"{wf_df['strategy_return'].mean():.3f}", "g" if wf_df['strategy_return'].mean()>0 else "r"), unsafe_allow_html=True)

                            acc_c = [C_ACCENT if v>0.52 else (C_YELLOW if v>0.50 else C_RED) for v in wf_df["accuracy"]]
                            fig_wf = go.Figure(go.Bar(x=wf_df["split_id"], y=wf_df["accuracy"],
                                marker_color=acc_c))
                            fig_wf.add_hline(y=0.5, line_dash="dot", line_color=C_MUTED)
                            pf(fig_wf, 220)
                            fig_wf.update_layout(showlegend=False, title="Accuracy per Split",
                                                  title_font=dict(color=C_MUTED,size=11))
                            st.plotly_chart(fig_wf, use_container_width=True)
                        with w2:
                            ret_c = [C_ACCENT if v>0 else C_RED for v in wf_df["strategy_return"]]
                            fig_ret = go.Figure(go.Bar(x=wf_df["split_id"],
                                y=wf_df["strategy_return"]*100, marker_color=ret_c))
                            fig_ret.add_hline(y=0, line_color=C_MUTED)
                            pf(fig_ret, 220)
                            fig_ret.update_layout(showlegend=False, title="Strategy Return per Split (%)",
                                                   title_font=dict(color=C_MUTED,size=11),
                                                   yaxis_ticksuffix="%")
                            st.plotly_chart(fig_ret, use_container_width=True)

                        # Brier score per walk-forward split (Task 3.2)
                        if "brier_score" in wf_df.columns:
                            with st.expander("Brier Score per Walk-Forward Split"):
                                brier_colors = [
                                    C_GREEN if v < 0.22 else (C_YELLOW if v < 0.25 else C_RED)
                                    for v in wf_df["brier_score"]
                                ]
                                fig_bs = go.Figure(go.Bar(
                                    x=wf_df["split_id"], y=wf_df["brier_score"],
                                    marker_color=brier_colors,
                                    text=[f"{v:.4f}" for v in wf_df["brier_score"]],
                                    textposition="outside",
                                ))
                                fig_bs.add_hline(y=0.25, line_dash="dot", line_color=C_RED,
                                                  annotation_text="Coin flip (0.25)",
                                                  annotation_font_color=C_RED)
                                pf(fig_bs, 220)
                                fig_bs.update_layout(
                                    showlegend=False,
                                    title="Brier Score per Split (lower = better calibration)",
                                    title_font=dict(color=C_MUTED, size=11),
                                    yaxis=dict(range=[0, 0.30]),
                                )
                                st.plotly_chart(fig_bs, use_container_width=True)
                                st.caption(
                                    f"Mean Brier = **{wf_df['brier_score'].mean():.4f}** across {len(wf_df)} splits. "
                                    "Green < 0.22 · Yellow < 0.25 · Red ≥ 0.25 (coin flip)."
                                )

                        st.dataframe(wf_df, use_container_width=True, hide_index=True, height=200)
                    else:
                        st.info("Walk-forward results not found. Run: `python train_all.py`")
                else:
                    _section("Walk-Forward Validation")
                    _locked("Walk-Forward Results", "Basic")

                # ══════════════════════════════════════════════════════════════
                # Model Health section — includes Brier score KPIs (Task 3.2)
                # and reliability diagram
                # ══════════════════════════════════════════════════════════════
                _section("Model Health")
                mp = meta_path(sel_pair)
                if os.path.exists(mp):
                    with open(mp) as f: meta = json.load(f)

                    # Row 1: core accuracy KPIs
                    h1,h2,h3,h4 = st.columns(4)
                    h1.markdown(_kpi("Train Accuracy", f"{meta.get('accuracy_train',0):.3f}", "y"), unsafe_allow_html=True)
                    h2.markdown(_kpi("Test Accuracy",  f"{meta.get('accuracy_test',0):.3f}",
                                     "g" if meta.get('accuracy_test',0)>0.52 else "r"), unsafe_allow_html=True)
                    h3.markdown(_kpi("WF Accuracy",    f"{meta.get('walk_forward_mean_accuracy',0):.3f}",
                                     "g" if meta.get('walk_forward_mean_accuracy',0)>0.52 else "r"), unsafe_allow_html=True)
                    h4.markdown(_kpi("Training Rows",  str(meta.get('rows_total','?')), "a"), unsafe_allow_html=True)

                    # Row 2: gross WF performance
                    p1,p2,p3,p4 = st.columns(4)
                    pf_val  = meta.get("walk_forward_mean_profit_factor")
                    sh_val  = meta.get("walk_forward_mean_sharpe")
                    ex_val  = meta.get("walk_forward_mean_expectancy")
                    exp_val = meta.get("walk_forward_mean_exposure")
                    p1.markdown(_kpi("WF Profit Factor (gross)", f"{pf_val:.2f}" if pf_val is not None else "—", "m"), unsafe_allow_html=True)
                    p2.markdown(_kpi("WF Sharpe", f"{sh_val:.2f}" if sh_val is not None else "—",
                                     "g" if (sh_val or 0) > 0 else "r"), unsafe_allow_html=True)
                    p3.markdown(_kpi("Expectancy", f"{ex_val:+.5f}" if ex_val is not None else "—",
                                     "g" if (ex_val or 0) > 0 else "r"), unsafe_allow_html=True)
                    p4.markdown(_kpi("Exposure", f"{exp_val*100:.1f}%" if exp_val is not None else "—", "m"), unsafe_allow_html=True)

                    # Row 3: net-of-realistic-cost figures
                    net_pf         = meta.get("walk_forward_mean_net_profit_factor")
                    net_sh         = meta.get("walk_forward_mean_net_sharpe")
                    net_ex         = meta.get("walk_forward_mean_net_expectancy")
                    net_splits_pct = meta.get("walk_forward_net_profitable_splits_pct")
                    if net_pf is not None:
                        n1,n2,n3,n4 = st.columns(4)
                        n1.markdown(_kpi("Net Profit Factor", f"{net_pf:.2f}",
                                         "g" if net_pf > TRADABLE_MIN_NET_PROFIT_FACTOR else "r",
                                         sub="after realistic spread, slippage & swap"), unsafe_allow_html=True)
                        n2.markdown(_kpi("Net Sharpe", f"{net_sh:.2f}" if net_sh is not None else "—",
                                         "g" if (net_sh or 0) > 0 else "r"), unsafe_allow_html=True)
                        n3.markdown(_kpi("Net Expectancy", f"{net_ex:+.5f}" if net_ex is not None else "—",
                                         "g" if (net_ex or 0) > 0 else "r"), unsafe_allow_html=True)
                        n4.markdown(_kpi("Net Profitable Splits", f"{net_splits_pct*100:.0f}%" if net_splits_pct is not None else "—",
                                         "g" if (net_splits_pct or 0) >= TRADABLE_MIN_PROFITABLE_SPLITS_PCT else "r"), unsafe_allow_html=True)

                    # ── Row 4: Calibration quality (Task 3.2) ─────────────────
                    # Brier score is the primary calibration metric:
                    #   perfect = 0.0  |  coin flip ≈ 0.25  |  target < 0.22
                    # Platt scaling (sigmoid) reduces over-confidence in RF
                    # probabilities without requiring isotonic regression's
                    # larger data budget.
                    brier_test    = meta.get("brier_score_test")
                    brier_wf_mean = meta.get("brier_score_wf_mean")
                    is_calibrated = meta.get("is_calibrated", False)
                    cal_method    = meta.get("calibration_method", "—")

                    b1, b2, b3 = st.columns(3)
                    b1.markdown(_kpi(
                        "Brier Score (test)",
                        f"{brier_test:.4f}" if brier_test is not None else "—",
                        "g" if (brier_test or 1) < 0.22 else ("y" if (brier_test or 1) < 0.25 else "r"),
                        sub="Lower is better · coin flip ≈ 0.25",
                    ), unsafe_allow_html=True)
                    b2.markdown(_kpi(
                        "Brier Score (WF mean)",
                        f"{brier_wf_mean:.4f}" if brier_wf_mean is not None else "—",
                        "g" if (brier_wf_mean or 1) < 0.22 else ("y" if (brier_wf_mean or 1) < 0.25 else "r"),
                        sub="Mean across walk-forward splits",
                    ), unsafe_allow_html=True)
                    b3.markdown(_kpi(
                        "Calibration",
                        f"✓ {cal_method.upper()}" if is_calibrated else "✗ None",
                        "g" if is_calibrated else "r",
                        sub="Platt scaling via CalibratedClassifierCV",
                    ), unsafe_allow_html=True)

                    # ── Status banner ──────────────────────────────────────────
                    gap = (meta.get("accuracy_train",0) - meta.get("accuracy_test",0))
                    is_tradable = meta.get("is_tradable_edge")
                    if gap > 0.15:
                        st.warning(f"⚠️ Overfitting gap is {gap:.3f}. Consider retraining.")
                    elif is_tradable is None:
                        st.warning("🟡 This model predates net-of-cost reporting. Retrain to get an accurate tradable-edge check.")
                    elif not is_tradable:
                        st.error(
                            f"🔴 Weak tradable edge after realistic costs — needs WF accuracy > {TRADABLE_MIN_WF_ACCURACY:.0%}, "
                            f"net profit factor > {TRADABLE_MIN_NET_PROFIT_FACTOR:.2f}, and "
                            f"≥{TRADABLE_MIN_PROFITABLE_SPLITS_PCT:.0%} profitable splits. Not recommended for auto-trading."
                        )
                    else:
                        st.success(f"✅ Model looks healthy net of realistic costs. Trained: {meta.get('trained_at','?')[:10]}")

                    # ── Task 3.3: paper validation status ─────────────────────
                    try:
                        from src.paper_validator import get_paper_stats, PAPER_SIGNALS_NEEDED
                        pstats       = get_paper_stats(sel_pair)
                        model_status = pstats["model_status"]
                        paper_count  = pstats["paper_count"]
                        paper_wr     = pstats["paper_win_rate"]
                        val_at       = (pstats["validated_at"] or "")[:10]
                        is_override  = pstats["validation_override"]

                        _section("Auto-Trade Validation Status")
                        if model_status == "paper_only":
                            needed   = max(0, PAPER_SIGNALS_NEEDED - paper_count)
                            wr_str   = f", win rate {paper_wr*100:.0f}%" if paper_wr is not None else ""
                            phase    = (
                                f"Needs {needed} more tradeable signals."
                                if needed > 0
                                else "Signal count met — win rate threshold not yet reached."
                            )
                            st.info(
                                f"📄 **Paper Validation Mode** — "
                                f"{paper_count}/{PAPER_SIGNALS_NEEDED} signals resolved{wr_str}. "
                                f"{phase} Auto-trading is **disabled** until the gate clears. "
                                f"Signals are still logged and outcomes tracked automatically."
                            )
                        elif model_status == "validated":
                            wr_str       = f" · paper win rate {paper_wr*100:.0f}%" if paper_wr is not None else ""
                            override_str = " *(admin override)*" if is_override else ""
                            val_str      = f" — validated {val_at}" if val_at else ""
                            st.success(
                                f"✅ **Model Validated**{override_str}{val_str}{wr_str}. "
                                f"Auto-trading is **enabled** for this pair."
                            )
                        else:
                            st.warning(
                                f"⚠️ Unknown model status: `{model_status}`. "
                                "Retrain to reset the validation gate."
                            )
                    except ImportError:
                        pass   # paper_validator not yet deployed — show nothing
                    except Exception as _ve:
                        logger.debug(f"paper_validator status display error: {_ve}")

                    # ── Reliability diagram (calibration curve) ────────────────
                    # Rendered when is_calibrated=True (models trained after
                    # Task 3.2).  Uses the held-out 20% test split so the curve
                    # reflects genuine out-of-sample calibration quality.
                    if is_calibrated:
                        _render_reliability_diagram(df, meta, sel_pair)
                    elif meta.get("trained_at"):
                        st.info(
                            "🔄 This model was trained before calibration was added (Task 3.2). "
                            "Run `python train_all.py --fetch` to retrain with Platt scaling "
                            "and unlock the Brier score KPIs and reliability diagram."
                        )

                else:
                    st.info(f"Model not trained for {sel_pair}. Run: `python train_all.py --fetch`")

            except Exception as e:
                st.error(f"Analysis error: {e}")

    # ══ TRADING TAB ════════════════════════════════════════════════════════════
    with t_trading:
        if not can_trade:
            _locked("Auto-Trading & Order Management", "Pro")
            st.markdown("<br>", unsafe_allow_html=True)
            _section("Why Upgrade to Pro?")
            st.markdown(f"""
            <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:12px;padding:20px 24px;">
              <div style="font-size:13px;line-height:2.2;color:{C_MUTED};">
                ✅ &nbsp;Connect your Oanda practice or live account<br>
                ✅ &nbsp;Place BUY/SELL orders directly from signals<br>
                ✅ &nbsp;Automatic stop loss & take profit calculation<br>
                ✅ &nbsp;Risk-based position sizing<br>
                ✅ &nbsp;Close individual trades from the dashboard<br>
                ✅ &nbsp;View open positions & pending orders<br>
                ✅ &nbsp;Full transaction history from Oanda<br>
                ✅ &nbsp;Trade all 4 pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD)
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            accounts = get_trading_accounts(user["id"])
            settings = get_user_trading_settings(user["id"])
            selected_account_id = settings.get("trading_account_id")
            if accounts and not selected_account_id:
                selected_account_id = accounts[0]["id"]

            _section("Trading Accounts")
            if accounts:
                account_options = {acc["id"]: f"{acc['account_name']} · {acc['account_id']} · {acc['environment'].upper()}" for acc in accounts}
                selected_account_id = st.selectbox(
                    "Active account for trading",
                    options=list(account_options.keys()),
                    index=list(account_options.keys()).index(selected_account_id)
                    if selected_account_id in account_options else 0,
                    format_func=lambda aid: account_options[aid],
                    key="trading_active_account",
                )
                if selected_account_id != settings.get("trading_account_id"):
                    update_user_trading_settings(user["id"], trading_account_id=selected_account_id)
                    settings = get_user_trading_settings(user["id"])
                for acc in accounts:
                    env_color = C_YELLOW if acc["environment"] == "live" else C_ACCENT2
                    c1, c2, c3 = st.columns([4, 2, 1])
                    with c1:
                        st.markdown(f"""
                        <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:12px 16px;">
                          <div style="font-size:13px;font-weight:600;color:{C_TEXT};">{acc["account_name"]}</div>
                          <div style="font-size:11px;color:{C_MUTED};margin-top:4px;">
                            {acc["broker"].upper()} · <code style="color:{C_TEXT};">{acc["account_id"]}</code> ·
                            <span style="color:{env_color};">{acc["environment"].upper()}</span>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        try:
                            client_test = OandaClient_for_acc(acc)
                            summary_test = client_test.get_account_summary()
                            st.markdown(f'<div style="font-size:12px;color:{C_GREEN};padding:14px 0;">✓ Connected · ${summary_test["balance"]:,.2f}</div>', unsafe_allow_html=True)
                        except Exception as e:
                            st.markdown(f'<div style="font-size:12px;color:{C_RED};padding:14px 0;">✗ Error</div>', unsafe_allow_html=True)
                    with c3:
                        if st.button("Remove", key=f"rm_{acc['id']}"):
                            try:
                                api_delete(f"/account/trading-accounts/{acc['id']}")
                                st.rerun()
                            except ApiError as e:
                                st.error(e.detail)
            else:
                st.info("No trading account connected yet.")

            with st.expander("➕ Connect Oanda Account"):
                with st.form("add_account_form"):
                    st.markdown(f"""
                    <div style="font-size:12px;color:{C_MUTED};margin-bottom:12px;line-height:1.8;">
                      1. Sign up at <b style="color:{C_TEXT};">oanda.com/register</b> (free practice account)<br>
                      2. Go to <b>My Account → Manage API Access → Generate Token</b><br>
                      3. Find your Account ID on the dashboard (e.g. 101-001-XXXXXXX-001)
                    </div>""", unsafe_allow_html=True)
                    acc_name = st.text_input("Account label", placeholder="My Practice Account")
                    api_key  = st.text_input("Oanda API Token", type="password",
                                              placeholder="Paste your API token here")
                    acc_id   = st.text_input("Account ID", placeholder="101-001-XXXXXXX-001")
                    env      = st.selectbox("Environment", ["practice", "live"])
                    if env == "live":
                        st.error("⚠️ Live trading is not available for user accounts in v1. Please select **practice** mode.")
                        st.stop()
                    if st.form_submit_button("Verify & Connect", type="primary"):
                        if acc_name and api_key and acc_id:
                            with st.spinner("Verifying credentials with Oanda..."):
                                try:
                                    resp = api_post("/account/trading-accounts", json={
                                        "account_name": acc_name,
                                        "api_key": api_key,
                                        "account_id": acc_id,
                                        "environment": env,
                                    })
                                    st.success(f"✅ Connected! Balance: ${resp['balance']:,.2f} {resp['currency']}")
                                    st.rerun()
                                except ApiError as e:
                                    st.error(e.detail)
                                except Exception as e:
                                    st.error(str(e))
                        else:
                            st.error("All fields are required.")

            if accounts:
                try:
                    from src.oanda_client import OandaClient
                    acc    = next((a for a in accounts if int(a["id"]) == int(selected_account_id)), accounts[0])
                    client = OandaClient(api_key=acc["api_key_enc"], account_id=acc["account_id"], environment=acc["environment"])
                    summary = client.get_account_summary()
                    upl     = summary["unrealized_pl"]

                    _section("Account Overview")
                    try: daily_pnl = client.get_daily_pnl()
                    except Exception: daily_pnl = 0.0
                    k1,k2,k3,k4,k5,k6 = st.columns(6)
                    k1.markdown(_kpi("Balance",       f"${summary['balance']:,.2f}", "a"), unsafe_allow_html=True)
                    k2.markdown(_kpi("NAV",           f"${summary['nav']:,.2f}", "g" if summary["nav"]>=summary["balance"] else "r"), unsafe_allow_html=True)
                    k3.markdown(_kpi("Unrealized P&L",f"${upl:+,.2f}","g" if upl>=0 else "r"), unsafe_allow_html=True)
                    k4.markdown(_kpi("Realized P&L",  f"${summary['realized_pl']:+,.2f}","g" if summary['realized_pl']>=0 else "r"), unsafe_allow_html=True)
                    k5.markdown(_kpi("Daily P&L",     f"${daily_pnl:+,.2f}","g" if daily_pnl>=0 else "r"), unsafe_allow_html=True)
                    k6.markdown(_kpi("Margin Used",   f"${summary['margin_used']:,.2f}", "y"), unsafe_allow_html=True)

                    _section("Place Trade")
                    oc1,oc2,oc3,oc4,oc5 = st.columns(5)
                    trade_pair  = oc1.selectbox("Pair",      pairs,          key="tp")
                    trade_dir   = oc2.selectbox("Direction", ["BUY","SELL"],  key="td")
                    trade_units = oc3.number_input("Units",  100, 100000, 1000, 100, key="tu")
                    trade_sl    = oc4.number_input("SL pips", 5, 200, int(sl_pips), 5, key="tsl")
                    trade_tp    = oc5.number_input("TP pips", 5, 200, int(tp_pips), 5, key="ttp")

                    try:
                        lp = client.get_live_price(trade_pair)
                        entry_est = lp["mid"]
                        pip = 0.01 if "JPY" in trade_pair else 0.0001
                        sl_est = round(entry_est - trade_sl*pip if trade_dir=="BUY" else entry_est + trade_sl*pip, 5)
                        tp_est = round(entry_est + trade_tp*pip if trade_dir=="BUY" else entry_est - trade_tp*pip, 5)
                        risk_est = trade_units * trade_sl * pip
                        st.markdown(f"""
                        <div style="background:{C_SURF2};border:1px solid {C_BORDER};border-radius:10px;padding:12px 16px;margin:10px 0;font-size:12px;font-family:IBM Plex Mono,monospace;">
                          <span style="color:{C_MUTED};">Entry ~</span> <b style="color:{C_TEXT};">{entry_est:.5f}</b> &nbsp;·&nbsp;
                          <span style="color:{C_MUTED};">SL</span> <b style="color:{C_RED};">{sl_est:.5f}</b> &nbsp;·&nbsp;
                          <span style="color:{C_MUTED};">TP</span> <b style="color:{C_GREEN};">{tp_est:.5f}</b> &nbsp;·&nbsp;
                          <span style="color:{C_MUTED};">Est. risk</span> <b style="color:{C_YELLOW};">${risk_est:.2f}</b> &nbsp;·&nbsp;
                          <span style="color:{C_MUTED};">Spread</span> <b style="color:{C_TEXT};">{lp["spread"]:.1f} pips</b>
                        </div>""", unsafe_allow_html=True)
                    except Exception: pass

                    order_type = st.radio("Order Type", ["Market", "Limit"], horizontal=True, key="order_type_sel")
                    limit_price = None
                    if order_type == "Limit":
                        limit_price = st.number_input("Limit Price", value=float(entry_est) if 'entry_est' in dir() else 0.0, step=0.0001, format="%.5f", key="limit_px")

                    try:
                        from src.trading_engine import calculate_position_size
                        pip_val = 0.01 if "JPY" in trade_pair else 0.0001
                        auto_u  = calculate_position_size(summary["balance"], risk_pct, trade_sl, pip_val)
                        st.markdown(f'<div style="font-size:11px;color:{C_MUTED};margin-bottom:8px;">Risk-sized units at {risk_pct*100:.1f}%: <b style="color:{C_YELLOW};">{auto_u:,}</b></div>', unsafe_allow_html=True)
                    except Exception: auto_u = trade_units

                    c_btn1, c_btn2, c_btn3 = st.columns(3)
                    if c_btn1.button(f"▶ Place {order_type} {trade_dir}", type="primary"):
                        with st.spinner("Placing order..."):
                            try:
                                resp = api_post("/trading/place", json={"instrument": trade_pair, "direction": trade_dir, "units": trade_units, "sl_pips": float(trade_sl), "tp_pips": float(trade_tp), "order_type": order_type, "limit_price": limit_price, "account_db_id": selected_account_id})
                                if order_type == "Market":
                                    fill = resp["result"]["fill"]
                                    st.success(f"✅ {trade_dir} {trade_units} {trade_pair} @ {fill['fill_price']:.5f}")
                                else:
                                    st.success(f"✅ Limit {trade_dir} {trade_units} {trade_pair} @ {resp['entry']:.5f} placed")
                                st.rerun()
                            except ApiError as e: st.error(f"Order failed: {e.detail}")

                    if c_btn2.button("🤖 Signal + Trade"):
                        with st.spinner("Running ML signal check..."):
                            try:
                                result = api_post("/trading/signal-trade", params={"pair": trade_pair, "threshold": threshold, "sl_pips": float(trade_sl), "tp_pips": float(trade_tp), "units": trade_units, "account_db_id": selected_account_id})
                                if result["action"] == "order_placed":
                                    st.success(f"✅ Auto-trade: {result['signal']} @ {result.get('fill', {}).get('fill_price','?')}")
                                elif result["action"] == "error":
                                    st.error(f"Error: {result['reason']}")
                                else:
                                    st.info(f"Signal: {result.get('signal','—')} — Not traded: {result.get('reason','')}")
                                st.rerun()
                            except ApiError as e: st.error(e.detail)

                    _section("Open Positions")
                    open_trades = client.get_open_trades()
                    if open_trades:
                        for i, trade in enumerate(open_trades):
                            upl_c = C_GREEN if trade["unrealized_pl"]>=0 else C_RED
                            tc1,tc2,tc3,tc4,tc5,tc6 = st.columns([2,2,1,1,2,1])
                            tc1.markdown(f'<div style="font-size:13px;font-weight:600;padding:8px 0;">{trade["instrument"].replace("_","/")}</div>', unsafe_allow_html=True)
                            tc2.markdown(f'<div style="font-size:12px;color:{C_MUTED};padding:8px 0;">{trade["units"]:+d} @ {trade["open_price"]:.5f}</div>', unsafe_allow_html=True)
                            tc3.markdown(f'<div style="font-size:12px;color:{upl_c};font-weight:600;padding:8px 0;">${trade["unrealized_pl"]:+.2f}</div>', unsafe_allow_html=True)
                            tc4.markdown(f'<div style="font-size:11px;color:{C_MUTED};padding:8px 0;">SL:{trade["stop_loss"]}</div>', unsafe_allow_html=True)
                            with tc5:
                                new_sl = st.number_input("SL", value=0.0, step=0.0001, format="%.5f", key=f"usr_sl_{trade['trade_id']}_{i}", label_visibility="collapsed", placeholder="New SL")
                                new_tp = st.number_input("TP", value=0.0, step=0.0001, format="%.5f", key=f"usr_tp_{trade['trade_id']}_{i}", label_visibility="collapsed", placeholder="New TP")
                            with tc6:
                                if st.button("Mod", key=f"usr_mod_{trade['trade_id']}_{i}") and (new_sl or new_tp):
                                    try:
                                        api_post("/trading/modify", json={"trade_id": trade["trade_id"], "account_db_id": selected_account_id, "stop_loss": new_sl if new_sl > 0 else None, "take_profit": new_tp if new_tp > 0 else None})
                                        st.success("Modified."); st.rerun()
                                    except ApiError as e: st.error(e.detail)
                                if st.button("Close", key=f"close_{trade['trade_id']}_{i}"):
                                    with st.spinner(f"Closing {trade['instrument']}..."):
                                        try:
                                            user_trades = get_user_trades(user["id"], limit=30)
                                            match = next((t for t in user_trades if t.get("broker_trade_id")==trade["trade_id"] and t["status"]=="open"), None)
                                            db_trade_id = match["id"] if match else 0
                                            res = api_post("/trading/close", json={"broker_trade_id": trade["trade_id"], "db_trade_id": db_trade_id, "account_db_id": selected_account_id})["result"]
                                            st.success(f"Closed @ {res['fill_price']:.5f} | P&L: ${res['pl']:+.2f}")
                                            st.rerun()
                                        except ApiError as e: st.error(f"Close failed: {e.detail}")
                            st.markdown(f'<hr style="border-color:{C_DIM};margin:2px 0;">', unsafe_allow_html=True)
                        if st.button("🔴 Close All Positions"):
                            with st.spinner("Closing all..."):
                                try:
                                    api_post("/trading/close-all", params={"account_db_id": selected_account_id})
                                    st.success("All positions closed."); st.rerun()
                                except ApiError as e: st.error(e.detail)
                    else:
                        st.info("No open positions.")

                    _section("Oanda Transaction History")
                    try:
                        txns = client.get_transaction_history(count=50)
                        if txns:
                            st.dataframe(pd.DataFrame(txns), use_container_width=True, hide_index=True, height=280)
                        else:
                            st.info("No transactions yet.")
                    except Exception as e:
                        st.info(f"Transaction history unavailable: {e}")

                except ValueError as e: st.warning(str(e))
                except Exception as e: st.error(f"Account error: {e}")

    # ══ MY TRADES TAB ══════════════════════════════════════════════════════════
    with t_trades:
        if not can_trade:
            _locked("Trade History & P&L Analytics", "Pro")
        else:
            trade_stats = get_trade_stats(user["id"])
            s1,s2,s3,s4,s5 = st.columns(5)
            s1.markdown(_kpi("Total Trades",  str(trade_stats["total_trades"]), "a"), unsafe_allow_html=True)
            s2.markdown(_kpi("Closed",        str(trade_stats["closed_trades"]), "m"), unsafe_allow_html=True)
            s3.markdown(_kpi("Total P&L",     f"${trade_stats['total_pnl']:+,.2f}", "g" if trade_stats["total_pnl"]>=0 else "r"), unsafe_allow_html=True)
            s4.markdown(_kpi("Win Rate",      f"{trade_stats['win_rate']:.1f}%", "g" if trade_stats["win_rate"]>=50 else "r"), unsafe_allow_html=True)
            s5.markdown(_kpi("Avg P&L",       f"${trade_stats['avg_pnl']:+,.2f}", "g" if trade_stats["avg_pnl"]>=0 else "r"), unsafe_allow_html=True)

            trades = get_user_trades(user["id"], limit=200)
            if trades:
                _section("Trade Log")
                tr_df = pd.DataFrame(trades)
                st.dataframe(tr_df, use_container_width=True, hide_index=True, height=300)

                closed = [t for t in trades if t.get("pnl") is not None]
                if closed:
                    _section("P&L Analysis")
                    pnl_vals = [t["pnl"] for t in closed]
                    pnl_cum  = list(np.cumsum(pnl_vals))

                    pa1, pa2 = st.columns(2)
                    with pa1:
                        fig_bar = go.Figure(go.Bar(y=pnl_vals, marker_color=[C_ACCENT if v>=0 else C_RED for v in pnl_vals]))
                        fig_bar.add_hline(y=0, line_color=C_MUTED, line_width=1)
                        pf(fig_bar, 260)
                        fig_bar.update_layout(showlegend=False, title="P&L per Trade", title_font=dict(color=C_MUTED,size=11))
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with pa2:
                        fig_cum = go.Figure(go.Scatter(y=pnl_cum, mode="lines", line=dict(color=C_ACCENT if pnl_cum[-1]>=0 else C_RED, width=2), fill="tozeroy", fillcolor=f"rgba(0,212,168,0.1)" if pnl_cum[-1]>=0 else f"rgba(244,63,94,0.1)"))
                        fig_cum.add_hline(y=0, line_color=C_MUTED, line_width=1)
                        pf(fig_cum, 260)
                        fig_cum.update_layout(showlegend=False, title="Cumulative P&L", title_font=dict(color=C_MUTED,size=11))
                        st.plotly_chart(fig_cum, use_container_width=True)
            else:
                st.info("No trades yet. Use the Trading tab to place your first trade.")

    # ══ ALERTS TAB ═════════════════════════════════════════════════════════════
    with t_alerts:
        _section("Notifications")
        all_notifs = get_notifications(user["id"], unread_only=False)
        if all_notifs:
            if st.button("✓ Mark all as read"):
                try:
                    api_post("/account/notifications/mark-read", json={})
                    st.rerun()
                except ApiError as e: st.error(e.detail)
            for n in all_notifs:
                dot  = C_ACCENT if not n["is_read"] else C_MUTED
                type_icon = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🔴"}.get(n.get("type","info"),"ℹ️")
                st.markdown(f"""
                <div style="background:{C_CARD};border-left:3px solid {dot};border-radius:8px;padding:12px 16px;margin:6px 0;">
                  <div style="font-size:13px;font-weight:600;color:{C_TEXT};">{type_icon} {n["title"]}</div>
                  <div style="font-size:12px;color:{C_MUTED};margin-top:4px;">{n["message"]}</div>
                  <div style="font-size:10px;color:{C_DIM};margin-top:6px;font-family:IBM Plex Mono,monospace;">{n["created_at"][:16]}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No notifications yet.")

        if pui["alerts"]:
            _section("Telegram Alerts")
            from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
            tg_ok = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_TOKEN not in ("","YOUR_BOT_TOKEN_HERE"))
            chat_ok = bool(TELEGRAM_CHAT_ID and TELEGRAM_CHAT_ID not in ("","YOUR_CHAT_ID_HERE"))
            t_col1, t_col2 = st.columns([3, 2])
            with t_col1:
                st.markdown(f"""
                <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:14px 16px;font-size:12px;line-height:2;">
                  <div>Bot token: <b style="color:{'#22c55e' if tg_ok else '#f43f5e'};">{'✓ Set' if tg_ok else '✗ Not set'}</b></div>
                  <div>Chat ID: <b style="color:{'#22c55e' if chat_ok else '#f43f5e'};">{'✓ Set' if chat_ok else '✗ Not set'}</b></div>
                  <div>Status: <b style="color:{'#22c55e' if (tg_ok and chat_ok) else '#f43f5e'};">{'Connected' if (tg_ok and chat_ok) else 'Not configured'}</b></div>
                </div>""", unsafe_allow_html=True)
            with t_col2:
                if st.button("📨 Send Test Message"):
                    from src.alerter import Alerter
                    ok = Alerter().test()
                    if ok: st.success("✅ Sent!")
                    else: st.error("Failed. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env")
        else:
            _section("Telegram Alerts")
            _locked("Telegram Alerts & Notifications", "Pro")

    # ══ ACCOUNT TAB ════════════════════════════════════════════════════════════
    with t_account:
        ac1, ac2 = st.columns(2, gap="large")

        with ac1:
            _section("Edit Profile")
            with st.form("profile_form"):
                new_name  = st.text_input("Full name",  value=user.get("full_name",""))
                new_phone = st.text_input("Phone",      value=user.get("phone",""))
                new_email = st.text_input("Email",      value=user.get("email",""))
                if st.form_submit_button("Save Profile", type="primary"):
                    try:
                        api_post("/account/profile", json={"full_name": new_name.strip(), "phone": new_phone.strip(), "email": new_email.strip()})
                        st.session_state["user"]["full_name"] = new_name.strip()
                        st.session_state["user"]["email"]     = new_email.strip()
                        st.success("Profile updated.")
                    except ApiError as e: st.error(e.detail)

            _section("Change Password")
            with st.form("password_form"):
                cur_pw  = st.text_input("Current password",  type="password")
                new_pw  = st.text_input("New password",      type="password")
                conf_pw = st.text_input("Confirm new",       type="password")
                if st.form_submit_button("Update Password"):
                    if new_pw != conf_pw: st.error("Passwords do not match.")
                    elif len(new_pw) < 8: st.error("Password must be at least 8 characters.")
                    else:
                        try:
                            api_post("/account/password", json={"current_value": cur_pw, "new_value": new_pw})
                            st.success("Password updated.")
                        except ApiError as e: st.error(e.detail)

            _section("Session")
            st.markdown(f'<div style="font-size:12px;color:{C_MUTED};margin-bottom:8px;">You will stay signed in when refreshing this browser tab. Use Sign out to end the session.</div>', unsafe_allow_html=True)
            render_logout_button(sidebar=False, key="user_logout_account")

        with ac2:
            _section("Trading Mode")
            mode_settings = get_user_trading_settings(user["id"])
            account_rows = get_trading_accounts(user["id"])
            can_auto = plan in ("pro", "enterprise")
            mode_labels = {"signals_only": "Signals only", "manual": "Manual trading", "auto": "Auto trade"}
            allowed_modes = ["signals_only", "manual", "auto"] if can_auto else ["signals_only"]
            current_mode = mode_settings.get("mode", "signals_only")
            if current_mode not in allowed_modes: current_mode = "signals_only"

            with st.form("trading_mode_form"):
                mode = st.radio("Mode", allowed_modes, format_func=lambda m: mode_labels[m], index=allowed_modes.index(current_mode), horizontal=True)
                selected_acc = mode_settings.get("trading_account_id")
                if account_rows:
                    account_ids = [a["id"] for a in account_rows]
                    selected_acc = st.selectbox("Trading account", options=account_ids, index=account_ids.index(selected_acc) if selected_acc in account_ids else 0, format_func=lambda aid: next(f"{a['account_name']} · {a['account_id']} · {a['environment'].upper()}" for a in account_rows if a["id"] == aid))
                else:
                    st.info("Connect an Oanda account to enable manual or auto execution.")
                    selected_acc = None

                c1, c2 = st.columns(2)
                threshold_cfg = c1.slider("Auto threshold", 0.50, 0.90, float(mode_settings.get("threshold") or 0.55), 0.01)
                risk_cfg = c2.slider("Risk per trade (%)", 0.5, 5.0, float(mode_settings.get("risk_pct") or 0.01) * 100, 0.5) / 100
                c3, c4 = st.columns(2)
                sl_cfg = c3.number_input("Stop loss pips", 5, 200, int(mode_settings.get("sl_pips") or 20), 5)
                tp_cfg = c4.number_input("Take profit pips", 5, 300, int(mode_settings.get("tp_pips") or 40), 5)
                c5, c6 = st.columns(2)
                units_cfg = c5.number_input("Default units", 100, 100000, int(mode_settings.get("units") or 1000), 100)
                max_pos_cfg = c6.number_input("Max open positions", 1, 10, int(mode_settings.get("max_positions") or 3), 1)
                regime_cfg = st.checkbox("Use regime filter", value=bool(mode_settings.get("use_regime_filter", True)))
                auto_enabled = bool(can_auto and mode == "auto" and selected_acc)

                if st.form_submit_button("Save Trading Mode", type="primary"):
                    if mode in ("manual", "auto") and not can_auto:
                        st.error("Manual and auto trading require Pro or Enterprise.")
                    elif mode in ("manual", "auto") and not selected_acc:
                        st.error("Connect a trading account first.")
                    else:
                        try:
                            api_post("/account/settings", json={"mode": mode if can_auto else "signals_only", "auto_trade_enabled": auto_enabled, "trading_account_id": selected_acc, "threshold": threshold_cfg, "risk_pct": risk_cfg, "sl_pips": sl_cfg, "tp_pips": tp_cfg, "units": units_cfg, "max_positions": max_pos_cfg, "use_regime_filter": regime_cfg})
                            st.success("Trading mode saved."); st.rerun()
                        except ApiError as e: st.error(e.detail)

            _section("Subscription Plan")
            PLAN_DETAILS = {
                "free":  {"price":"$0/month",  "color":C_MUTED,   "desc":"Get started with 1 pair and basic signals.", "features":["1 pair (EUR/USD)","Daily signals","Basic price charts","Email support"]},
                "basic": {"price":"$9/month",  "color":C_ACCENT2, "desc":"More pairs and deeper analysis.",            "features":["2 pairs","Daily signals","Full charts","Walk-forward results","Email support"]},
                "pro":   {"price":"$29/month", "color":C_YELLOW,  "desc":"Full platform with live trading.",           "features":["All 4 pairs","Auto-trading","Order placement & management","Trade history","Telegram alerts","Priority support"]},
                "enterprise": {"price":"Custom","color":C_ACCENT, "desc":"Custom setup for institutions.",             "features":["Unlimited pairs","API access","Custom models","Dedicated support"]},
            }
            for pname, pinfo in PLAN_DETAILS.items():
                _plan_card(pname, pinfo, plan == pname)

            if plan not in ("pro","enterprise"):
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⬆ Upgrade to Pro — $29/month", type="primary"):
                    st.info("📧 Contact **admin@forexchautari.com** to upgrade your plan, or use the payment link provided by your admin.")


# ── Helper to build client from stored account ────────────────────────────────

def OandaClient_for_acc(acc: dict):
    from src.oanda_client import OandaClient
    return OandaClient(
        api_key=acc["api_key_enc"],
        account_id=acc["account_id"],
        environment=acc["environment"],
    )
