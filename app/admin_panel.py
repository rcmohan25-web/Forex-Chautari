"""
ForexChautari — Admin Panel v2
All backend features exposed.

Tabs:
  📊 Overview     — platform KPIs, signal log, plan distribution
  👥 Users        — full CRUD, plan management, per-user account viewer
  🌐 Portfolio    — all-pair signals, regime, model health
  ⚡ Trading      — admin Oanda account, place/close trades, live prices,
                    position sizing, risk metrics, all-user trade log
  🔔 Notifications — system notifications, audit log
  🖥 System       — API health, retrain, fetch, run commands
"""

import os, sys, json, time, requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.auth import render_logout_button
from app.api_client import api_get, api_delete, ApiError
from src.database import (
    get_all_users, update_user_plan, deactivate_user, reactivate_user,
    get_signals_log, get_audit_log, get_platform_stats,
    get_user_trades, get_trade_stats, get_notifications,
    mark_notifications_read, create_user,
    get_trading_accounts, add_trading_account, remove_trading_account,
    log_trade, close_trade as db_close_trade,
    get_user_trading_settings, update_user_trading_settings,
    get_platform_settings, update_platform_settings, setting_bool,
    has_plaintext_api_keys, plaintext_api_keys_count,
)
from config.settings import (
    ACTIVE_PAIRS, meta_path, data_path, DEFAULT_SIGNAL_THRESHOLD,
    APP_BRAND, APP_NAME,
)

API_BASE = "http://127.0.0.1:8000"

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG     = "#070b13"; C_SURF   = "#0d1520"; C_SURF2  = "#111e2d"
C_CARD   = "#0f1a28"; C_BORDER = "#1a2d42"; C_ACCENT = "#00d4a8"
C_ACCENT2= "#0891b2"; C_RED    = "#f43f5e"; C_YELLOW = "#eab308"
C_GREEN  = "#22c55e"; C_ORANGE = "#f97316"; C_TEXT   = "#e2eaf4"
C_MUTED  = "#4a6580"; C_DIM    = "#1e3348"

PLOT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#08111c",
    font=dict(family="'IBM Plex Mono',monospace", color=C_TEXT, size=11),
    xaxis=dict(gridcolor=C_BORDER, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=C_BORDER, showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER),
    margin=dict(l=8, r=8, t=36, b=8),
    hovermode="x unified",
)
def pf(fig, h=320): fig.update_layout(**PLOT, height=h); return fig

# Prefer the remote API client for admin actions — use the JWT-authenticated
# client so callers can rely on ApiError exceptions.
from app.api_client import api_get, api_post as _api_post_remote, api_delete as _api_delete_remote, ApiError as _ApiError
api_post = _api_post_remote
api_delete = _api_delete_remote
ApiError = _ApiError

def _kpi(label, value, cls="", sub=""):
    sc = {"g":C_GREEN,"r":C_RED,"y":C_YELLOW,"a":C_ACCENT,"m":C_MUTED}.get(cls, C_TEXT)
    sub_html = f'<div style="font-size:10px;color:{C_MUTED};margin-top:2px;">{sub}</div>' if sub else ""
    return f"""<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:12px;padding:14px 16px;margin-bottom:10px;">
<div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;">{label}</div>
<div style="font-size:22px;font-weight:700;color:{sc};font-family:'IBM Plex Mono',monospace;">{value}</div>{sub_html}</div>"""

def _sec(title):
    st.markdown(f'<div style="font-size:10px;color:{C_MUTED};letter-spacing:3px;text-transform:uppercase;margin:20px 0 10px;padding-bottom:6px;border-bottom:1px solid {C_BORDER};">{title}</div>', unsafe_allow_html=True)

def _feature_row(name, status, detail):
    color = C_GREEN if status == "Enabled" else (C_YELLOW if status == "Partial" else C_MUTED)
    return {
        "Feature": name,
        "Status": status,
        "Detail": detail,
        "_color": color,
    }

def _css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Syne+Mono&family=Space+Grotesk:wght@500;600;700&display=swap');
    html,body,[class*="css"]{{font-family:'Space Grotesk',sans-serif;background:{C_BG};color:{C_TEXT};}}
    section[data-testid="stSidebar"]{{background:{C_SURF}!important;border-right:1px solid {C_BORDER};}}
    section[data-testid="stSidebar"] *{{color:{C_TEXT}!important;}}
    button[data-baseweb="tab"]{{font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;letter-spacing:1.5px!important;color:{C_MUTED}!important;text-transform:uppercase!important;padding:12px 16px!important;}}
    button[data-baseweb="tab"][aria-selected="true"]{{color:{C_ACCENT}!important;border-bottom:2px solid {C_ACCENT}!important;}}
    div[data-baseweb="tab-list"]{{background:transparent!important;border-bottom:1px solid {C_BORDER}!important;gap:0!important;}}
    .stButton>button{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT}!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;font-size:11px!important;transition:all .15s!important;}}
    .stButton>button:hover{{border-color:{C_ACCENT}!important;color:{C_ACCENT}!important;}}
    .stButton>button[kind="primary"]{{background:{C_ACCENT}!important;color:#000!important;border-color:{C_ACCENT}!important;font-weight:700!important;}}
    .stTextInput input,.stNumberInput input{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;color:{C_TEXT}!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;font-size:13px!important;}}
    .stSelectbox>div>div{{background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;border-radius:8px!important;}}
    div[data-testid="stDataFrame"]{{border:1px solid {C_BORDER}!important;border-radius:10px!important;overflow:hidden;}}
    .stAlert{{border-radius:10px!important;font-size:12px!important;}}
    #MainMenu,footer,header{{visibility:hidden;}}
    .block-container{{padding-top:0.6rem!important;max-width:1480px;}}
    </style>""", unsafe_allow_html=True)


def _build_client(acc: dict):
    from src.oanda_client import OandaClient
    return OandaClient(
        api_key=acc["api_key_enc"],
        account_id=acc["account_id"],
        environment=acc["environment"],
    )


def _render_first_run_wizard(user: dict):
    """
    Blocks the entire admin panel and forces a password change.
    Only dismissed once the new password is saved and verified.
    """
    from src.database import (
        update_user_password, verify_password,
        get_db,
    )

    # Sidebar is minimal during setup
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Syne Mono,monospace;font-size:15px;'
            f'color:{C_ACCENT};font-weight:700;padding:14px 0 4px;">⬡ {APP_BRAND}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:11px;color:{C_RED};font-weight:700;'
            f'letter-spacing:1px;">⚠ SETUP REQUIRED</div>',
            unsafe_allow_html=True,
        )
        render_logout_button()

    # ── Warning banner ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a0308,#1f040c);
         border:2px solid {C_RED};border-radius:14px;
         padding:24px 30px;margin-bottom:24px;">
      <div style="font-family:'Syne Mono',monospace;font-size:22px;
           font-weight:700;color:{C_RED};margin-bottom:8px;">
        ⚠ First-Run Security Setup
      </div>
      <div style="font-size:13px;color:{C_TEXT};line-height:1.9;">
        The admin account is using the <b style="color:{C_RED};">default password
        'admin123'</b>.<br>
        <b>All platform features are locked</b> until you set a strong password.<br>
        This is enforced at both the API and UI layers and cannot be skipped.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Password change form ───────────────────────────────────────────────────
    st.markdown(
        f'<div style="max-width:480px;margin:0 auto;">',
        unsafe_allow_html=True,
    )

    with st.form("first_run_password_form", clear_on_submit=True):
        st.markdown(
            f'<div style="font-size:10px;color:{C_MUTED};letter-spacing:3px;'
            f'text-transform:uppercase;margin-bottom:16px;">Set Admin Password</div>',
            unsafe_allow_html=True,
        )
        current_pw = st.text_input(
            "Current password",
            type="password",
            placeholder="admin123",
            help="Enter the current default password to confirm your identity.",
        )
        new_pw = st.text_input(
            "New password",
            type="password",
            placeholder="Minimum 12 characters",
        )
        confirm_pw = st.text_input(
            "Confirm new password",
            type="password",
            placeholder="Repeat new password",
        )
        submitted = st.form_submit_button("Set Password & Continue", type="primary")

    if submitted:
        errors = []

        # Verify the current password first — prevents a logged-in session
        # from being hijacked to change the password without knowing it.
        with get_db() as conn:
            row = conn.execute(
                text("SELECT password_hash, salt FROM users WHERE username=:username"),
                {"username": "admin"}
            ).mappings().fetchone()
        if not row or not verify_password(current_pw, row["salt"], row["password_hash"]):
            errors.append("Current password is incorrect.")

        if len(new_pw) < 12:
            errors.append("New password must be at least 12 characters.")

        if new_pw == "admin123":
            errors.append("You cannot reuse the default password.")

        if new_pw != confirm_pw:
            errors.append("New passwords do not match.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            update_user_password(user["id"], new_pw)
            st.success(
                "✅ Password updated. The platform is now fully operational. "
                "Reloading…"
            )
            # Give the user a moment to read the message, then reload
            import time as _time
            _time.sleep(1.5)
            st.rerun()

    # Validation hints shown below the form
    st.markdown(f"""
    <div style="background:{C_SURF};border:1px solid {C_BORDER};border-radius:10px;
         padding:14px 18px;margin-top:16px;font-size:12px;color:{C_MUTED};line-height:2;">
      <b style="color:{C_TEXT};">Password requirements</b><br>
      ✓ &nbsp;Minimum 12 characters<br>
      ✓ &nbsp;Cannot be <code>admin123</code><br>
      ✓ &nbsp;Must match in both fields<br><br>
      <b style="color:{C_TEXT};">What is blocked until setup completes</b><br>
      🔒 &nbsp;All API endpoints except <code>/health</code><br>
      🔒 &nbsp;Admin panel tabs (Overview, Users, Trading, etc.)<br>
      🔒 &nbsp;User dashboard logins still work normally
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_admin(user: dict):
    _css()

    # ── First-run setup wizard ─────────────────────────────────────────────────
    # Shown instead of the full admin panel when the default password is still set.
    # No other tab is rendered until the password is changed.
    from src.database import is_admin_password_default
    if is_admin_password_default():
        _render_first_run_wizard(user)
        return          # hard stop — nothing below this line runs

    # ── Plaintext API key warning banner ──────────────────────────────────────
    if has_plaintext_api_keys():
        count = plaintext_api_keys_count()
        st.warning(
            f"⚠️ **Security: {count} trading account(s) have plaintext API keys.**  "
            f"Anyone with read access to the SQLite file can steal Oanda credentials.  \n\n"
            f"Run the migration (takes < 5 seconds):  \n"
            f"```\npython scripts/migrate_encrypt_keys.py --dry-run\n"
            f"python scripts/migrate_encrypt_keys.py\n```  \n"
            f"See the Deployment Checklist in README.md for full instructions.",
            icon="🔑",
        )

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:14px 0 10px;">
          <div style="font-family:'Syne Mono',monospace;font-size:15px;color:{C_ACCENT};font-weight:700;">⬡ {APP_BRAND}</div>
          <div style="font-size:10px;color:{C_MUTED};margin-top:3px;letter-spacing:1px;">ADMIN CONSOLE</div>
          <div style="margin-top:8px;display:inline-block;padding:2px 10px;border-radius:4px;
               font-size:10px;font-weight:700;letter-spacing:1px;
               background:rgba(0,212,168,.15);color:{C_ACCENT};border:1px solid {C_ACCENT}44;">ADMIN</div>
          <div style="margin-top:8px;font-size:13px;font-weight:600;color:{C_TEXT};">{user.get("full_name","Administrator")}</div>
        </div>
        <hr style="border-color:{C_BORDER};margin:10px 0;">
        """, unsafe_allow_html=True)

        health_data, h_err, h_ms = api_get("/health")
        if h_err:
            st.markdown(f'<div style="font-size:11px;color:{C_RED};">● API OFFLINE</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:11px;color:{C_GREEN};">● API ONLINE · {h_ms}ms</div>', unsafe_allow_html=True)

        st.markdown(f'<hr style="border-color:{C_BORDER};margin:10px 0;">', unsafe_allow_html=True)
        render_logout_button()

    # ── Header ─────────────────────────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d  %H:%M")
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{C_SURF},{C_SURF2});border:1px solid {C_BORDER};
         border-radius:14px;padding:18px 26px;display:flex;justify-content:space-between;
         align-items:center;margin-bottom:18px;">
      <div>
        <div style="font-family:'Syne Mono',monospace;font-size:20px;color:{C_ACCENT};font-weight:700;">
          ⬡ Admin Console — {APP_BRAND}
        </div>
        <div style="font-size:12px;color:{C_MUTED};margin-top:3px;">{APP_NAME} · Full Access</div>
      </div>
      <div style="text-align:right;font-size:11px;color:{C_MUTED};font-family:'IBM Plex Mono',monospace;">{now}</div>
    </div>""", unsafe_allow_html=True)
    _, logout_col = st.columns([8, 1])
    with logout_col:
        render_logout_button(sidebar=False, key="admin_logout_top")

    # ── Tabs ───────────────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "  📊  OVERVIEW  ", "  👥  USERS  ",
        "  🌐  PORTFOLIO  ", "  ⚡  TRADING  ",
        "  🔔  NOTIFICATIONS  ", "  🖥  SYSTEM  ",
    ])

    # ══ TAB 1 — Overview ══════════════════════════════════════════════════════
    with t1:
        stats       = get_platform_stats()
        trade_stats = get_trade_stats(user_id=None)
        plans       = stats.get("plans", {})

        k1,k2,k3,k4,k5,k6 = st.columns(6)
        k1.markdown(_kpi("Total Users",   str(stats["total_users"]),  "a"), unsafe_allow_html=True)
        k2.markdown(_kpi("New Today",     str(stats.get("new_today",0)), "y"), unsafe_allow_html=True)
        k3.markdown(_kpi("Total Trades",  str(stats["total_trades"]), "m"), unsafe_allow_html=True)
        k4.markdown(_kpi("Total P&L",     f"${stats.get('total_pnl',0):+,.2f}",
                          "g" if stats.get("total_pnl",0)>=0 else "r"), unsafe_allow_html=True)
        k5.markdown(_kpi("Win Rate",      f"{trade_stats.get('win_rate',0):.1f}%", "y"), unsafe_allow_html=True)
        k6.markdown(_kpi("Signals",       str(stats["total_signals"]), "a"), unsafe_allow_html=True)

        _sec("Admin Feature Coverage")
        feature_rows = [
            _feature_row("Users and plans", "Enabled", "Create users, change plans, deactivate/reactivate"),
            _feature_row("User trading accounts", "Enabled", "Inspect linked Oanda accounts and trading mode"),
            _feature_row("All-pair signals", "Enabled", "EUR/USD, GBP/USD, USD/JPY, AUD/USD"),
            _feature_row("Model training", "Enabled", "Fetch data and retrain one/all models"),
            _feature_row("Admin trading", "Enabled", "Manual, limit, signal+trade, close, modify SL/TP"),
            _feature_row("Scheduled automation", "Enabled", "Auto-fetch, auto-train, user auto-trading toggles"),
            _feature_row("Risk controls", "Enabled", "Risk sizing, max positions, account risk metrics"),
        ]
        st.dataframe(pd.DataFrame(feature_rows).drop(columns=["_color"]), width="stretch", hide_index=True, height=260)

        ov1, ov2 = st.columns(2, gap="large")
        with ov1:
            _sec("Recent Signals")
            sigs = get_signals_log(limit=20)
            if sigs:
                df_s = pd.DataFrame(sigs)
                show = [c for c in ["created_at","pair","signal","confidence","tradeable","price"] if c in df_s.columns]
                st.dataframe(df_s[show], width="stretch", hide_index=True, height=260)
            else:
                st.info("No signals yet.")

        with ov2:
            _sec("Plan Distribution")
            if plans:
                fig_pie = go.Figure(go.Pie(
                    labels=list(plans.keys()), values=list(plans.values()),
                    hole=0.55,
                    marker_colors=[C_ACCENT, C_ACCENT2, C_YELLOW, C_RED],
                ))
                pf(fig_pie, 220)
                fig_pie.update_layout(showlegend=True)
                st.plotly_chart(fig_pie, width="stretch")

            if sigs:
                df_s2 = pd.DataFrame(sigs)
                if "pair" in df_s2.columns:
                    pc = df_s2["pair"].value_counts()
                    fig_bar = go.Figure(go.Bar(x=pc.index, y=pc.values, marker_color=C_ACCENT2))
                    pf(fig_bar, 180)
                    fig_bar.update_layout(showlegend=False, title="Signals by Pair",
                                           title_font=dict(color=C_MUTED, size=11))
                    st.plotly_chart(fig_bar, width="stretch")

    # ══ TAB 2 — Users ═════════════════════════════════════════════════════════
    with t2:
        users = get_all_users()

        _sec("All Users")
        if users:
            u_df = pd.DataFrame(users)
            show = [c for c in ["id","username","email","full_name","role","plan","is_active","last_login","created_at"] if c in u_df.columns]
            st.dataframe(u_df[show], width="stretch", hide_index=True, height=240)
        else:
            st.info("No users.")

        # User selector for management
        non_admin = [u for u in users if u.get("role") != "admin"]
        if non_admin:
            _sec("Manage User")
            m1,m2,m3,m4 = st.columns(4)
            with m1:
                target    = st.selectbox("Select user", [u["username"] for u in non_admin], key="mgmt_user")
                target_obj= next((u for u in users if u["username"]==target), None)
                target_id = target_obj["id"] if target_obj else None
                is_active = bool(target_obj.get("is_active",1)) if target_obj else True
            with m2:
                new_plan = st.selectbox("Change plan", ["free","basic","pro","enterprise"], key="new_plan")
            with m3:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("✓ Update Plan") and target_id:
                    try:
                        api_post(f"/admin/users/{target_id}/plan", json={"plan": new_plan})
                        st.success(f"Plan → {new_plan}")
                        st.rerun()
                    except ApiError as e:
                        st.error(e.detail)
            with m4:
                st.markdown("<br>", unsafe_allow_html=True)
                if is_active:
                    if st.button("✗ Deactivate") and target_id:
                        try:
                            api_post(f"/admin/users/{target_id}/deactivate", json={})
                            st.warning("Deactivated.")
                            st.rerun()
                        except ApiError as e:
                            st.error(e.detail)
                else:
                    if st.button("✓ Reactivate") and target_id:
                        try:
                            api_post(f"/admin/users/{target_id}/reactivate", json={})
                            st.success("Reactivated.")
                            st.rerun()
                        except ApiError as e:
                            st.error(e.detail)

            # Per-user trading accounts viewer
            if target_id:
                _sec(f"Trading Accounts — {target}")
                user_accs = get_trading_accounts(target_id)
                if user_accs:
                    for acc in user_accs:
                        env_c = C_YELLOW if acc["environment"]=="live" else C_ACCENT2
                        st.markdown(f"""
                        <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;
                             padding:12px 16px;margin:4px 0;font-size:12px;">
                          <b style="color:{C_TEXT};">{acc["account_name"]}</b> &nbsp;·&nbsp;
                          <code style="color:{C_TEXT};">{acc["account_id"]}</code> &nbsp;·&nbsp;
                          <span style="color:{env_c};">{acc["environment"].upper()}</span> &nbsp;·&nbsp;
                          <span style="color:{C_MUTED};">added {acc["created_at"][:10]}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info(f"No trading accounts linked to {target}.")

                _sec(f"Trading Mode — {target}")
                try:
                    ts = get_user_trading_settings(target_id)
                    mode_label = {
                        "signals_only": "Signals only",
                        "manual": "Manual trading",
                        "auto": "Auto trade",
                    }.get(ts.get("mode"), ts.get("mode"))
                    st.markdown(f"""
                    <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;
                         padding:12px 16px;margin:4px 0;font-size:12px;line-height:2;">
                      Mode: <b style="color:{C_ACCENT};">{mode_label}</b> &nbsp;·&nbsp;
                      Auto enabled: <b style="color:{C_GREEN if ts.get('auto_trade_enabled') else C_MUTED};">{'YES' if ts.get('auto_trade_enabled') else 'NO'}</b><br>
                      Threshold: <b>{float(ts.get('threshold') or 0):.2f}</b> &nbsp;·&nbsp;
                      Risk: <b>{float(ts.get('risk_pct') or 0)*100:.1f}%</b> &nbsp;·&nbsp;
                      SL/TP: <b>{ts.get('sl_pips')}/{ts.get('tp_pips')} pips</b> &nbsp;·&nbsp;
                      Units: <b>{int(ts.get('units') or 0):,}</b>
                    </div>""", unsafe_allow_html=True)
                    if ts.get("auto_trade_enabled"):
                            if st.button("Disable Auto Trade", key=f"disable_auto_{target_id}"):
                                try:
                                    api_post(f"/admin/users/{target_id}/disable-auto-trade", json={})
                                    st.success("Auto-trade disabled for user.")
                                    st.rerun()
                                except ApiError as e:
                                    st.error(e.detail)
                except Exception as e:
                    st.info(f"Trading settings unavailable: {e}")

                # Per-user trade stats
                user_ts = get_trade_stats(target_id)
                if user_ts["total_trades"] > 0:
                    uts1,uts2,uts3 = st.columns(3)
                    uts1.markdown(_kpi("Trades", str(user_ts["total_trades"]), "a"), unsafe_allow_html=True)
                    uts2.markdown(_kpi("P&L",    f"${user_ts['total_pnl']:+,.2f}",
                                       "g" if user_ts["total_pnl"]>=0 else "r"), unsafe_allow_html=True)
                    uts3.markdown(_kpi("Win Rate", f"{user_ts['win_rate']:.1f}%", "y"), unsafe_allow_html=True)

        _sec("Create New User")
        with st.form("admin_create_user"):
            c1,c2,c3 = st.columns(3)
            nu = c1.text_input("Username"); ne = c2.text_input("Email"); nf = c3.text_input("Full name")
            c4,c5,c6 = st.columns(3)
            np_ = c4.text_input("Password"); nr = c5.selectbox("Role",["user","admin"]); npl = c6.selectbox("Plan",["free","basic","pro","enterprise"])
            if st.form_submit_button("Create User", type="primary"):
                if nu and ne and np_:
                    try:
                        api_post("/admin/users", json={
                            "username": nu, "email": ne, "password": np_,
                            "full_name": nf, "role": nr, "plan": npl,
                        })
                        st.success(f"User '{nu}' created."); st.rerun()
                    except ApiError as e:
                        st.error(e.detail)
                    except Exception as ex:
                        st.error(str(ex))
                else:
                    st.error("Username, email and password are required.")

    # ══ TAB 3 — Portfolio ═════════════════════════════════════════════════════
    with t3:
        _sec("Live Portfolio Signals")
        try:
            from src.multi_pair_manager import get_portfolio_signals
            with st.spinner("Loading signals..."):
                port_df = get_portfolio_signals(DEFAULT_SIGNAL_THRESHOLD)

            ok_df = port_df[port_df["ok"]==True] if not port_df.empty else pd.DataFrame()
            if not ok_df.empty:
                cols = st.columns(len(ok_df))
                for idx, (_, row) in enumerate(ok_df.iterrows()):
                    if idx >= len(cols): break
                    is_b = row["signal"]=="BUY"; sc = C_ACCENT if is_b else C_RED
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="background:{C_CARD};border:1px solid {C_BORDER};
                             border-left:4px solid {sc};border-radius:12px;padding:14px;">
                          <div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;">{row["pair"].replace("_","/")}</div>
                          <div style="font-family:'Syne Mono',monospace;font-size:24px;font-weight:700;color:{sc};">
                            {"▲" if is_b else "▼"} {row["signal"]}
                          </div>
                          <div style="font-size:11px;color:{C_MUTED};margin-top:6px;line-height:1.8;">
                            {row["confidence"]} · {row.get("regime","?").upper()}<br>
                            Prob: {row["prob_up"]:.4f}
                          </div>
                        </div>""", unsafe_allow_html=True)
            else:
                st.info("No signals. Run: `python train_all.py --fetch`")

            _sec("Portfolio Table")
            show = [c for c in ["pair","signal","prob_up","confidence","regime","trend","vol_regime","tradeable","wf_accuracy","test_accuracy","latest_date"] if c in port_df.columns]
            st.dataframe(port_df[show], width="stretch", hide_index=True)

        except Exception as e:
            st.error(f"Portfolio error: {e}")

        _sec("Per-Pair Model Health")
        pair_rows = []
        for p in ACTIVE_PAIRS:
            mp = meta_path(p)
            if os.path.exists(mp):
                with open(mp) as f: m = json.load(f)
                gap = m.get("accuracy_train",0) - m.get("accuracy_test",0)
                pair_rows.append({
                    "Pair": p.replace("_","/"),
                    "Test Acc":  f"{m.get('accuracy_test',0):.3f}",
                    "WF Acc":    f"{m.get('walk_forward_mean_accuracy',0):.3f}",
                    "WF PF":     f"{m.get('walk_forward_mean_profit_factor',0):.2f}" if m.get("walk_forward_mean_profit_factor") is not None else "—",
                    "WF Sharpe": f"{m.get('walk_forward_mean_sharpe',0):.2f}" if m.get("walk_forward_mean_sharpe") is not None else "—",
                    "Expectancy":f"{m.get('walk_forward_mean_expectancy',0):+.5f}" if m.get("walk_forward_mean_expectancy") is not None else "—",
                    "Gap":       f"{gap:.3f}",
                    "WF Profit": f"{m.get('walk_forward_profitable_splits',0)}/{m.get('walk_forward_total_splits',0)}",
                    "Rows":      m.get("rows_total","?"),
                    "Trained":   m.get("trained_at","?")[:10],
                    "Status":    "⚠️ Overfit" if gap>0.15 else (
                        "✅ Tradable edge" if (
                            m.get("walk_forward_mean_accuracy",0)>0.51
                            and (m.get("walk_forward_mean_profit_factor") or 0) >= 1.05
                        ) else "🔴 Weak edge"
                    ),
                })
            else:
                pair_rows.append({"Pair":p.replace("_","/"),"Test Acc":"—","WF Acc":"—","WF PF":"—","WF Sharpe":"—","Expectancy":"—","Gap":"—","WF Profit":"—","Rows":"—","Trained":"not trained","Status":"⚠️ Not trained"})
        st.dataframe(pd.DataFrame(pair_rows), width="stretch", hide_index=True)

    # ══ TAB 4 — Trading ═══════════════════════════════════════════════════════
    with t4:
        # ── Platform trade stats ───────────────────────────────────────────────
        trade_stats_all = get_trade_stats(user_id=None)
        ts1,ts2,ts3,ts4,ts5 = st.columns(5)
        ts1.markdown(_kpi("Total Trades",  str(trade_stats_all["total_trades"]), "a"), unsafe_allow_html=True)
        ts2.markdown(_kpi("Closed",        str(trade_stats_all["closed_trades"]), "m"), unsafe_allow_html=True)
        ts3.markdown(_kpi("Total P&L",     f"${trade_stats_all['total_pnl']:+,.2f}",
                           "g" if trade_stats_all["total_pnl"]>=0 else "r"), unsafe_allow_html=True)
        ts4.markdown(_kpi("Win Rate",      f"{trade_stats_all['win_rate']:.1f}%", "y"), unsafe_allow_html=True)
        ts5.markdown(_kpi("Avg P&L",       f"${trade_stats_all['avg_pnl']:+,.2f}",
                           "g" if trade_stats_all["avg_pnl"]>=0 else "r"), unsafe_allow_html=True)

        trade_l, trade_r = st.columns([1,1], gap="large")

        # ── LEFT: Admin Oanda account ──────────────────────────────────────────
        with trade_l:
            _sec("Admin Trading Account")
            admin_accs = get_trading_accounts(user["id"])
            selected_admin_account_id = admin_accs[0]["id"] if admin_accs else None

            if admin_accs:
                account_options = {
                    acc["id"]: f"{acc['account_name']} · {acc['account_id']} · {acc['environment'].upper()}"
                    for acc in admin_accs
                }
                selected_admin_account_id = st.selectbox(
                    "Active admin trading account",
                    options=list(account_options.keys()),
                    format_func=lambda aid: account_options[aid],
                    key="admin_active_account",
                )
                for acc in admin_accs:
                    env_c = C_YELLOW if acc["environment"]=="live" else C_ACCENT2
                    c1, c2, c3 = st.columns([3,2,1])
                    with c1:
                        st.markdown(f"""
                        <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:12px 14px;">
                          <div style="font-size:13px;font-weight:600;color:{C_TEXT};">{acc["account_name"]}</div>
                          <div style="font-size:11px;color:{C_MUTED};margin-top:4px;">
                            {acc["account_id"]} · <span style="color:{env_c};">{acc["environment"].upper()}</span>
                          </div>
                        </div>""", unsafe_allow_html=True)
                    with c2:
                        try:
                            cl = _build_client(acc)
                            sm = cl.get_account_summary()
                            st.markdown(f'<div style="font-size:12px;color:{C_GREEN};padding:14px 0;">✓ ${sm["balance"]:,.2f}</div>', unsafe_allow_html=True)
                        except Exception:
                            st.markdown(f'<div style="font-size:12px;color:{C_RED};padding:14px 0;">✗ Error</div>', unsafe_allow_html=True)
                    with c3:
                        if st.button("Remove", key=f"admin_rm_{acc['id']}"):
                            try:
                                api_delete(f"/account/trading-accounts/{acc['id']}")
                                st.rerun()
                            except ApiError as e:
                                st.error(e.detail)
            else:
                st.info("No admin trading account connected.")

            with st.expander("➕ Connect Admin Oanda Account"):
                with st.form("admin_account_form"):
                    st.markdown(f'<div style="font-size:11px;color:{C_MUTED};margin-bottom:10px;">Connect an Oanda practice or live account for admin trading.</div>', unsafe_allow_html=True)
                    a_name = st.text_input("Label", placeholder="Admin Practice Account")
                    a_key  = st.text_input("API Token", type="password")
                    a_id   = st.text_input("Account ID", placeholder="101-001-XXXXXXX-001")
                    a_env  = st.selectbox("Environment", ["practice","live"])
                    if a_env == "live":
                        st.error("⚠️ LIVE mode uses real money.")
                    if st.form_submit_button("Verify & Connect", type="primary"):
                        if a_name and a_key and a_id:
                            with st.spinner("Verifying..."):
                                try:
                                    from src.oanda_client import OandaClient
                                    cl     = OandaClient(api_key=a_key, account_id=a_id, environment=a_env)
                                    result = cl.validate_credentials()
                                    if result["valid"]:
                                        new_id = add_trading_account(
                                            user["id"], a_name, a_key, a_id, a_env, is_admin=True
                                        )
                                        current_settings = get_user_trading_settings(user["id"])
                                        if not current_settings.get("trading_account_id"):
                                            update_user_trading_settings(user["id"], trading_account_id=new_id)
                                        st.success(f"✅ Connected! Balance: ${result['balance']:,.2f} {result['currency']}")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {result.get('error')}")
                                except Exception as e:
                                    st.error(str(e))
                        else:
                            st.error("All fields required.")

            # ── Live account dashboard ─────────────────────────────────────────
            if admin_accs:
                try:
                    from src.oanda_client import OandaClient
                    from src.trading_engine import calculate_sl_tp, calculate_position_size
                    acc    = next(
                        (a for a in admin_accs if int(a["id"]) == int(selected_admin_account_id)),
                        admin_accs[0],
                    )
                    client = _build_client(acc)
                    summary = client.get_account_summary()
                    upl     = summary["unrealized_pl"]

                    _sec("Account Overview")
                    ak1,ak2,ak3,ak4 = st.columns(4)
                    ak1.markdown(_kpi("Balance",       f"${summary['balance']:,.2f}", "a"), unsafe_allow_html=True)
                    ak2.markdown(_kpi("NAV",           f"${summary['nav']:,.2f}",
                                       "g" if summary["nav"]>=summary["balance"] else "r"), unsafe_allow_html=True)
                    ak3.markdown(_kpi("Unrealized P&L",f"${upl:+,.2f}","g" if upl>=0 else "r"), unsafe_allow_html=True)
                    ak4.markdown(_kpi("Margin Used",   f"${summary['margin_used']:,.2f}", "y"), unsafe_allow_html=True)

                    _sec("Live Market Prices")
                    try:
                        prices = client.get_all_prices(ACTIVE_PAIRS)
                        pc = st.columns(len(prices))
                        for i, p in enumerate(prices):
                            tc = C_GREEN if p.get("tradeable") else C_RED
                            with pc[i]:
                                st.markdown(f"""
                                <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:10px;padding:12px 14px;">
                                  <div style="font-size:10px;color:{C_MUTED};letter-spacing:2px;margin-bottom:4px;">{p["instrument"].replace("_","/")}</div>
                                  <div style="font-size:18px;font-weight:700;color:{C_TEXT};font-family:'IBM Plex Mono',monospace;">{p["mid"]:.5f}</div>
                                  <div style="font-size:11px;color:{C_MUTED};margin-top:3px;">
                                    Spread: <span style="color:{tc};">{p["spread_pips"]:.1f} pips</span>
                                  </div>
                                </div>""", unsafe_allow_html=True)
                    except Exception as e:
                        st.info(f"Live prices unavailable: {e}")

                    # Daily P&L
                    try:
                        daily_pnl = client.get_daily_pnl()
                        ak1b,ak2b = st.columns(2)
                        ak1b.markdown(_kpi("Daily P&L",    f"${daily_pnl:+,.2f}",
                                           "g" if daily_pnl>=0 else "r"), unsafe_allow_html=True)
                        ak2b.markdown(_kpi("Margin Avail.",f"${summary['margin_avail']:,.2f}", "a"), unsafe_allow_html=True)
                    except Exception:
                        pass

                    _sec("Place Trade")
                    tp1,tp2,tp3,tp4,tp5,tp6 = st.columns(6)
                    t_pair  = tp1.selectbox("Pair",       ACTIVE_PAIRS,       key="adm_pair")
                    t_dir   = tp2.selectbox("Direction",  ["BUY","SELL"],      key="adm_dir")
                    t_otype = tp3.selectbox("Order Type", ["Market","Limit"],  key="adm_otype")
                    t_units = tp4.number_input("Units",   100, 100000, 1000, 100, key="adm_units")
                    t_sl    = tp5.number_input("SL pips", 5, 200, 20, 5,       key="adm_sl")
                    t_tp    = tp6.number_input("TP pips", 5, 200, 40, 5,       key="adm_tp")

                    t_risk = st.slider("Risk % of balance", 0.5, 5.0, 1.0, 0.5, key="adm_risk") / 100
                    limit_price_adm = None
                    if t_otype == "Limit":
                        limit_price_adm = st.number_input("Limit Price", value=0.0, step=0.0001,
                                                           format="%.5f", key="adm_limit_px")

                    # Live preview
                    try:
                        lp    = client.get_live_price(t_pair)
                        entry = lp["mid"]
                        sl, tp = calculate_sl_tp(t_pair, t_dir, entry, t_sl, t_tp)
                        auto_units = calculate_position_size(
                            summary["balance"], t_risk, t_sl,
                            pip_value=0.01 if "JPY" in t_pair else 0.0001
                        )
                        st.markdown(f"""
                        <div style="background:{C_SURF2};border:1px solid {C_BORDER};border-radius:10px;
                             padding:12px 16px;margin:10px 0;font-size:12px;font-family:'IBM Plex Mono',monospace;">
                          Entry ~ <b style="color:{C_TEXT};">{entry:.5f}</b> &nbsp;·&nbsp;
                          SL <b style="color:{C_RED};">{sl:.5f}</b> &nbsp;·&nbsp;
                          TP <b style="color:{C_GREEN};">{tp:.5f}</b> &nbsp;·&nbsp;
                          Auto-units <b style="color:{C_YELLOW};">{auto_units:,}</b> &nbsp;·&nbsp;
                          Spread <b style="color:{C_TEXT};">{lp["spread_pips"]:.1f}p</b>
                        </div>""", unsafe_allow_html=True)
                    except Exception:
                        pass

                    btn1, btn2, btn3, btn4 = st.columns(4)
                    if btn1.button(f"▶ Place {t_otype} {t_dir}", type="primary", key="adm_place"):
                        with st.spinner("Placing order..."):
                            try:
                                api_post("/trading/place", json={
                                    "instrument": t_pair,
                                    "direction": t_dir,
                                    "units": t_units,
                                    "sl_pips": float(t_sl),
                                    "tp_pips": float(t_tp),
                                    "order_type": t_otype,
                                    "limit_price": limit_price_adm,
                                    "account_db_id": selected_admin_account_id,
                                })
                                st.success(f"✅ {t_otype} {t_dir} order placed.")
                                st.rerun()
                            except ApiError as e:
                                st.error(e.detail)

                    if btn2.button("🤖 Signal + Trade", key="adm_signal_trade"):
                        with st.spinner("Running ML signal..."):
                            try:
                                result = api_post("/trading/signal-trade", json={
                                    "pair": t_pair,
                                    "threshold": DEFAULT_SIGNAL_THRESHOLD,
                                    "sl_pips": float(t_sl),
                                    "tp_pips": float(t_tp),
                                    "units": t_units,
                                    "account_db_id": selected_admin_account_id,
                                })[0]
                                if result and result.get("action")=="order_placed":
                                    st.success(f"✅ Signal trade: {result['signal']} @ {result.get('price','?')}")
                                elif result and result.get("action")=="error":
                                    st.error(f"Error: {result['reason']}")
                                else:
                                    st.info(f"{result.get('signal','—') if result else '—'} — Not traded")
                                st.rerun()
                            except ApiError as e:
                                st.error(e.detail)

                    if btn3.button("🔴 Close All", key="adm_close_all"):
                        with st.spinner("Closing all positions..."):
                            try:
                                api_post("/trading/close-all", json={
                                    "account_db_id": selected_admin_account_id,
                                })
                                st.success("All positions closed.")
                                st.rerun()
                            except ApiError as e:
                                st.error(e.detail)

                    if btn4.button("⚡ Portfolio Auto Check", key="adm_portfolio_auto"):
                        with st.spinner("Checking all pairs and trading eligible signals..."):
                            try:
                                from src.multi_pair_manager import run_portfolio_signal_check
                                results = run_portfolio_signal_check(
                                    threshold=DEFAULT_SIGNAL_THRESHOLD,
                                    max_positions=3,
                                    user_id=user["id"],
                                    account_db_id=selected_admin_account_id,
                                    default_units=int(t_units),
                                    sl_pips=float(t_sl),
                                    tp_pips=float(t_tp),
                                    use_regime_filter=True,
                                )
                                st.dataframe(pd.DataFrame(results), width="stretch", hide_index=True)
                            except Exception as e:
                                st.error(str(e))

                    # ── Open positions ─────────────────────────────────────────
                    _sec("Open Positions")
                    open_trades = client.get_open_trades()
                    if open_trades:
                        for i, trade in enumerate(open_trades):
                            upl_c = C_GREEN if trade["unrealized_pl"]>=0 else C_RED
                            oc1,oc2,oc3,oc4,oc5,oc6,oc7 = st.columns([2,2,1,1,1,2,1])
                            oc1.markdown(f'<div style="padding:8px 0;font-size:13px;font-weight:600;">{trade["instrument"].replace("_","/")}</div>', unsafe_allow_html=True)
                            oc2.markdown(f'<div style="padding:8px 0;font-size:12px;color:{C_MUTED};">{trade["units"]:+d} @ {trade["open_price"]:.5f}</div>', unsafe_allow_html=True)
                            oc3.markdown(f'<div style="padding:8px 0;font-size:13px;font-weight:600;color:{upl_c};">${trade["unrealized_pl"]:+.2f}</div>', unsafe_allow_html=True)
                            oc4.markdown(f'<div style="padding:8px 0;font-size:11px;color:{C_MUTED};">SL:{trade["stop_loss"]}</div>', unsafe_allow_html=True)
                            oc5.markdown(f'<div style="padding:8px 0;font-size:11px;color:{C_MUTED};">TP:{trade["take_profit"]}</div>', unsafe_allow_html=True)

                            # Modify SL/TP
                            with oc6:
                                new_sl = st.number_input("New SL", value=0.0, step=0.0001, format="%.5f",
                                                          key=f"adm_sl_{trade['trade_id']}_{i}", label_visibility="collapsed")
                                new_tp = st.number_input("New TP", value=0.0, step=0.0001, format="%.5f",
                                                          key=f"adm_tp_{trade['trade_id']}_{i}", label_visibility="collapsed")
                            with oc7:
                                if st.button("Mod", key=f"adm_mod_{trade['trade_id']}_{i}") and (new_sl or new_tp):
                                    try:
                                        api_post("/trading/modify", json={
                                            "trade_id": trade["trade_id"],
                                            "account_db_id": selected_admin_account_id,
                                            "stop_loss": new_sl if new_sl>0 else None,
                                            "take_profit": new_tp if new_tp>0 else None,
                                        })
                                        st.success("Modified")
                                        st.rerun()
                                    except ApiError as e:
                                        st.error(e.detail)
                                if st.button("Close", key=f"adm_close_{trade['trade_id']}_{i}"):
                                    try:
                                        # Get DB trade ID for API call
                                        db_trades = get_user_trades(user["id"], limit=30)
                                        match = next((t for t in db_trades
                                                      if t.get("broker_trade_id")==trade["trade_id"]
                                                      and t["status"]=="open"), None)
                                        db_trade_id = match["id"] if match else None
                                        api_post("/trading/close", json={
                                            "broker_trade_id": trade["trade_id"],
                                            "db_trade_id": db_trade_id,
                                            "account_db_id": selected_admin_account_id,
                                        })
                                        st.success("Trade closed.")
                                        st.rerun()
                                    except ApiError as e:
                                        st.error(e.detail)
                            st.markdown(f'<hr style="border-color:{C_DIM};margin:2px 0;">', unsafe_allow_html=True)
                    else:
                        st.info("No open positions.")

                    # ── Pending orders ─────────────────────────────────────────
                    _sec("Pending Orders")
                    try:
                        pending = client.get_pending_orders()
                        if pending:
                            for i, order in enumerate(pending):
                                po1,po2,po3 = st.columns([3,3,1])
                                po1.write(f"{order['instrument']} {order['type']} @ {order['price']}")
                                po2.write(f"Units: {order['units']}")
                                if po3.button("Cancel", key=f"adm_cancel_{order['order_id']}_{i}"):
                                    try:
                                        client.cancel_order(order["order_id"])
                                        st.success("Cancelled."); st.rerun()
                                    except Exception as e:
                                        st.error(str(e))
                        else:
                            st.info("No pending orders.")
                    except Exception as e:
                        st.info(f"Pending orders: {e}")

                    # ── Transaction history ────────────────────────────────────
                    _sec("Oanda Transaction History")
                    try:
                        txns = client.get_transaction_history(count=50)
                        if txns:
                            st.dataframe(pd.DataFrame(txns), width="stretch", hide_index=True, height=260)
                        else:
                            st.info("No transactions yet.")
                    except Exception as e:
                        st.info(f"Transactions: {e}")

                except Exception as e:
                    st.error(f"Account error: {e}")

        # ── RIGHT: Platform-wide trade log & P&L ──────────────────────────────
        with trade_r:
            _sec("Platform-wide Trade Log")
            all_trades = get_user_trades(user_id=None, limit=200)
            if all_trades:
                tr_df = pd.DataFrame(all_trades)
                st.dataframe(tr_df, width="stretch", hide_index=True, height=280)

                closed = [t for t in all_trades if t.get("pnl") is not None]
                if closed:
                    pnl_vals = [t["pnl"] for t in closed]

                    fig_pnl = go.Figure(go.Bar(
                        y=pnl_vals,
                        marker_color=[C_ACCENT if v>=0 else C_RED for v in pnl_vals],
                    ))
                    fig_pnl.add_hline(y=0, line_color=C_MUTED, line_width=1)
                    pf(fig_pnl, 220)
                    fig_pnl.update_layout(showlegend=False, title="Platform P&L per Trade",
                                           title_font=dict(color=C_MUTED, size=11))
                    st.plotly_chart(fig_pnl, width="stretch")

                    # Pair breakdown
                    _sec("P&L by Pair")
                    pair_pnl = tr_df.dropna(subset=["pnl"]).groupby("pair")["pnl"].agg(["sum","count","mean"]).reset_index()
                    pair_pnl.columns = ["Pair","Total P&L","Trades","Avg P&L"]
                    st.dataframe(pair_pnl, width="stretch", hide_index=True)
            else:
                st.info("No trades logged yet.")

            _sec("Risk Metrics — Admin Account")
            if admin_accs:
                try:
                    from src.trading_engine import get_risk_metrics
                    rm = get_risk_metrics(user["id"], account_db_id=selected_admin_account_id)
                    if "error" not in rm:
                        rm1,rm2 = st.columns(2)
                        rm1.markdown(_kpi("Daily P&L",    f"${rm.get('daily_pnl',0):+,.2f}",
                                           "g" if rm.get("daily_pnl",0)>=0 else "r"), unsafe_allow_html=True)
                        rm2.markdown(_kpi("Account Risk",  f"{rm.get('risk_pct',0):.2f}%",
                                           "g" if rm.get("risk_pct",0)<2 else ("y" if rm.get("risk_pct",0)<5 else "r"),
                                           sub="of balance at risk"), unsafe_allow_html=True)
                        rm1.markdown(_kpi("Margin Avail.", f"${rm.get('margin_available',0):,.2f}", "a"), unsafe_allow_html=True)
                        rm2.markdown(_kpi("Open Trades",   str(rm.get("open_trades",0)), "m"), unsafe_allow_html=True)
                    else:
                        st.warning(f"Risk metrics: {rm['error']}")
                except Exception as e:
                    st.info(f"Risk metrics unavailable: {e}")

    # ══ TAB 5 — Notifications ═════════════════════════════════════════════════
    with t5:
        _sec("System Notifications")
        notifs = get_notifications(user["id"], unread_only=False)
        if notifs:
            if st.button("✓ Mark all read"):
                mark_notifications_read(user["id"])
                st.rerun()
            for n in notifs:
                dot = C_ACCENT if not n["is_read"] else C_MUTED
                icon = {"success":"✅","info":"ℹ️","warning":"⚠️","error":"🔴"}.get(n.get("type","info"),"ℹ️")
                st.markdown(f"""
                <div style="background:{C_CARD};border-left:3px solid {dot};border-radius:8px;
                     padding:12px 16px;margin:6px 0;">
                  <div style="font-size:13px;font-weight:600;color:{C_TEXT};">{icon} {n["title"]}</div>
                  <div style="font-size:12px;color:{C_MUTED};margin-top:4px;">{n["message"]}</div>
                  <div style="font-size:10px;color:{C_DIM};margin-top:6px;font-family:'IBM Plex Mono',monospace;">{n["created_at"][:16]}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No notifications.")

        _sec("Audit Log")
        audit = get_audit_log(limit=100)
        if audit:
            a_df = pd.DataFrame(audit)
            show = [c for c in ["created_at","username","event","detail","ip_address"] if c in a_df.columns]
            st.dataframe(a_df[show], width="stretch", hide_index=True, height=300)
        else:
            st.info("No audit events yet.")

    # ══ TAB 6 — System ════════════════════════════════════════════════════════
    with t6:
        s1,s2 = st.columns(2, gap="large")

        with s1:
            _sec("API Health")
            if h_err:
                st.error(f"API offline: {h_err}")
                st.code("uvicorn app.api:app --reload --host 127.0.0.1 --port 8000")
            else:
                st.json(health_data)

            _sec("Endpoint Latency")
            eps = ["/health","/predict/all","/portfolio/signals","/portfolio/health"]
            lat_rows = []
            for ep in eps:
                _, err, ms = api_get(ep)
                lat_rows.append({"Endpoint":ep,"Latency (ms)":ms,"Status":"🟢" if ms<300 else "🟡"})
            st.dataframe(pd.DataFrame(lat_rows), width="stretch", hide_index=True)

            _sec("Actions")
            a1,a2,a3 = st.columns(3)
            if a1.button("⬇ Fetch All Pairs"):
                with st.spinner("Fetching data..."):
                    try:
                        from src.multi_pair_manager import fetch_all_pairs
                        res = fetch_all_pairs(count=100)
                        for p,r in res.items():
                            st.write(f"{'✅' if r['ok'] else '❌'} {p}: {r.get('rows',r.get('error'))}")
                    except Exception as e:
                        st.error(str(e))
            if a2.button("⚙ Retrain All"):
                with st.spinner("Retraining all pairs (~3 min)..."):
                    try:
                        res = api_post("/retrain", params={"pair": "all"})
                        st.success("Retrain complete")
                        st.code(res.get("output",""))
                    except ApiError as e:
                        st.error(e.detail)
            if a3.button("⬇⚙ Fetch + Retrain"):
                with st.spinner("Fetching fresh data, then retraining all models..."):
                    try:
                        fetch_res = api_post("/fetch-data", params={"pair": "all", "outputsize": "compact"})
                        st.write(fetch_res)
                        res = api_post("/retrain", params={"pair": "all"})
                        st.success("Fetch + retrain complete")
                        st.code(res.get("output",""))
                    except ApiError as e:
                        st.error(e.detail)
                    except Exception as e:
                        st.error(str(e))

            _sec("Automation Settings")
            settings = get_platform_settings()
            with st.form("platform_automation_settings"):
                f1, f2, f3 = st.columns(3)
                auto_fetch = f1.checkbox(
                    "Auto data fetch",
                    value=setting_bool(settings, "auto_fetch_enabled", True),
                )
                auto_train = f2.checkbox(
                    "Auto model training",
                    value=setting_bool(settings, "auto_train_enabled", True),
                )
                auto_trade = f3.checkbox(
                    "Platform auto-trading",
                    value=setting_bool(settings, "auto_trade_enabled", True),
                )
                t1c, t2c, t3c, t4c = st.columns(4)
                fetch_time = t1c.text_input("Fetch time UTC", value=settings.get("fetch_time_utc", "08:05"))
                signal_time = t2c.text_input("Signal check UTC", value=settings.get("signal_check_time_utc", "08:10"))
                summary_time = t3c.text_input("Daily summary UTC", value=settings.get("daily_summary_time_utc", "08:30"))
                train_time = t4c.text_input("Train time UTC", value=settings.get("train_time_utc", "00:01"))
                c1, c2, c3 = st.columns(3)
                fetch_count = c1.number_input("Fetch candles", 50, 5000, int(settings.get("fetch_count", 100) or 100), 50)
                train_weekday = c2.selectbox(
                    "Train weekday UTC",
                    options=list(range(7)),
                    index=int(settings.get("train_weekday_utc", 0) or 0),
                    format_func=lambda i: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][i],
                )
                min_pf = c3.number_input(
                    "Min profit factor",
                    0.50, 5.00,
                    float(settings.get("minimum_profit_factor", 1.05) or 1.05),
                    0.05,
                )
                min_acc = st.slider(
                    "Minimum walk-forward accuracy for tradable status",
                    0.45, 0.65,
                    float(settings.get("minimum_wf_accuracy", 0.51) or 0.51),
                    0.01,
                )
                if st.form_submit_button("Save Automation Settings", type="primary"):
                    update_platform_settings({
                        "auto_fetch_enabled": int(auto_fetch),
                        "auto_train_enabled": int(auto_train),
                        "auto_trade_enabled": int(auto_trade),
                        "fetch_time_utc": fetch_time.strip() or "08:05",
                        "signal_check_time_utc": signal_time.strip() or "08:10",
                        "daily_summary_time_utc": summary_time.strip() or "08:30",
                        "train_time_utc": train_time.strip() or "00:01",
                        "fetch_count": int(fetch_count),
                        "train_weekday_utc": int(train_weekday),
                        "minimum_profit_factor": float(min_pf),
                        "minimum_wf_accuracy": float(min_acc),
                    })
                    st.success("Automation settings saved. Restart the scheduler for changed times to take effect.")
                    st.rerun()

        with s2:
            _sec("Run Commands")
            st.markdown(f"""
            <div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:12px;
                 padding:16px 18px;font-size:12px;font-family:'IBM Plex Mono',monospace;line-height:2.4;">
              <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;">API SERVER</div>
              <code style="color:{C_ACCENT};">uvicorn app.api:app --reload</code><br>
              <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin-top:8px;">DASHBOARD</div>
              <code style="color:{C_ACCENT};">streamlit run app/main.py</code><br>
              <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin-top:8px;">TRAIN ALL PAIRS</div>
              <code style="color:{C_ACCENT};">python train_all.py --fetch</code><br>
              <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin-top:8px;">SCHEDULER</div>
              <code style="color:{C_ACCENT};">python run_scheduler.py</code><br>
              <div style="color:{C_MUTED};font-size:10px;letter-spacing:1px;margin-top:8px;">TESTS</div>
              <code style="color:{C_ACCENT};">python tests/test_all.py</code>
            </div>""", unsafe_allow_html=True)

            _sec("API Endpoints Reference")
            endpoints = [
                ("GET", "/health",           "System liveness"),
                ("GET", "/predict/latest",   "Single pair signal"),
                ("GET", "/predict/all",      "All pairs signals"),
                ("GET", "/portfolio/signals","Ranked portfolio"),
                ("GET", "/portfolio/health", "Model health all pairs"),
                ("GET", "/model-info",       "Model metadata"),
                ("POST","/retrain",          "Retrain models"),
                ("POST","/fetch-data",       "Fetch candles"),
                ("GET", "/history",          "OHLC history"),
                ("GET", "/walk-forward",     "WF results"),
            ]
            ep_rows = "".join([
                f'<div style="display:flex;gap:10px;padding:6px 0;border-bottom:1px solid {C_DIM};font-size:11px;">'
                f'<span style="color:{C_YELLOW};flex:0 0 44px;font-family:IBM Plex Mono,monospace;">{m}</span>'
                f'<span style="color:{C_ACCENT};flex:0 0 170px;font-family:IBM Plex Mono,monospace;">{p}</span>'
                f'<span style="color:{C_MUTED};">{d}</span></div>'
                for m,p,d in endpoints
            ])
            st.markdown(f'<div style="background:{C_CARD};border:1px solid {C_BORDER};border-radius:12px;padding:14px 16px;">{ep_rows}</div>', unsafe_allow_html=True)
