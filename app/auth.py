"""
Authentication UI components for ForexChautari.
Renders login, register, and forgot-password forms in Streamlit.
Manages session state via st.session_state.
"""

import streamlit as st
from src.database import (
    init_db, authenticate_user, create_user, get_session,
    create_session, destroy_session, get_user_by_id,
)

# ── Colour tokens (match dashboard palette) ───────────────────────────────────
C_ACCENT = "#00c896"
C_RED    = "#f43f5e"
C_SURF   = "#0f1623"
C_SURF2  = "#141d2b"
C_BORDER = "#1e2d42"
C_TEXT   = "#dce8f5"
C_MUTED  = "#546e8a"


def _auth_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html,body,[class*="css"]{{background:#080c14;color:{C_TEXT};font-family:'JetBrains Mono',monospace;}}
    section[data-testid="stSidebar"]{{display:none!important;}}
    #MainMenu,footer,header{{visibility:hidden;}}
    .block-container{{max-width:460px;margin:0 auto;padding-top:4rem!important;}}

    .brand{{text-align:center;margin-bottom:2rem;}}
    .brand-name{{font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{C_ACCENT};letter-spacing:-1px;}}
    .brand-sub{{font-size:11px;color:{C_MUTED};letter-spacing:2px;text-transform:uppercase;margin-top:4px;}}

    .auth-card{{background:{C_SURF};border:1px solid {C_BORDER};border-radius:16px;padding:2rem;}}
    .auth-title{{font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:4px;color:{C_TEXT};}}
    .auth-sub{{font-size:12px;color:{C_MUTED};margin-bottom:1.5rem;}}

    .plan-card{{background:{C_SURF2};border:1px solid {C_BORDER};border-radius:12px;padding:14px 16px;margin:6px 0;cursor:pointer;transition:border-color .15s;}}
    .plan-card:hover,.plan-card.selected{{border-color:{C_ACCENT};}}
    .plan-name{{font-weight:600;font-size:13px;color:{C_TEXT};}}
    .plan-price{{font-size:11px;color:{C_MUTED};}}
    .plan-features{{font-size:11px;color:{C_MUTED};margin-top:4px;}}

    .stTextInput input,.stSelectbox select{{
        background:{C_SURF2}!important;border:1px solid {C_BORDER}!important;
        color:{C_TEXT}!important;border-radius:8px!important;
        font-family:'JetBrains Mono',monospace!important;font-size:13px!important;
    }}
    .stTextInput input:focus{{border-color:{C_ACCENT}!important;box-shadow:0 0 0 2px {C_ACCENT}22!important;}}

    .stButton>button{{
        background:{C_ACCENT}!important;color:#000!important;border:none!important;
        border-radius:10px!important;font-weight:700!important;font-size:13px!important;
        letter-spacing:.5px!important;padding:10px!important;width:100%!important;
        font-family:'JetBrains Mono',monospace!important;transition:opacity .15s!important;
    }}
    .stButton>button:hover{{opacity:.88!important;}}

    .link-btn{{background:transparent!important;color:{C_ACCENT}!important;
        border:none!important;font-size:12px!important;padding:4px 0!important;
        text-decoration:underline!important;width:auto!important;font-weight:400!important;}}
    div[data-testid="stAlert"]{{border-radius:10px!important;font-size:12px!important;}}
    .divider{{text-align:center;color:{C_MUTED};font-size:11px;margin:12px 0;}}
    </style>
    """, unsafe_allow_html=True)


def render_brand():
    st.markdown("""
    <div class="brand">
        <div class="brand-name">⬡ ForexChautari</div>
        <div class="brand-sub">Forex Market Prediction &amp; Analysis ML Model</div>
    </div>
    """, unsafe_allow_html=True)


PLANS = {
    "free":  {"label": "Free",  "price": "$0/mo",   "features": "1 pair · Signals only · No auto-trade"},
    "basic": {"label": "Basic", "price": "$9/mo",    "features": "2 pairs · Signals only · No auto-trade"},
    "pro":   {"label": "Pro",   "price": "$29/mo",   "features": "4 pairs · Auto-trade · Priority support"},
}


def _get_query_token() -> str | None:
    try:
        token = st.query_params.get("session")
        if isinstance(token, list):
            return token[0] if token else None
        return token
    except Exception:
        return None


def _set_query_token(token: str):
    try:
        st.query_params["session"] = token
    except Exception:
        pass


def _clear_query_token():
    try:
        if "session" in st.query_params:
            del st.query_params["session"]
    except Exception:
        pass


def _user_session_payload(user_id: int) -> dict | None:
    row = get_user_by_id(user_id)
    if not row:
        return None
    return {
        "id":         row["id"],
        "username":   row["username"],
        "email":      row["email"],
        "full_name":  row.get("full_name") or "",
        "phone":      row.get("phone") or "",
        "role":       row["role"],
        "plan":       row.get("plan") or "free",
        "auto_trade": bool(row.get("auto_trade")),
        "max_pairs":  row.get("max_pairs") or 1,
    }


def render_login():
    _auth_css()
    render_brand()

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Sign in</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Welcome back to ForexChautari</div>', unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In")

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            user = authenticate_user(username.strip(), password)
            if user:
                token = create_session(user["id"], user["role"])
                st.session_state["token"]    = token
                st.session_state["user"]     = user
                st.session_state["auth_view"] = "app"
                _set_query_token(token)
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.markdown('<div class="divider">— or —</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    if col1.button("Create account", key="goto_register"):
        st.session_state["auth_view"] = "register"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Warn operators who land on the login page that setup is pending.
    # Regular users do not need to see this.
    try:
        from src.database import is_admin_password_default
        if is_admin_password_default():
            st.warning(
                "⚠️ **Admin setup required.** "
                "The platform is running with the default admin password. "
                "Sign in as admin to complete first-run setup.",
                icon="🔒",
            )
    except Exception:
        pass  # Never break the login page due to a DB check failure


def render_register():
    _auth_css()
    render_brand()

    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Create account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Join ForexChautari — free to start</div>', unsafe_allow_html=True)

    with st.form("register_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        full_name = col1.text_input("Full name", placeholder="Mohan Rokaya")
        username  = col2.text_input("Username",  placeholder="mohan123")
        email     = st.text_input("Email", placeholder="you@email.com")
        phone     = st.text_input("Phone (optional)", placeholder="+977 98...")
        password  = st.text_input("Password", type="password", placeholder="Min 8 characters")
        confirm   = st.text_input("Confirm password", type="password", placeholder="Repeat password")

        st.markdown("**Choose a plan:**")
        plan = st.selectbox(
            "Plan",
            options=list(PLANS.keys()),
            format_func=lambda p: f"{PLANS[p]['label']} — {PLANS[p]['price']} — {PLANS[p]['features']}",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Create Account")

    if submitted:
        errors = []
        if not all([full_name, username, email, password]):
            errors.append("Full name, username, email and password are required.")
        if len(password) < 8:
            errors.append("Password must be at least 8 characters.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if "@" not in email:
            errors.append("Please enter a valid email address.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                user_data = create_user(
                    username=username.strip(),
                    email=email.strip(),
                    password=password,
                    full_name=full_name.strip(),
                    phone=phone.strip(),
                    plan=plan,
                )
                user = authenticate_user(username.strip(), password)
                token = create_session(user["id"], user["role"])
                st.session_state["token"]     = token
                st.session_state["user"]      = user
                st.session_state["auth_view"] = "app"
                _set_query_token(token)
                st.success(f"Welcome to ForexChautari, {full_name.split()[0]}!")
                st.rerun()
            except Exception as e:
                if "UNIQUE constraint" in str(e):
                    st.error("Username or email already taken. Try a different one.")
                else:
                    st.error(f"Registration failed: {e}")

    st.markdown('<div class="divider">— already have an account? —</div>', unsafe_allow_html=True)
    if st.button("Sign in instead", key="goto_login"):
        st.session_state["auth_view"] = "login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def get_current_user() -> dict | None:
    """Return the current logged-in user from session state."""
    token = st.session_state.get("token")
    if not token:
        token = _get_query_token()
        if token:
            st.session_state["token"] = token
        else:
            return None
    session = get_session(token)
    if not session:
        st.session_state.pop("token", None)
        st.session_state.pop("user", None)
        _clear_query_token()
        return None
    user = _user_session_payload(session["user_id"])
    if not user:
        st.session_state.pop("token", None)
        st.session_state.pop("user", None)
        _clear_query_token()
        return None
    st.session_state["user"] = user
    return user


def require_auth() -> dict:
    """
    Call at the top of any page that requires authentication.
    Renders login/register if not authenticated.
    Returns the user dict if authenticated.
    """
    init_db()

    view = st.session_state.get("auth_view", "login")
    user = get_current_user()

    if not user:
        if view == "register":
            render_register()
        else:
            render_login()
        st.stop()

    return user


def require_admin() -> dict:
    """Require admin role. Redirects non-admins."""
    user = require_auth()
    if user["role"] != "admin":
        st.error("⛔ Admin access required.")
        st.stop()
    return user


def _logout():
    token = st.session_state.get("token") or _get_query_token()
    if token:
        destroy_session(token)
    _clear_query_token()
    st.session_state.clear()
    st.rerun()


def render_logout_button(sidebar: bool = True, key: str = "logout_btn"):
    """Render a logout button in the sidebar."""
    container = st.sidebar if sidebar else st
    if container.button("⎋ Sign out", key=key):
        _logout()
