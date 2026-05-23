"""
ForexChautari — main entry point.
Routes to admin panel or user dashboard based on role.

Run:  streamlit run app/main.py
"""

import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.startup_checks import validate_env
from app.auth import require_auth, render_logout_button

# Validate environment before rendering anything — surfaces config errors
# as a clear Streamlit error rather than a cryptic mid-render crash.
try:
    validate_env()
except SystemExit:
    st.error(
        "**Missing required environment variables.** "
        "Check the terminal output for the full list and generation commands. "
        "Copy `.env.example` to `.env`, fill in the missing values, and restart."
    )
    st.stop()

st.set_page_config(
    page_title="ForexChautari",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

user = require_auth()

if user["role"] == "admin":
    from app.admin_panel import render_admin
    render_admin(user)
else:
    from app.user_dashboard import render_user_dashboard
    render_user_dashboard(user)
