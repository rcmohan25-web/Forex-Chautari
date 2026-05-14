"""
ForexChautari — main entry point.
Routes to admin panel or user dashboard based on role.

Run:  streamlit run app/main.py
"""

import streamlit as st
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.auth import require_auth, render_logout_button

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
