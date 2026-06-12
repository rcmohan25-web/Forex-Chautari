"""
app/api_client.py — thin JWT-authenticated wrapper around the FastAPI backend
for write operations from Streamlit dashboards.
"""
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"


class ApiError(Exception):
    def __init__(self, status_code, detail):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"{status_code}: {detail}")


def login_and_store_jwt(username: str, password: str) -> bool:
    """Call /auth/login and stash access+refresh tokens in session_state."""
    try:
        r = requests.post(f"{API_BASE}/auth/login",
                           json={"username": username, "password": password}, timeout=10)
        if r.ok:
            data = r.json()
            st.session_state["jwt_access"]  = data["access_token"]
            st.session_state["jwt_refresh"] = data["refresh_token"]
            return True
    except Exception:
        pass
    return False


def _refresh() -> bool:
    refresh = st.session_state.get("jwt_refresh")
    if not refresh:
        return False
    try:
        r = requests.post(f"{API_BASE}/auth/refresh",
                           json={"refresh_token": refresh}, timeout=10)
        if r.ok:
            data = r.json()
            st.session_state["jwt_access"]  = data["access_token"]
            st.session_state["jwt_refresh"] = data["refresh_token"]
            return True
    except Exception:
        pass
    return False


def _headers():
    token = st.session_state.get("jwt_access")
    if not token:
        raise ApiError(401, "Not authenticated with API — please log in again.")
    return {"Authorization": f"Bearer {token}"}


def _handle(resp, retry_fn):
    if resp.status_code == 401 and _refresh():
        return retry_fn()
    if not resp.ok:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = resp.text
        raise ApiError(resp.status_code, detail)
    return resp.json()


def api_get(path, **kw):
    def call():
        return requests.get(f"{API_BASE}{path}", headers=_headers(), timeout=15, **kw)
    r = call()
    return _handle(r, lambda: _handle(call(), lambda: (_ for _ in ()).throw(ApiError(401, "expired"))))


def api_post(path, **kw):
    def call():
        return requests.post(f"{API_BASE}{path}", headers=_headers(), timeout=30, **kw)
    r = call()
    return _handle(r, lambda: _handle(call(), lambda: (_ for _ in ()).throw(ApiError(401, "expired"))))


def api_delete(path, **kw):
    def call():
        return requests.delete(f"{API_BASE}{path}", headers=_headers(), timeout=15, **kw)
    r = call()
    return _handle(r, lambda: _handle(call(), lambda: (_ for _ in ()).throw(ApiError(401, "expired"))))