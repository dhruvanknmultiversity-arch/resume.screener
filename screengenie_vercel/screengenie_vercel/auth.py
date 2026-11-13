"""Day-0 access gate for Screen Genie.

Every route except /login, /logout, /healthz, /cron/keepalive and /static now requires a signed-in
session. Sign-in = the shared team password. If ALLOWED_EMAIL_DOMAINS or
ALLOWED_EMAILS is set, a matching work email is required as well. This is a
stop-gap until Day 1 replaces it with Supabase Auth + Google/Microsoft SSO.

The gate FAILS CLOSED: if SECRET_KEY or TEAM_PASSWORD is not configured in the
environment, nobody can sign in and every page shows a "not configured" notice
instead of candidate data.

Environment variables
---------------------
SECRET_KEY             Long random string used to sign session cookies (required).
TEAM_PASSWORD          Shared password for the HR team (required).
ALLOWED_EMAIL_DOMAINS  Optional. Comma-separated, e.g. "multiversity.in".
ALLOWED_EMAILS         Optional. Comma-separated extra addresses.
                       If neither is set, sign-in is password only.
SESSION_HOURS          Optional, default 12.
"""

import hmac
import os
import sys
import time
from datetime import timedelta
from urllib.parse import urlparse

from flask import (
    jsonify, redirect, render_template, request, session, url_for,
)

_DEV_KEY = "screen-genie-internal-dev-key"
_PUBLIC_ENDPOINTS = {"login", "logout", "healthz", "static", "cron_keepalive"}
_FAILED_DELAY_SECONDS = 1.0


def _csv_env(name):
    return {v.strip().lower() for v in os.environ.get(name, "").split(",") if v.strip()}


def _config_problem():
    """Return a human-readable reason the gate cannot work, or None if OK."""
    key = os.environ.get("SECRET_KEY", "").strip()
    pw = os.environ.get("TEAM_PASSWORD", "")
    problems = []
    if not key:
        problems.append("SECRET_KEY is not set for this environment.")
    elif key == _DEV_KEY or len(key) < 32:
        problems.append(f"SECRET_KEY is too short ({len(key)} characters; needs 32 or more).")
    if not pw:
        problems.append("TEAM_PASSWORD is not set for this environment.")
    elif len(pw) < 8:
        problems.append(f"TEAM_PASSWORD is too short ({len(pw)} characters; needs 8 or more).")
    return " ".join(problems) or None


def email_required():
    return bool(_csv_env("ALLOWED_EMAIL_DOMAINS") or _csv_env("ALLOWED_EMAILS"))


def email_allowed(email):
    if not email_required():
        return True
    email = (email or "").strip().lower()
    if "@" not in email or email.count("@") != 1:
        return False
    if email in _csv_env("ALLOWED_EMAILS"):
        return True
    domain = email.rsplit("@", 1)[1]
    return domain in _csv_env("ALLOWED_EMAIL_DOMAINS")


def password_ok(candidate):
    expected = os.environ.get("TEAM_PASSWORD", "")
    if not expected:
        return False
    return hmac.compare_digest(
        (candidate or "").encode("utf-8"), expected.encode("utf-8")
    )


def _safe_next(target):
    """Only allow redirects back to a path on this site."""
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc or not target.startswith("/") or target.startswith("//"):
        return None
    return target


def _wants_json():
    return (request.path.startswith("/scan")
            or request.accept_mimetypes.best == "application/json"
            or request.headers.get("X-Requested-With") == "fetch")


def current_user():
    return session.get("user_email")


def init_auth(app):
    """Register the gate. Call right after creating the Flask app so the gate
    runs before any other before_request hook (including DB initialisation)."""
    secret = os.environ.get("SECRET_KEY", "")
    app.secret_key = secret if secret else os.urandom(32)  # unusable random key if unset
    hours = int(os.environ.get("SESSION_HOURS", "12") or 12)
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=bool(os.environ.get("VERCEL")) or os.environ.get("FORCE_SECURE_COOKIES") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=hours),
    )

    @app.before_request
    def _require_login():
        if request.endpoint in _PUBLIC_ENDPOINTS:
            return None
        problem = _config_problem()
        if problem:
            print(f"[screen-genie] access gate not configured: {problem}", file=sys.stderr)
            if _wants_json():
                return jsonify({"error": "Screen Genie is locked: sign-in is not configured yet."}), 503
            return render_template("login.html", locked=True, problem=problem), 503
        if session.get("signed_in") and email_allowed(current_user()):
            return None
        session.clear()
        if _wants_json():
            return jsonify({"error": "Your session has expired. Please sign in again."}), 401
        return redirect(url_for("login", next=request.full_path.rstrip("?")))

    @app.after_request
    def _security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        resp.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "img-src 'self' data: blob:; font-src 'self' data: https://fonts.gstatic.com; "
            "connect-src 'self'; frame-ancestors 'self'; "
            "base-uri 'self'; form-action 'self'; object-src 'self'",
        )
        if session.get("signed_in"):
            # Candidate data must never sit in shared or browser caches.
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.context_processor
    def _inject_user():
        return {"current_user_email": current_user(), "signed_in": bool(session.get("signed_in")),
                "vercel_env": os.environ.get("VERCEL_ENV", "local")}

    @app.route("/login", methods=["GET", "POST"])
    def login():
        problem = _config_problem()
        if problem:
            return render_template("login.html", locked=True, problem=problem), 503
        next_url = _safe_next(request.values.get("next")) or url_for("index")
        if request.method == "GET":
            if session.get("signed_in") and email_allowed(current_user()):
                return redirect(next_url)
            return render_template("login.html", next_url=next_url, need_email=email_required())

        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        if email_allowed(email) and password_ok(password):
            session.clear()
            session.permanent = True
            session["signed_in"] = True
            session["user_email"] = email if email_required() else None
            session["signed_in_at"] = int(time.time())
            print(f"[screen-genie] sign-in ok: {email or 'team password'}", file=sys.stderr)
            return redirect(next_url)

        time.sleep(_FAILED_DELAY_SECONDS)  # slow down password guessing
        print(f"[screen-genie] sign-in failed for: {email or '(password only)'}", file=sys.stderr)
        return render_template(
            "login.html", next_url=next_url, email=email, need_email=email_required(),
            error=("That email isn't allowed or the password is wrong."
                   if email_required() else "That password is wrong."),
        ), 401

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("login"))
