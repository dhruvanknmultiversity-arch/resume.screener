"""Access-gate tests. Run from screengenie_vercel/screengenie_vercel:  python -m pytest tests -q"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

_KEYS = ["SECRET_KEY", "TEAM_PASSWORD", "ALLOWED_EMAIL_DOMAINS", "ALLOWED_EMAILS", "DATABASE_URL"]
PW = "hunter2-team-pass"


def client(**env):
    for k in _KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    for m in ("app", "auth"):
        sys.modules.pop(m, None)
    import app as app_module
    import auth
    auth._FAILED_DELAY_SECONDS = 0
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def pw_only():
    return client(SECRET_KEY="x" * 40, TEAM_PASSWORD=PW)


def test_unconfigured_is_locked():
    c = client()
    r = c.get("/dashboard")
    assert r.status_code == 503 and b"locked" in r.data
    r = c.post("/scan/start")
    assert r.status_code == 503 and r.is_json


def test_short_secret_or_password_is_locked():
    assert client(SECRET_KEY="short", TEAM_PASSWORD=PW).get("/").status_code == 503
    assert client(SECRET_KEY="x" * 40, TEAM_PASSWORD="abc").get("/").status_code == 503


def test_healthz_is_public_and_leaks_nothing():
    assert client().get("/healthz").get_json() == {"status": "ok"}


def test_anonymous_requests_blocked(pw_only):
    r = pw_only.get("/candidate/334")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert pw_only.get("/candidate/334/download").status_code == 302
    assert pw_only.post("/scan/start").status_code == 401
    assert pw_only.post("/history/1/delete").status_code in (302, 401)


def test_password_only_login_page_has_no_email_field(pw_only):
    r = pw_only.get("/login")
    assert r.status_code == 200 and b"Team password" in r.data and b"Work email" not in r.data


def test_wrong_password_refused(pw_only):
    assert pw_only.post("/login", data={"password": "nope"}).status_code == 401


def test_password_only_sign_in_and_out(pw_only):
    r = pw_only.post("/login", data={"password": PW, "next": "/history"})
    assert r.status_code == 302 and r.headers["Location"] == "/history"
    r = pw_only.get("/")
    assert r.status_code == 200 and b"Sign out" in r.data
    for h in ("Content-Security-Policy", "X-Frame-Options", "X-Content-Type-Options", "Referrer-Policy"):
        assert h in r.headers
    assert r.headers.get("Cache-Control") == "no-store"
    pw_only.post("/logout")
    assert pw_only.get("/").status_code == 302


def test_open_redirect_blocked(pw_only):
    r = pw_only.post("/login", data={"password": PW, "next": "https://evil.com"})
    assert r.status_code == 302 and r.headers["Location"] == "/"
    r = pw_only.post("/login", data={"password": PW, "next": "//evil.com"})
    assert r.headers["Location"] == "/"


def test_optional_email_mode():
    c = client(SECRET_KEY="x" * 40, TEAM_PASSWORD=PW, ALLOWED_EMAIL_DOMAINS="multiversity.in",
               ALLOWED_EMAILS="boss@gmail.com")
    assert b"Work email" in c.get("/login").data
    assert c.post("/login", data={"email": "x@evil.com", "password": PW}).status_code == 401
    assert c.post("/login", data={"email": "BOSS@gmail.com", "password": PW}).status_code == 302
    c2 = client(SECRET_KEY="x" * 40, TEAM_PASSWORD=PW, ALLOWED_EMAIL_DOMAINS="multiversity.in")
    assert c2.post("/login", data={"email": "hr@multiversity.in", "password": PW}).status_code == 302
    assert b"hr@multiversity.in" in c2.get("/").data


def test_locked_page_names_the_problem_without_values():
    r = client(SECRET_KEY="tooshort", TEAM_PASSWORD="").get("/")
    assert r.status_code == 503
    assert b"SECRET_KEY is too short (8 characters" in r.data
    assert b"TEAM_PASSWORD is not set" in r.data
    assert b"tooshort" not in r.data
