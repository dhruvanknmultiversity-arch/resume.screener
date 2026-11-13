"""HR feedback against a real Postgres. Runs only when SG_TEST_DATABASE_URL is set, e.g.
SG_TEST_DATABASE_URL=postgresql://postgres:pw@127.0.0.1:5499/postgres pytest tests/test_feedback_pg.py"""
import io
import json
import os
import re
import sys

import pytest

URL = os.environ.get("SG_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs SG_TEST_DATABASE_URL")
FIX = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture()
def client(monkeypatch):
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    monkeypatch.setenv("DATABASE_URL", URL)
    monkeypatch.setenv("SECRET_KEY", "x" * 40)
    monkeypatch.setenv("TEAM_PASSWORD", "hunter2-team-pass")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import pg8000.dbapi
    orig = pg8000.dbapi.connect
    monkeypatch.setattr(pg8000.dbapi, "connect", lambda *a, **k: orig(*a, **dict(k, ssl_context=None)))
    for m in ("app", "auth"):
        sys.modules.pop(m, None)
    import app as app_module
    c = app_module.app.test_client()
    with c.session_transaction() as s:
        s["signed_in"] = True
    return c


def test_feedback_round_trip(client):
    jd = (io.BytesIO(b"Data Analyst\nMust have: 3+ years experience; SQL; Python"), "jd.txt")
    sid = client.post("/scan/start", data={"jd_file": jd}, content_type="multipart/form-data").get_json()["scan_id"]
    with open(os.path.join(FIX, "two_column.pdf"), "rb") as f:
        client.post(f"/scan/{sid}/batch", data={"resume_files": [(f, "two_column.pdf")]},
                    content_type="multipart/form-data")
    client.post(f"/scan/{sid}/finalize")
    cid = int(re.search(r'/candidate/(\d+)"', client.get(f"/results/{sid}").data.decode()).group(1))
    H = {"Content-Type": "application/json"}

    def fb(**k):
        r = client.post(f"/candidate/{cid}/feedback", data=json.dumps(k), headers=H)
        return r.status_code, r.get_json()

    assert fb(target="overall", agree=True)[1]["agree"] is True
    code, d = fb(target="criterion", criterion_id="c2", agree=False, hr_value="not_met")
    assert code == 200 and d["system_value"] == "met" and d["agree"] is False
    assert fb(target="criterion", criterion_id="c2", agree=False, hr_value="maybe")[0] == 400
    assert fb(target="experience", agree=False, hr_value=110)[1]["hr_value"] == "110"
    assert fb(target="experience", agree=False, hr_value=113)[1]["agree"] is True      # same as computed
    r = client.post(f"/candidate/{cid}/feedback", data=json.dumps({"target": "overall", "agree": True}),
                    headers=dict(H, Origin="https://evil.example"))
    assert r.status_code == 403
    assert fb(target="overall", clear=True)[1]["cleared"] is True
    page = client.get(f"/candidate/{cid}").data.decode()
    assert "2 of 5 checks answered" in page       # result + 3 requirements + experience


def test_accuracy_page_reports_answers(client):
    r = client.get("/accuracy")
    page = r.data.decode()
    assert r.status_code == 200 and "Overall result right" in page and "Latest corrections" in page
    for d in ("7", "30", "90", "all"):
        assert client.get(f"/accuracy?days={d}").status_code == 200
