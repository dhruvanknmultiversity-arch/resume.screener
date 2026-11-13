"""Engine v2 tests. No network: a fake client stands in for the Claude API.
Run from screengenie_vercel/screengenie_vercel:  python -m pytest tests -q"""
import json
import os
import sys
from datetime import date
from types import SimpleNamespace

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import engine  # noqa: E402

TODAY = date(2026, 11, 3)

RESUME = """Aarav Mehta
aarav.mehta@example.com | +91 98765 43210 | linkedin.com/in/aarav
Experience
ML Intern, Acme Labs   Jan 2025 - Jun 2025
- Built a churn model in scikit-learn and PyTorch for 2M customer records.
Software Engineer, Foo Pvt Ltd   Jul 2025 - Present
- Deployed a demo API on AWS EC2.
Education
B.Tech Computer Science, XYZ College 2020 - 2024
Skills: Python, Git, SQL
"""

CRITERIA = engine.parse_criteria_json(json.dumps([
    {"label": "Python for machine learning", "kind": "must", "weight": 3, "min_years": 0},
    {"label": "PyTorch or TensorFlow", "kind": "must", "weight": 3, "min_years": 0},
    {"label": "Cloud: AWS or GCP", "kind": "nice", "weight": 1, "min_years": 0},
    {"label": "2+ years of software experience", "kind": "must", "weight": 2, "min_years": 2},
    {"label": "Valid work authorisation in India", "kind": "knockout", "weight": 3, "min_years": 0},
]))


class FakeClient:
    """Mimics client.messages.create; records calls and returns canned JSON."""
    def __init__(self, payload):
        self.payload, self.calls = payload, []
        self.messages = self

    def create(self, **kw):
        self.calls.append(kw)
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        usage = SimpleNamespace(input_tokens=1200, output_tokens=800, cache_read_input_tokens=3000,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(stop_reason="end_turn", model="claude-sonnet-5-5", usage=usage,
                               content=[SimpleNamespace(type="text", text=text)])


def fake_eval(**overrides):
    data = {
        "current_title": "Software Engineer",
        "roles": [
            {"title": "ML Intern", "organization": "Acme Labs", "start": "2025-01", "end": "2025-06", "is_internship": True},
            {"title": "Software Engineer", "organization": "Foo", "start": "2025-07", "end": "present", "is_internship": False},
        ],
        "education": [{"degree": "B.Tech CS", "institution": "XYZ College", "year": "2024"}],
        "skills": ["Python", "PyTorch", "AWS", "Git", "SQL"],
        "criteria": [
            {"id": "c1", "verdict": "met", "evidence": "Built a churn model in scikit-learn and PyTorch", "reason": "ML work shown."},
            {"id": "c2", "verdict": "met", "evidence": "Fine-tuned a ResNet in TensorFlow", "reason": "Invented quote."},
            {"id": "c3", "verdict": "partial", "evidence": "Deployed a demo API on AWS EC2.", "reason": "Demo only."},
            {"id": "c4", "verdict": "met", "evidence": "Software Engineer, Foo Pvt Ltd", "reason": "Claims enough experience."},
            {"id": "c5", "verdict": "not_met", "evidence": "", "reason": "Not stated."},
        ],
        "summary": ["Strong ML basics.", "Limited cloud depth."],
        "interview_questions": ["How was the churn model validated?"],
        "red_flags": [],
    }
    data.update(overrides)
    return data


# --- criteria ---------------------------------------------------------------

def test_criteria_are_cleaned_and_bounded():
    c = engine.parse_criteria_json(json.dumps([
        {"label": "  Python  ", "kind": "must", "weight": 9, "min_years": -3},
        {"label": "python", "kind": "must"},                 # duplicate
        {"label": "", "kind": "nice"},                       # empty
        {"label": "Docker", "kind": "weird", "weight": "x"},
    ]))
    assert [x["label"] for x in c] == ["Python", "Docker"]
    assert c[0]["weight"] == 3 and c[0]["min_years"] == 0
    assert c[1]["kind"] == "must" and c[1]["id"] == "c2"
    assert engine.parse_criteria_json("not json") == []


def test_requisition_uses_model_then_falls_back():
    fake = FakeClient({"role_title": "AI/ML Intern", "seniority": "Internship", "criteria": [
        {"label": "Python", "kind": "must", "weight": 3, "min_years": 0}]})
    d = engine.draft_requisition("We need Python.", client=fake)
    assert d["engine"] == "ai" and d["criteria"][0]["label"] == "Python"
    body = fake.calls[0]["extra_body"]["output_config"]
    assert body["format"]["type"] == "json_schema" and body["effort"] == engine.EFFORT
    assert fake.calls[0]["model"] == engine.MODEL

    broken = FakeClient("this is not json")
    d2 = engine.draft_requisition("Looking for python, sql and aws. 3+ years of experience required.", client=broken)
    assert d2["engine"] == "rule" and d2["criteria"]
    assert any(c["min_years"] == 3 for c in d2["criteria"])


# --- masking, evidence, experience -----------------------------------------

def test_pii_is_masked_before_the_model_sees_it():
    m = engine.mask_pii(RESUME, "Aarav Mehta")
    for secret in ("aarav.mehta@example.com", "98765 43210", "Aarav", "Mehta", "linkedin.com/in/aarav"):
        assert secret.lower() not in m.lower()
    assert "[EMAIL]" in m and "[PHONE]" in m and "[NAME]" in m
    assert "2020 - 2024" in m      # years are not mistaken for phone numbers


def test_evidence_must_be_in_the_resume():
    assert engine.evidence_found("built a churn model in scikit-learn", RESUME)
    assert engine.evidence_found("Built a churn model … for 2M customer records", RESUME)
    assert not engine.evidence_found("Fine-tuned a ResNet in TensorFlow", RESUME)
    assert not engine.evidence_found("", RESUME)


def test_experience_union_excludes_overlap_and_separates_internships():
    exp = engine.experience_months([
        {"start": "2019-03", "end": "present", "is_internship": False},
        {"start": "2016-06", "end": "2019-02", "is_internship": False},
        {"start": "2018-01", "end": "2019-12", "is_internship": False},   # overlaps both
        {"start": "2015", "end": "2015", "is_internship": True},
    ], TODAY)
    assert exp["work"] == (2026 * 12 + 11) - (2016 * 12 + 6) + 1      # Jun 2016 → Nov 2026
    assert exp["internship"] == 12
    assert engine.experience_months([{"start": "2027-01", "end": "2027-05"}], TODAY)["work"] == 0


# --- evaluation and scoring ------------------------------------------------

def test_evaluation_downgrades_invented_quotes_and_checks_years():
    fake = FakeClient(fake_eval())
    ev = engine.evaluate_resume("JD", CRITERIA, RESUME, name="Aarav Mehta", client=fake, today=TODAY)
    res = {r["id"]: r for r in ev["results"]}
    assert ev["engine"] == "ai"
    assert res["c1"]["verdict"] == "met" and res["c1"]["verified"]
    assert res["c2"]["verdict"] == "partial" and res["c2"]["evidence"] == ""        # quote not in resume
    assert res["c4"]["verdict"] in ("partial", "not_met")                          # ~1.3 yrs vs 2 asked
    assert ev["experience"]["work"] == 17 and ev["experience"]["internship"] == 6  # Jul 2025 → Nov 2026
    assert ev["knockouts_failed"] == ["Valid work authorisation in India"]
    # the prompt sent to the model contains no contact details
    sent = json.dumps(fake.calls[0]["messages"])
    assert "aarav.mehta@example.com" not in sent and "Aarav" not in sent


def test_score_formula():
    crit = engine.parse_criteria_json(json.dumps([
        {"label": "A", "kind": "must", "weight": 3}, {"label": "B", "kind": "must", "weight": 3},
        {"label": "C", "kind": "nice", "weight": 1}]))
    v = {"c1": {"verdict": "met"}, "c2": {"verdict": "partial"}, "c3": {"verdict": "not_met"}}
    s = engine.score_evaluation(crit, v)
    assert s["score"] == round(100 * (3 + 1.5 + 0) / 7)
    assert s["must_met"] == 1 and s["must_total"] == 2 and s["knockouts_failed"] == []


def test_api_failure_falls_back_to_rules():
    ev = engine.evaluate_resume("JD", CRITERIA, RESUME, client=FakeClient("{broken"), today=TODAY)
    assert ev["engine"] == "rule"
    res = {r["id"]: r for r in ev["results"]}
    assert res["c1"]["verdict"] == "met" and "python" in res["c1"]["evidence"].lower()
    assert res["c5"]["verdict"] == "not_met"
    assert ev["experience"]["work"] == 17     # education years not counted


def test_app_pipeline_uses_criteria(monkeypatch):
    os.environ.update(SECRET_KEY="x" * 40, TEAM_PASSWORD="hunter2-team-pass")
    for m in ("app", "auth"):
        sys.modules.pop(m, None)
    import app as app_module
    monkeypatch.setattr(engine, "llm_available", lambda: False)
    cand, used_ai = app_module.process_resume_doc("JD text with python", ["python"], "a.pdf", RESUME, b"x", CRITERIA)
    assert used_ai is False
    assert cand["must_total"] == 4 and cand["knockout_failed"] is True and cand["status"] == "Reject"
    assert cand["experience_months"] == engine.experience_months(
        [{"start": "2025-07", "end": "present"}])["work"]
    assert cand["reasons"][0].startswith("Does not meet a knockout requirement")
    assert any("Ask for a concrete example" in q for q in cand["suggestions"])


# --- engine health (Day 1) ---------------------------------------------------

def test_meta_records_tokens_cost_and_path():
    ev = engine.evaluate_resume("JD", CRITERIA, RESUME, client=FakeClient(fake_eval()), today=TODAY)
    m = ev["meta"]
    assert m["path"] == "ai" and m["model"] == "claude-sonnet-5-5" and m["error"] is None
    assert (m["input_tokens"], m["output_tokens"], m["cache_read_tokens"]) == (1200, 800, 3000)
    expected = (1200 * engine.PRICE_IN + 800 * engine.PRICE_OUT + 3000 * engine.PRICE_CACHE_READ) / 1e6
    assert abs(m["cost_usd"] - expected) < 1e-9 and m["attempts"] == 1


def test_meta_records_fallback_reason():
    ev = engine.evaluate_resume("JD", CRITERIA, RESUME, client=FakeClient("{broken"), today=TODAY)
    assert ev["meta"]["path"] == "rule" and "JSONDecodeError" in ev["meta"]["error"]
    d = engine.draft_requisition("Need python and sql.", client=FakeClient("nope"))
    assert d["meta"]["kind"] == "requisition" and d["meta"]["path"] == "rule" and d["meta"]["error"]


def test_engine_run_is_logged_and_page_renders():
    os.environ.update(SECRET_KEY="x" * 40, TEAM_PASSWORD="hunter2-team-pass")
    for m in ("app", "auth"):
        sys.modules.pop(m, None)
    import app as app_module

    class Cur:
        def __init__(self): self.rows = []
        def execute(self, sql, params=None): self.rows.append((sql, params))
    cur = Cur()
    meta = dict(engine.new_meta("resume"), path="ai", model="claude-sonnet-5-5", cost_usd=0.02, seconds=4.2)
    app_module.log_engine_run(cur, 7, "a.pdf", meta)
    sql, params = [r for r in cur.rows if "INSERT INTO engine_runs" in r[0]][0]
    assert "INSERT INTO engine_runs" in sql and params[1] == 7 and params[3] == "ai" and params[5] == "a.pdf"
    n = len(cur.rows)
    app_module.log_engine_run(cur, 7, "a.pdf", None)          # nothing to log, no error
    assert len(cur.rows) == n

    from flask import render_template
    stats = dict(runs=10, ai=9, rule=1, ai_share=90, p50=6.1, p95=14.8, cost_inr=21.5, cost_per_resume_inr=2.1,
                 avg_in=6200, avg_out=1400, cache_share=40, req_runs=2, req_ai=2)
    config = dict(model="claude-sonnet-5-5", effort="low", api_key=True, parallel="5", timeout=55.0,
                  version="sg-engine-2.0", usd_inr=88.0, price_in=2.0, price_out=10.0)
    days = [dict(day="2026-11-13", resumes=10, ai=9, rule=1, cost=0.24)]
    failures = [dict(created_at="2026-11-13T09:00:00", kind="resume", path="rule", filename="b.pdf", scan_id=7,
                     error="RateLimitError: too many requests")]
    with app_module.app.test_request_context("/engine-health"):
        html = render_template("engine_health.html", active="engine", stats=stats, days=days,
                               failures=failures, config=config)
    assert "90%" in html and "RateLimitError" in html and "₹2.10" in html and "Engine health" in html
