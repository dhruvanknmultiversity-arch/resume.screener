"""Day 1.5: bugs found in the live test of the rule-based backup (13 Nov 2026)."""
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine  # noqa: E402
import parsing  # noqa: E402

TODAY = datetime.date(2026, 11, 13)
JD = """Data Analyst (Multiversity, Bengaluru)
Must have: 3+ years of experience as a data analyst; strong SQL; Python for analysis; dashboards in Power BI or Tableau.
Nice to have: A/B testing and statistics; stakeholder presentations."""
AISHA = """Aisha Verma | Data Analyst | Bengaluru
EXPERIENCE
Senior Data Analyst, Flipkart  Jan 2022 - Present
- Built Python and SQL pipelines; Power BI dashboards for 40 stakeholders
Data Analyst, Mu Sigma  Jul 2019 - Dec 2021
EDUCATION
B.Tech Computer Science, VIT, 2015 - 2019"""


def test_experience_from_dates_is_met_not_partial():
    crit = engine.rule_requisition(JD)["criteria"]
    ev = engine.rule_evaluate(crit, AISHA, today=TODAY)
    exp = [r for r in ev["results"] if r["min_years"]][0]
    assert exp["verdict"] == "met" and "7 years 5 months of full-time work against 3 years asked" in exp["reason"]


def test_experience_check_without_quote_still_met_on_ai_path():
    crit = [{"id": "c1", "label": "At least 3 years of experience", "kind": "must", "weight": 3, "min_years": 3}]
    exp = parsing.rule_experience_months(AISHA, TODAY)
    out = engine._finish(crit, [{"id": "c1", "verdict": "met", "evidence": "", "reason": ""}], AISHA, exp, "ai")
    assert out["results"][0]["verdict"] == "met"


def test_name_from_pipe_header_and_no_year_range_phone():
    ci = parsing.extract_contact_info(AISHA)
    assert ci["name"] == "Aisha Verma" and ci["phone"] == "Not found"
    ci = parsing.extract_contact_info("Priya Nair · priya@x.in · +91 98765 43210\nB.Sc 2022 - 2025")
    assert ci["name"] == "Priya Nair" and ci["phone"] == "+91 98765 43210"
    assert parsing.extract_phone("2018 - 2020 2020 - 2022") is None


def test_masking_keeps_year_runs_for_the_model():
    masked = engine.mask_pii("2018 - 2020 2020 - 2022\n+91 98765 43210")
    assert "2018 - 2020 2020 - 2022" in masked and "98765" not in masked


def test_checklist_groups_alternatives_and_finds_nice_to_haves():
    crit = engine.rule_requisition(JD)["criteria"]
    labels = {c["label"]: c["kind"] for c in crit}
    viz = [l for l in labels if "Power BI" in l]
    assert len(viz) == 1 and "Tableau" in viz[0] and " or " in viz[0]
    assert labels.get("SQL") == "must" and labels.get("Python") == "must"
    assert labels.get("A/B testing") == "nice" and labels.get("Statistics") == "nice"
    assert engine.rule_requisition(JD)["role_title"].startswith("Data Analyst")


def test_slash_names_are_not_split():
    crit = [{"id": "c1", "label": "A/B testing", "kind": "nice", "weight": 1, "min_years": 0},
            {"id": "c2", "label": "CI/CD", "kind": "must", "weight": 2, "min_years": 0}]
    ev = engine.rule_evaluate(crit, "Ran A/B tests weekly.\nBuilt CI/CD with Jenkins.", today=TODAY)
    assert [r["verdict"] for r in ev["results"]] == ["met", "met"]
