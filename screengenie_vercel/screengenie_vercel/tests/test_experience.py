"""Experience calculation: every date format and trap seen in real resumes.

Expected months are counted by hand: inclusive of start and end month, ongoing roles up to
Nov 2026, a year with no month read as mid-year (Jul for a start, Jun for an end)."""
import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import experience as X  # noqa: E402

T = datetime.date(2026, 11, 13)

CASES = [
    # name, resume text, full-time months, internship months
    ("Mon YYYY", "EXPERIENCE\nAnalyst, ABC  Jan 2020 - Mar 2022", 27, 0),
    ("full month names", "EXPERIENCE\nAnalyst  September 2019 – August 2021", 24, 0),
    ("MM/YYYY", "EXPERIENCE\nAnalyst, ABC\n01/2020 - 03/2022", 27, 0),
    ("YYYY-MM", "EXPERIENCE\nAnalyst  2020-01 to 2022-03", 27, 0),
    ("DD/MM/YYYY", "EXPERIENCE\nAnalyst, ABC  01/06/2019 - 31/05/2021", 24, 0),
    ("Jan'20", "EXPERIENCE\nAnalyst, ABC  Jan'20 – Mar'22", 27, 0),
    ("Jan-2020", "EXPERIENCE\nAnalyst, ABC  Jan-2020 to Mar-2022", 27, 0),
    ("Jan 5, 2020", "EXPERIENCE\nAnalyst  Jan 5, 2020 – Dec 4, 2020", 12, 0),
    ("Till Date", "EXPERIENCE\nAnalyst, ABC  June 2021 - Till Date", 66, 0),
    ("Present", "EXPERIENCE\nAnalyst, ABC  Jan 2022 - Present", 59, 0),
    ("since", "EXPERIENCE\nAnalyst at ABC since March 2021", 69, 0),
    ("onwards", "EXPERIENCE\nSales Manager, XYZ 2021 onwards", 65, 0),
    ("2019-21", "EXPERIENCE\nAnalyst, ABC  2019-21", 24, 0),
    ("year only is ~2 years, not 3", "EXPERIENCE\nAnalyst, ABC  2019 - 2021", 24, 0),
    ("month then year only", "EXPERIENCE\nAnalyst  Mar 2019 - 2021", 28, 0),
    ("split over two lines", "EXPERIENCE\nAnalyst, ABC\nJan 2020 -\nPresent", 83, 0),
    ("From / To lines", "EXPERIENCE\nAnalyst, ABC\nFrom Jan 2020\nTo Dec 2020", 12, 0),
    ("table row", "Company | Role | Duration\nInfosys | Analyst | Jun 2023 - Present", 42, 0),
    ("overlap counted once", "EXPERIENCE\nA, X  Jan 2020 - Dec 2021\nB, Y  Jun 2021 - Jun 2022", 30, 0),
    ("future start not counted", "EXPERIENCE\nAnalyst, ABC  Jan 2027 - Present", 0, 0),
    ("end before start ignored", "EXPERIENCE\nAnalyst, ABC  Mar 2022 - Jan 2020", 0, 0),
    # jobs at educational institutions are real jobs
    ("teacher at a school", "EXPERIENCE\nMaths Teacher, Delhi Public School  Jun 2018 - May 2023", 60, 0),
    ("professor, no headings", "Assistant Professor, Amity University\nJul 2015 - Jun 2020", 60, 0),
    ("counsellor at an academy", "EXPERIENCE\nCounsellor, Byju's Academy  Apr 2021 - Mar 2023", 24, 0),
    # internships
    ("internship by title", "EXPERIENCE\nData Science Intern, Zoho  May 2024 - Jul 2024", 0, 3),
    ("internship section", "INTERNSHIPS\nData Intern, Zoho  May 2024 - Jul 2024\nEXPERIENCE\nAnalyst, Y  Aug 2024 - Present", 28, 3),
    ("trainee title above the dates", "EXPERIENCE\nMar 2020 - Feb 2022\nTrainee Engineer, L&T", 0, 24),
    ("a bullet about interns is not an internship",
     "EXPERIENCE\nSenior Analyst, X  Jan 2022 - Present\n- Ran the internship programme", 59, 0),
    # what is not work
    ("education section", "EDUCATION\nB.Tech, VIT  2015 - 2019\nEXPERIENCE\nAnalyst  Jul 2019 - Jun 2021", 24, 0),
    ("degree line with no headings", "B.Tech CSE, 2016 - 2020, CGPA 8.1\nSoftware Engineer, TCS  Jul 2020 - Jun 2023", 36, 0),
    ("projects section", "PROJECTS\nSales forecasting  Jan 2023 - Apr 2023\nEXPERIENCE\nAnalyst  Jan 2024 - Dec 2024", 12, 0),
    ("volunteering", "VOLUNTEER EXPERIENCE\nTeacher, NGO  Jan 2015 - Dec 2016", 0, 0),
    # headings in the wild
    ("heading with (5 years)", "EDUCATION\nB.Tech 2012-2016\nPROFESSIONAL EXPERIENCE (5 YEARS)\nAnalyst, X  Jan 2020 - Dec 2020", 12, 0),
    ("inline heading", "EDUCATION\nB.Tech 2012-2016\nExperience: Analyst at X, Jan 2020 - Dec 2020", 12, 0),
    ("organisational experience", "EDUCATION\nB.Tech 2012-2016\nORGANISATIONAL EXPERIENCE\nAnalyst, X  Jan 2020 - Dec 2020", 12, 0),
    ("research experience", "EDUCATION\nM.Sc 2012-2014\nResearch Experience\nResearch Assistant, IISc  Jan 2015 - Dec 2016", 24, 0),
    ("unknown heading after education", "EDUCATION\nB.Tech 2012-2016\nWHERE I'VE WORKED\nSoftware Engineer, Infosys  Jan 2017 - Dec 2018", 24, 0),
    ("job listed under education", "EDUCATION\nMBA, IIM  2018 - 2020\nProduct Manager, Swiggy  Jul 2020 - Jun 2022", 24, 0),
    ("work + internships heading", "Work Experience & Internships\nEngineer, X  Jan 2023 - Dec 2023", 12, 0),
    ("dates in the summary", "SUMMARY\nWorked at Infosys as analyst (2018 - 2020)", 24, 0),
]


@pytest.mark.parametrize("name,text,work,intern", CASES, ids=[c[0] for c in CASES])
def test_experience_cases(name, text, work, intern):
    e = X.from_text(text, T)
    assert (e["work"], e["internship"]) == (work, intern), [(r["text"], r["kind"], r["reason"]) for r in e["roles"]]


FULL = """KAVYA IYER
kavya.iyer@gmail.com | +91 98450 12345 | Bengaluru
PROFESSIONAL SUMMARY
Data professional with 8+ years of experience in analytics and teaching.
WORK EXPERIENCE
Senior Data Analyst | Swiggy | 01/2023 - Till Date
- Led a team of 4 analysts; ran the summer internship programme
Data Analyst, Mu Sigma
Jun'19 – Dec'22
Maths Teacher, Delhi Public School   2017 - 2019
Freelance SQL trainer   Mar 2021 - Aug 2021
INTERNSHIPS
Analytics Intern, Zoho   May 2016 - Jul 2016
EDUCATION
B.Sc Statistics, Christ University   2014 - 2017
PROJECTS
Churn model for telecom data   Jan 2018 - Mar 2018"""


def test_full_resume_breakdown():
    e = X.from_text(FULL, T)
    assert e["work"] == 113 and e["internship"] == 3          # Jul 2017 → Nov 2026, one union
    assert e["overlap"] == 7 and e["approximate"] and e["stated_years"] == 8
    kinds = [(r["title"].split(",")[0].split("|")[0].strip(), r["kind"]) for r in e["roles"]]
    assert kinds == [("Senior Data Analyst", "work"), ("Data Analyst", "work"), ("Maths Teacher", "work"),
                     ("Freelance SQL trainer", "work"), ("Analytics Intern", "internship"),
                     ("B.Sc Statistics", "excluded"), ("Churn model for telecom data", "excluded")]
    assert e["roles"][5]["reason"] == "Under “Education”"
    assert X.summary_note(e) == ""                            # 8+ stated vs 9.4 computed: close enough


def test_stated_years_far_from_dates_is_flagged():
    e = X.from_text("Summary: 10+ years of experience in sales.\nEXPERIENCE\nSales Exec, X  Jan 2022 - Dec 2023", T)
    assert "Resume states 10+ years; dated roles add up to 2 years." in X.summary_note(e)


def test_model_roles_must_have_dates_in_the_resume():
    text = "EXPERIENCE\nAnalyst, ABC  Jan 2020 - Present\nEngineer, XYZ  Jun 2017 - Dec 2019"
    roles = [{"title": "Analyst", "organization": "ABC", "start": "2020-01", "end": "present", "is_internship": False},
             {"title": "Engineer", "organization": "XYZ", "start": "2017-06", "end": "2019-12", "is_internship": False},
             {"title": "Invented", "organization": "Nowhere", "start": "2012-01", "end": "2014-12", "is_internship": False}]
    e = X.from_model_roles(roles, text, T)
    assert e["work"] == (2026 * 12 + 11) - (2017 * 12 + 6) + 1
    bad = [r for r in e["roles"] if r["title"].startswith("Invented")][0]
    assert bad["kind"] == "excluded" and "2012" in bad["reason"]


def test_model_cannot_make_a_finished_role_ongoing():
    e = X.from_model_roles([{"title": "Analyst", "start": "2020-01", "end": "present", "is_internship": False}],
                           "Analyst, ABC  Jan 2020 - Mar 2020", T)
    assert e["work"] == 1 and "ongoing" in e["roles"][0]["reason"]


def test_parse_month_compatibility():
    assert X.parse_month("2024-03", T) == 2024 * 12 + 2
    assert X.parse_month("Mar 2024", T) == 2024 * 12 + 2
    assert X.parse_month("present", T) == 2026 * 12 + 10
    assert X.parse_month("2024", T) == 2024 * 12 + 6 and X.parse_month("2024", T, is_end=True) == 2024 * 12 + 5
    assert X.parse_month("", T) is None and X.parse_month("soon", T) is None


def test_durations_are_years_and_months():
    f = X.format_duration
    assert [f(0), f(1), f(11), f(12), f(13), f(27), f(113)] == \
        ["0 mos", "1 mo", "11 mos", "1 yr", "1 yr 1 mo", "2 yrs 3 mos", "9 yrs 5 mos"]
    assert f(113, long=True) == "9 years 5 months" and X.format_years(2.5) == "2 years 6 months"
    assert f(None) == "—"
