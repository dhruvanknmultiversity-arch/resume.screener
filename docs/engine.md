# Screen Genie engine v2 (smart screening)

## Flow
1. **JD → requirement checklist** (`POST /requisition/draft`, `engine.draft_requisition`): 4–10 checkable
   criteria, each `must`, `nice` or `knockout`, weight 1–3 and optional minimum years. HR edits the list on
   Bulk Screener (step 3) before screening; it is stored on the scan (`scans.criteria`).
2. **Resume → evidence** (`engine.evaluate_resume`): one Claude call per resume with structured JSON output.
   Name, email, phone and links are masked first. For every criterion the model returns `met / partial /
   not_met`, an exact quote and a reason. A quote not found in the resume turns `met` into `partial`.
3. **Experience** is computed in code from the dated roles the model extracts: overlapping jobs count once,
   internships are kept separate, education/projects are not roles. Minimum-years criteria are capped by this.
4. **Score** = 100 × Σ(weight × value) / Σ(weight), value met = 1, partial = 0.5, not met = 0.
   A failed knockout sets status to Reject and is shown first in the reasons; it does not change the score.

## Settings (environment variables)
| Variable | Default | Meaning |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | — | Required for AI screening; without it the rule-based fallback runs |
| `SG_MODEL` | `claude-sonnet-5-5` | Model for drafting and judging |
| `SG_EFFORT` | `low` | `low`, `medium` or `high` (higher = slower, more careful) |
| `SG_TIMEOUT` | `55` | Seconds per API call |
| `SG_PARALLEL` | `5` | Concurrent API calls per instance |

## Fallback
If the API key is missing or a call fails, the rule engine runs: skills and aliases are matched line by line
and the matching line is quoted; experience comes from dated job lines with education and project lines
excluded. Failures are always logged with the prefix `[screengenie:engine]`.

## Tests
`tests/test_engine.py` (fake client, no network): criteria cleaning, model call shape, fallback, PII masking,
evidence verification, experience maths, score formula, knockouts and the app pipeline.

## Experience calculation (`experience.py`)

The same rules apply whether Claude or the rule-based backup screens the resume.

1. **Dates in any common format** are read: `Jan 2020 - Mar 2022`, `January 2020 – Present`,
   `01/2020 - 03/2022`, `2020-01 to 2022-03`, `01/06/2019 - 31/05/2021` (day first), `Jan'20 – Mar'22`,
   `Jan-2020 to Mar-2022`, `2019 - 2021`, `2019-21`, `since March 2021`, `2021 onwards`,
   `June 2021 - Till Date`, and ranges split over two lines.
2. **Sections decide what counts.** Ranges under Experience (any heading ending in "experience",
   "Employment", "Work history", "Career"...) count as work; under Internships as internship; under
   Education, Projects, Certifications, Volunteering, Achievements and similar they are not counted.
   Before any heading, a line counts unless it reads like education (degree, CGPA, semester), so a teacher
   at a school or a professor at a university is counted.
3. **Internship** only when the role's own title says intern / trainee / apprentice (or it sits under an
   Internships heading). A bullet such as "ran the internship programme" does not make a job an internship.
4. **Months are inclusive** (`Jan 2020 - Mar 2020` = 3). Ongoing roles count to the current month; future
   months never count. A **year with no month is read as mid-year**, so `2019 - 2021` ≈ 24 months, not 36;
   such totals are marked approximate.
5. **Overlaps count once** (union of intervals). Internship months are kept apart from full-time work.
6. **Claude's roles are checked**: a role whose start or end year is not in the resume is not counted, and
   a role the resume does not show as ongoing cannot be made "present". If Claude's total and the date scan
   differ by a year or more, the candidate page says so.
7. **Stated vs. dated**: if the resume claims "N+ years" and the dates differ by 1.5 years or more, the
   candidate page flags it.

The candidate page shows the full breakdown under **How experience was calculated**: every dated line,
its months, and whether it counted as work, internship or not at all, with the reason.
`tests/test_experience.py` holds 40+ cases covering every format and trap above.
