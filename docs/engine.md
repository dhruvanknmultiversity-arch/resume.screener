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
