# Screen Genie — current state (Day 0 review, 30 Oct 2026)

Reviewed commit `0852f87` on `main`. This note records what exists today so the
v2 rebuild can be checked against it.

## Stack

| Part | What it is |
| --- | --- |
| App | One Flask app (`screengenie_vercel/screengenie_vercel/app.py`, ~1,200 lines) with Jinja templates, deployed on Vercel's Python runtime (`vercel.json` routes everything to `app.py`) |
| Engine | `parsing.py` (~1,500 lines): pdfplumber / python-docx text extraction, regex contact + experience parsing, a hand-made skill alias list, TF-IDF cosine (scikit-learn), and an optional Claude call per resume |
| Database | Supabase used **only as plain Postgres** via `pg8000` and `DATABASE_URL`. No Supabase Auth, Storage or row-level security |
| Files | Original resumes stored as `BYTEA` inside the `candidates` table |
| Schema | Created at runtime by `init_db()` (`CREATE TABLE IF NOT EXISTS` + `ALTER TABLE … ADD COLUMN IF NOT EXISTS`) on the first request of each instance. Tables: `scans`, `candidates`, `jd_analyses`, `jd_fixes` |

## Root causes behind the QA findings

| QA finding | Cause in the code |
| --- | --- |
| Anyone can see and delete data | No authentication of any kind; all routes public; numeric IDs |
| "Rule-based engine" on results | `scan_finalize` labels a scan rule-based when fewer than half the resumes got a Claude result. `analyze_resume_with_claude` swallows every exception and returns `None`, and errors were only logged when `SCREENGENIE_DEBUG` was set, so failures were invisible. The call uses model id `claude-opus-4-8` with adaptive thinking and a 180 s timeout per resume; check the Vercel logs (now always logged) to confirm whether it is the model id, the API key, rate limits or function timeouts |
| "~8 years experience" for an intern | `candidate_years()` adds up **every** `YYYY – YYYY` range in the resume (education, projects, internships) and takes the max with any "N years" phrase |
| Noisy JD keywords ("teamwork", "projects", "healthcare") | `top_keywords()` ranks frequent JD words after a stop-word filter; any 3+ letter word that survives becomes a "skill" |
| Contradictory numbers on the candidate page | Overall score comes from Claude (or the rule engine), while keyword coverage, format and contact scores come from separate regex/TF-IDF code; they are never reconciled |
| Grade mix sums to 266, not 322 | Dashboard query counted grades A–D only; grade F (score < 40) was dropped |
| Two different "detected role" values | Screener uses `guess_role()` = first non-empty line of the JD; JD Analysis asks Claude |
| Duplicates inflate totals | Dedupe is client-side only (file name + size) and per batch; no candidate identity across scans |
| `/healthz` | Public and listed database-related environment variable names |
| Weak default secret | `SECRET_KEY` fell back to a hard-coded string |

## Day 0 changes (branch `v2/day00-login-gate`)

1. **Login gate** (`auth.py`, `templates/login.html`): every page, API call and
   resume file now requires sign-in with an allowed email + the team password.
   Fails closed when not configured. Sign-out in the sidebar.
2. **Security headers** on every response; uploaded resumes viewed inline are
   sandboxed; `Cache-Control: no-store` for signed-in pages.
3. **`/healthz`** returns only `{"status": "ok"}`.
4. **Claude failures are always logged** to the Vercel function logs.
5. **Dashboard grade mix** now includes grade F.

These are stop-gaps. Day 1 replaces the shared password with Supabase Auth + SSO
and adds row-level security and non-guessable IDs.

## Environment variables required after this branch

| Name | Example | Notes |
| --- | --- | --- |
| `SECRET_KEY` | 64 random characters | Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `TEAM_PASSWORD` | a long passphrase | Share with the HR team privately; rotate when someone leaves |
| `ALLOWED_EMAIL_DOMAINS` | `multiversity.in` | Comma-separated |
| `ALLOWED_EMAILS` | `someone@gmail.com` | Optional extra addresses |
| `SESSION_HOURS` | `12` | Optional |
