import os
import sys
import io
import csv
import json
import time
import mimetypes
import concurrent.futures
from datetime import datetime
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pg8000.dbapi
from flask import (
    Flask, render_template, request, redirect, url_for, g, flash,
    send_file, abort, jsonify, Response,
)

from parsing import (
    extract_text_from_file,
    extract_resumes_from_zip,
    top_keywords,
    analyze_resume,
    analyze_resume_with_claude,
    grade_for_score,
    status_for_score,
    analyze_jd_quality,
    analyze_jd_quality_with_claude,
    fix_jd_with_claude,
)
import engine

from auth import init_auth

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 3600  # static CSS/icons; bust with ?v=N
# Access gate: must be registered before any other before_request hook.
init_auth(app)


# ---------------------------------------------------------------------------
# Friendly error pages — never show a raw stack trace or bare Vercel error.
# JSON requests (our fetch calls) get a JSON body; page loads get the HTML page.
# ---------------------------------------------------------------------------
def _wants_json():
    return (request.path.startswith("/scan/") or
            request.accept_mimetypes.best == "application/json" or
            request.headers.get("X-Requested-With") == "fetch")


def _error_response(code, heading, message, icon="⚠️"):
    if _wants_json():
        return jsonify({"error": message}), code
    try:
        return render_template("error.html", code=code, heading=heading,
                               message=message, icon=icon), code
    except Exception:
        return f"{code} — {message}", code


@app.errorhandler(404)
def _err_404(e):
    return _error_response(404, "Page not found",
                           "That page or record doesn't exist. It may have been deleted.", "🔍")


@app.errorhandler(413)
def _err_413(e):
    return _error_response(413, "Upload too large",
                           "That upload exceeded the size limit. Very large files are skipped "
                           "automatically — please try again with the Files tab.", "📦")


@app.errorhandler(500)
@app.errorhandler(Exception)
def _err_500(e):
    import sys, traceback
    traceback.print_exc(file=sys.stderr)
    # Re-raise HTTP errors that already have their own handler
    code = getattr(e, "code", 500) or 500
    if code == 404:
        return _err_404(e)
    if code == 413:
        return _err_413(e)
    return _error_response(500, "Something went wrong",
                           "The server hit an unexpected error. Please try again in a moment.", "🛠️")

# Read at request time so Railway's env vars are always picked up
def get_database_url():
    # Try multiple env var names Railway might use
    url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_PRIVATE_URL") or ""
    return url


def parse_db_url(url):
    # urlparse misreads dotted usernames like postgres.xxxx in Supabase URLs.
    # Manually extract user:password@host:port/db from the URL.
    url = url.strip()
    # strip scheme
    rest = url.split("://", 1)[1]
    # split userinfo from hostinfo
    at_idx = rest.rfind("@")
    userinfo = rest[:at_idx]
    hostinfo = rest[at_idx + 1:]
    # split user:password
    if ":" in userinfo:
        user, password = userinfo.split(":", 1)
    else:
        user, password = userinfo, ""
    # split host:port/db
    if "/" in hostinfo:
        hostport, database = hostinfo.split("/", 1)
    else:
        hostport, database = hostinfo, ""
    if ":" in hostport:
        host, port = hostport.rsplit(":", 1)
        port = int(port)
    else:
        host, port = hostport, 5432
    return {
        "host": host,
        "port": port,
        "database": database.split("?")[0],
        "user": user,
        "password": password,
    }


def connect_db():
    """Open a fresh Postgres connection. Safe to call from background threads
    (Flask's `g`/get_db is request-scoped and must not be shared across threads).

    A 10s socket timeout means a slow/unreachable DB fails fast with an error
    page instead of hanging until Vercel kills the function (which looks like a
    crash). For best performance on serverless, point DATABASE_URL at Supabase's
    *connection pooler* (port 6543, transaction mode) — see deploy notes."""
    url = get_database_url()
    if not url or "://" not in url:
        raise RuntimeError(f"DATABASE_URL not set or invalid: '{url}'")
    p = parse_db_url(url)
    last_err = None
    for attempt in range(2):  # one retry for transient pooler hiccups
        try:
            conn = pg8000.dbapi.connect(
                host=p["host"], port=p["port"], database=p["database"],
                user=p["user"], password=p["password"], ssl_context=True,
                timeout=10,
            )
            conn.autocommit = False
            return conn
        except Exception as e:
            last_err = e
            if attempt == 0:
                time.sleep(0.8)
    raise last_err


def get_db():
    if "db" not in g:
        g.db = connect_db()
    return g.db


def fetchall_dict(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def fetchone_dict(cursor):
    cols = [d[0] for d in cursor.description]
    row = cursor.fetchone()
    return dict(zip(cols, row)) if row else None


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = connect_db()
    cur = db.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id SERIAL PRIMARY KEY,
            jd_filename TEXT NOT NULL,
            jd_role TEXT,
            keywords TEXT,
            resume_count INTEGER NOT NULL,
            avg_score REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            id SERIAL PRIMARY KEY,
            scan_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            score INTEGER NOT NULL,
            grade TEXT NOT NULL,
            status TEXT NOT NULL,
            section_scores TEXT,
            matched_keywords TEXT,
            missing_keywords TEXT,
            matched_skills TEXT,
            missing_skills TEXT,
            extra_skills TEXT,
            reasons TEXT,
            file_data BYTEA,
            suggestions TEXT,
            passed_checks TEXT,
            warning_checks TEXT,
            issue_checks TEXT,
            semantic_score INTEGER,
            FOREIGN KEY (scan_id) REFERENCES scans (id)
        )
    """)
    # Columns added after v1 — safe to run repeatedly
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS duration_seconds REAL")
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS engine TEXT")
    # Batched-upload support: JD text + keywords are stored on the scan so each
    # resume batch (uploaded in a separate request) can reuse them.
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS jd_text TEXT")
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS status TEXT")
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS claude_count INTEGER DEFAULT 0")
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS session_name TEXT")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS cand_years INTEGER")
    # Engine v2: requirement checklist per scan, evidence per candidate
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS criteria TEXT")
    cur.execute("ALTER TABLE scans ADD COLUMN IF NOT EXISTS engine_version TEXT")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS criteria_results TEXT")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS experience_months INTEGER")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS internship_months INTEGER")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS experience_detail TEXT")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS knockout_failed BOOLEAN DEFAULT FALSE")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS must_met INTEGER")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS must_total INTEGER")
    cur.execute("ALTER TABLE candidates ADD COLUMN IF NOT EXISTS current_title TEXT")
    # Engine health log: one row per model call or fallback (Day 1)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS engine_runs (
            id SERIAL PRIMARY KEY,
            created_at TEXT NOT NULL,
            scan_id INTEGER,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            model TEXT,
            filename TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            seconds REAL DEFAULT 0,
            cost_usd REAL DEFAULT 0,
            attempts INTEGER DEFAULT 0,
            error TEXT
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS engine_runs_created_idx ON engine_runs (created_at)")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jd_analyses (
            id SERIAL PRIMARY KEY,
            jd_filename TEXT,
            jd_text TEXT,
            description TEXT,
            overall_score INTEGER,
            scores TEXT,
            missing_sections TEXT,
            missing_keywords TEXT,
            issues TEXT,
            strengths TEXT,
            suggestions TEXT,
            seniority_level TEXT,
            detected_role TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS jd_fixes (
            id SERIAL PRIMARY KEY,
            analysis_id INTEGER REFERENCES jd_analyses(id),
            fixed_jd_text TEXT,
            created_at TEXT
        )
    """)
    # Indexes — make per-scan candidate lookups and ranking fast
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_scan_id ON candidates (scan_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_candidates_score ON candidates (score DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scans_created ON scans (created_at DESC)")
    db.commit()
    cur.close()
    db.close()


def guess_role(jd_text, jd_filename):
    for line in jd_text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return jd_filename


@app.route("/healthz")
def healthz():
    # Public endpoint: report liveness only, never configuration details.
    return {"status": "ok"}


@app.route("/cron/keepalive")
def cron_keepalive():
    """Daily Vercel cron: touch the database so Supabase never pauses the project for inactivity.

    Vercel sends "Authorization: Bearer <CRON_SECRET>" when CRON_SECRET is set; then nothing else
    may call this. Without the secret it still only writes one timestamp, so it is harmless."""
    import hmac
    secret = os.environ.get("CRON_SECRET", "")
    if secret:
        sent = request.headers.get("Authorization", "")
        if not hmac.compare_digest(sent.encode(), f"Bearer {secret}".encode()):
            return {"status": "forbidden"}, 403
    try:
        db = get_db()
        cur = db.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS heartbeat (id INTEGER PRIMARY KEY, last_seen TEXT NOT NULL)")
        cur.execute("INSERT INTO heartbeat (id, last_seen) VALUES (1, %s) "
                    "ON CONFLICT (id) DO UPDATE SET last_seen = EXCLUDED.last_seen",
                    (datetime.utcnow().isoformat(),))
        db.commit()
        cur.close()
        return {"status": "ok"}
    except Exception as e:
        print(f"[screen-genie] keepalive failed: {type(e).__name__}: {e}", file=sys.stderr)
        return {"status": "error"}, 503


@app.route("/")
def index():
    return render_template("bulk_screener.html")


# ---------------------------------------------------------------------------
# Synchronous scan — runs entirely within the POST request.
# Works on Vercel (serverless) and locally. The browser shows a full-page
# loading overlay while the server processes; on completion it is redirected
# to /results/<scan_id> automatically.
# ---------------------------------------------------------------------------

def _legacy_candidate(jd_text, keywords, filename, text, raw):
    """Pre-v2 path, used only when a scan has no requirement checklist."""
    claude_result = analyze_resume_with_claude(jd_text, text, keywords)
    used_claude = claude_result is not None
    analysis = claude_result or analyze_resume(jd_text, text, keywords)
    score = analysis["score"]
    return {
        "filename": filename,
        "name": analysis["contact"]["name"],
        "email": analysis["contact"]["email"],
        "phone": analysis["contact"]["phone"],
        "score": score,
        "grade": grade_for_score(score),
        "status": status_for_score(score),
        "section_scores": analysis["section_scores"],
        "matched_keywords": analysis["matched_keywords"],
        "missing_keywords": analysis["missing_keywords"],
        "matched_skills": analysis["matched_skills"],
        "missing_skills": analysis["missing_skills"],
        "extra_skills": analysis["extra_skills"],
        "reasons": analysis["reasons"],
        "suggestions": analysis.get("suggestions", []),
        "passed_checks": analysis.get("passed_checks", []),
        "warning_checks": analysis.get("warning_checks", []),
        "issue_checks": analysis.get("issue_checks", []),
        "semantic_score": analysis.get("semantic_score", 0),
        "cand_years": analysis.get("cand_years"),
        "engine_meta": dict(engine.new_meta("resume"), path="ai" if used_claude else "rule",
                            model=os.environ.get("CLAUDE_MODEL") if used_claude else None,
                            error=None if used_claude else "legacy path: no requirement checklist"),
        "raw": raw,
    }, used_claude


def process_resume_doc(jd_text, keywords, filename, text, raw, criteria=None):
    """Score one resume. Returns (candidate_dict, used_ai)."""
    if not criteria:
        return _legacy_candidate(jd_text, keywords, filename, text, raw)

    rule = analyze_resume(jd_text, text, keywords)          # contact, keywords, format checks
    contact = rule["contact"]
    ev = engine.evaluate_resume(jd_text, criteria, text, name=contact["name"])
    results = ev["results"]
    by_verdict = lambda v: [r["label"] for r in results if r["verdict"] == v]

    score = ev["score"]
    status = "Reject" if ev["knockouts_failed"] else status_for_score(score)
    work_years = ev["experience"]["work"] / 12.0
    need_years = max([c.get("min_years") or 0 for c in criteria] or [0])
    exp_score = (min(100, round(100 * work_years / need_years)) if need_years
                 else rule["section_scores"]["experience"])
    must_pct = round(100 * ev["must_met"] / ev["must_total"]) if ev["must_total"] else score
    section_scores = dict(rule["section_scores"], skills=must_pct, experience=exp_score)

    reasons = [f"Does not meet a knockout requirement: {k}." for k in ev["knockouts_failed"]]
    reasons += ev.get("summary") or []
    if not ev.get("summary"):
        met, part, miss = by_verdict("met"), by_verdict("partial"), by_verdict("not_met")
        if met:
            reasons.append("Shows " + ", ".join(met[:5]) + ".")
        if part:
            reasons.append("Partly shows " + ", ".join(part[:4]) + ".")
        if miss:
            reasons.append("No evidence found for " + ", ".join(miss[:4]) + ".")
    questions = ev.get("interview_questions") or [
        f"Ask for a concrete example of: {r['label']}." for r in results
        if r["kind"] in ("must", "knockout") and r["verdict"] != "met"][:4]

    label_set = {r["label"].lower() for r in results}
    extra = [sk for sk in (ev.get("skills") or rule["extra_skills"]) if sk.lower() not in label_set][:20]

    return {
        "filename": filename,
        "name": contact["name"],
        "email": contact["email"],
        "phone": contact["phone"],
        "score": score,
        "grade": grade_for_score(score),
        "status": status,
        "section_scores": section_scores,
        "matched_keywords": rule["matched_keywords"],
        "missing_keywords": rule["missing_keywords"],
        "matched_skills": by_verdict("met"),
        "missing_skills": by_verdict("not_met"),
        "extra_skills": extra,
        "reasons": reasons,
        "suggestions": questions,
        "passed_checks": rule.get("passed_checks", []),
        "warning_checks": rule.get("warning_checks", []) + (ev.get("red_flags") or []),
        "issue_checks": rule.get("issue_checks", []),
        "semantic_score": rule.get("semantic_score", 0),
        "cand_years": int(round(work_years)),
        "criteria_results": results,
        "experience_months": ev["experience"]["work"],
        "internship_months": ev["experience"]["internship"],
        "experience_detail": experience_detail(ev["experience"]),
        "knockout_failed": bool(ev["knockouts_failed"]),
        "must_met": ev["must_met"],
        "must_total": ev["must_total"],
        "current_title": ev.get("current_title") or None,
        "engine_meta": ev.get("meta"),
        "raw": raw,
    }, ev["engine"] == "ai"


def log_engine_run(cur, scan_id, filename, meta):
    """Store one engine run for the Engine health page. Never breaks a scan."""
    if not meta:
        return
    try:
        cur.execute("SAVEPOINT sg_log")
    except Exception:
        return
    try:
        cur.execute(
            "INSERT INTO engine_runs (created_at, scan_id, kind, path, model, filename, input_tokens, "
            "output_tokens, cache_read_tokens, cache_write_tokens, seconds, cost_usd, attempts, error) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (datetime.utcnow().isoformat(), scan_id, meta.get("kind", "resume"), meta.get("path", "rule"),
             meta.get("model"), (filename or "")[:255], meta.get("input_tokens", 0), meta.get("output_tokens", 0),
             meta.get("cache_read_tokens", 0), meta.get("cache_write_tokens", 0), meta.get("seconds", 0),
             meta.get("cost_usd", 0), meta.get("attempts", 0), (meta.get("error") or None)),
        )
        cur.execute("RELEASE SAVEPOINT sg_log")
    except Exception as e:  # logging must never fail a scan: undo only the log row
        print(f"[screen-genie] engine_runs insert failed: {type(e).__name__}: {e}", file=sys.stderr)
        try:
            cur.execute("ROLLBACK TO SAVEPOINT sg_log")
        except Exception:
            pass


@app.template_filter("duration")
def duration_filter(months):
    """{{ 113|duration }} -> "9 yrs 5 mos"."""
    import experience
    return experience.format_duration(months)


def experience_detail(exp):
    """What the candidate page shows under "How experience was calculated"."""
    import experience
    rows = [{"title": r.get("title") or "", "dates": r.get("text") or "", "from": r.get("from"), "to": r.get("to"),
             "months": r.get("months", 0), "kind": r.get("kind"), "reason": r.get("reason") or "",
             "precise": bool(r.get("precise", True))} for r in (exp.get("roles") or [])][:25]
    note = experience.summary_note(exp)
    if exp.get("cross_check"):
        note = (note + " " + exp["cross_check"]).strip()
    return {"roles": rows, "overlap": exp.get("overlap", 0), "approximate": bool(exp.get("approximate")),
            "stated_years": exp.get("stated_years"), "note": note}


def insert_candidate(cur, scan_id, c):
    cur.execute(
        "INSERT INTO candidates (scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_keywords, missing_keywords, matched_skills, missing_skills, "
        "extra_skills, reasons, file_data, suggestions, passed_checks, warning_checks, "
        "issue_checks, semantic_score, cand_years, criteria_results, experience_months, "
        "internship_months, knockout_failed, must_met, must_total, current_title, experience_detail) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (scan_id, c["filename"], c["name"], c["email"], c["phone"], c["score"], c["grade"],
         c["status"], json.dumps(c["section_scores"]), json.dumps(c["matched_keywords"]),
         json.dumps(c["missing_keywords"]), json.dumps(c["matched_skills"]),
         json.dumps(c["missing_skills"]), json.dumps(c["extra_skills"]),
         json.dumps(c["reasons"]), c["raw"],
         json.dumps(c["suggestions"]), json.dumps(c["passed_checks"]),
         json.dumps(c["warning_checks"]), json.dumps(c["issue_checks"]),
         c["semantic_score"], c.get("cand_years"),
         json.dumps(c["criteria_results"]) if c.get("criteria_results") is not None else None,
         c.get("experience_months"), c.get("internship_months"), bool(c.get("knockout_failed")),
         c.get("must_met"), c.get("must_total"), c.get("current_title"),
         json.dumps(c["experience_detail"]) if c.get("experience_detail") else None),
    )
    log_engine_run(cur, scan_id, c.get("filename"), c.get("engine_meta"))


@app.route("/requisition/draft", methods=["POST"])
def requisition_draft():
    """Turn an uploaded JD into an editable requirement checklist."""
    jd_file = request.files.get("jd_file")
    if not jd_file or jd_file.filename == "":
        return jsonify({"error": "Please upload a job description file."}), 400
    jd_text = extract_text_from_file(jd_file.filename, jd_file.stream)
    if not jd_text.strip():
        return jsonify({"error": "Could not read text from the job description file."}), 400
    draft = engine.draft_requisition(jd_text, request.form.get("context", ""))
    draft["role_title"] = draft.get("role_title") or guess_role(jd_text, jd_file.filename)
    meta = draft.pop("meta", None)
    try:
        db = get_db()
        cur = db.cursor()
        log_engine_run(cur, None, jd_file.filename, meta)
        db.commit()
    except Exception as e:
        print(f"[screen-genie] could not log requisition run: {e}", file=sys.stderr)
    return jsonify(draft)


# ---------------------------------------------------------------------------
# Batched scan flow — solves Vercel's 4.5 MB request cap and 60 s timeout.
#
#   1. POST /scan/start         → JD only. Creates a pending scan, returns id.
#   2. POST /scan/<id>/batch    → a few resumes at a time (client loops).
#   3. POST /scan/<id>/finalize → compute avg/engine, mark done, return results URL.
#
# Each request carries only a handful of files, so it stays well under both the
# payload limit and the function timeout no matter how many resumes total.
# ---------------------------------------------------------------------------

@app.route("/scan/start", methods=["POST"])
def scan_start():
    jd_file = request.files.get("jd_file")
    if not jd_file or jd_file.filename == "":
        return jsonify({"error": "Please upload a job description file."}), 400

    jd_text = extract_text_from_file(jd_file.filename, jd_file.stream)
    if not jd_text.strip():
        return jsonify({"error": "Could not read text from the job description file."}), 400

    keywords = top_keywords(jd_text, n=15)
    session_name = (request.form.get("session_name") or "").strip()[:120] or None
    # The recruiter's reviewed checklist; drafted here if the browser sent none.
    criteria = engine.parse_criteria_json(request.form.get("criteria_json"))
    role_title = (request.form.get("role_title") or "").strip()[:120]
    draft_meta = None
    if not criteria:
        draft = engine.draft_requisition(jd_text)
        criteria = draft["criteria"]
        draft_meta = draft.get("meta")
        role_title = role_title or draft.get("role_title", "")
    jd_role = role_title or guess_role(jd_text, jd_file.filename)

    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO scans (jd_filename, jd_role, keywords, resume_count, avg_score, "
        "created_at, jd_text, status, claude_count, session_name, criteria, engine_version) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (jd_file.filename, jd_role, json.dumps(keywords), 0, 0,
         datetime.utcnow().isoformat(), jd_text, "pending", 0, session_name,
         json.dumps(criteria) if criteria else None, engine.ENGINE_VERSION),
    )
    scan_id = cur.fetchone()[0]
    log_engine_run(cur, scan_id, jd_file.filename, draft_meta)
    db.commit()
    cur.close()
    return jsonify({"scan_id": scan_id})


@app.route("/scan/<int:scan_id>/batch", methods=["POST"])
def scan_batch(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT jd_text, keywords, resume_count, criteria FROM scans WHERE id = %s", (scan_id,))
    row = fetchone_dict(cur)
    if not row or not row["jd_text"]:
        return jsonify({"error": "Scan not found."}), 404
    # Defensive cap — max 50 resumes per scan even if the client is bypassed
    if (row["resume_count"] or 0) >= 50:
        return jsonify({"processed": 0, "skipped": 0, "capped": True})
    jd_text = row["jd_text"]
    keywords = json.loads(row["keywords"])
    criteria = json.loads(row["criteria"]) if row.get("criteria") else None

    files = [f for f in request.files.getlist("resume_files") if f and f.filename]
    docs = []
    for f in files:
        raw = f.read()
        text = extract_text_from_file(f.filename, io.BytesIO(raw))
        if text.strip():
            docs.append((f.filename, text, raw))

    processed, claude_in_batch = 0, 0
    if docs:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(docs))) as pool:
            futures = [pool.submit(process_resume_doc, jd_text, keywords, n, t, r, criteria)
                       for n, t, r in docs]
            for fut in concurrent.futures.as_completed(futures):
                cand, used_claude = fut.result()
                insert_candidate(cur, scan_id, cand)
                processed += 1
                if used_claude:
                    claude_in_batch += 1

    cur.execute(
        "UPDATE scans SET resume_count = resume_count + %s, claude_count = claude_count + %s "
        "WHERE id = %s",
        (processed, claude_in_batch, scan_id),
    )
    db.commit()
    cur.close()
    return jsonify({"processed": processed, "skipped": len(files) - processed})


@app.route("/scan/<int:scan_id>/finalize", methods=["POST"])
def scan_finalize(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT resume_count, claude_count, created_at, "
        "COALESCE((SELECT ROUND(AVG(score)::numeric,1) FROM candidates WHERE scan_id=%s),0) AS avg "
        "FROM scans WHERE id = %s",
        (scan_id, scan_id),
    )
    row = fetchone_dict(cur)
    if not row:
        return jsonify({"error": "Scan not found."}), 404

    total = row["resume_count"] or 0
    if total == 0:
        # nothing was screenable — delete the empty scan
        cur.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
        db.commit()
        cur.close()
        return jsonify({"error": "No readable resumes were found (check formats: PDF, DOCX, TXT)."}), 400

    claude_count = row["claude_count"] or 0
    engine_label = "Screen Genie Engine" if claude_count >= max(1, total // 2) else "Rule-based"
    try:
        started = datetime.fromisoformat(row["created_at"])
        duration = round((datetime.utcnow() - started).total_seconds(), 1)
    except Exception:
        duration = None

    cur.execute(
        "UPDATE scans SET avg_score = %s, engine = %s, duration_seconds = %s, status = 'done' "
        "WHERE id = %s",
        (float(row["avg"]), engine_label, duration, scan_id),
    )
    db.commit()
    cur.close()
    return jsonify({"redirect": url_for("results", scan_id=scan_id)})


@app.route("/scan/<int:scan_id>/cancel", methods=["POST"])
def scan_cancel(scan_id):
    """Remove a half-finished scan if the user aborts mid-upload."""
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM candidates WHERE scan_id = %s", (scan_id,))
    cur.execute("DELETE FROM scans WHERE id = %s AND status = 'pending'", (scan_id,))
    db.commit()
    cur.close()
    return jsonify({"ok": True})


# Legacy single-request scan — still used for ZIP uploads (one file, extracted
# server-side). Large ZIPs remain bound by the 60 s limit, so the UI steers big
# batches toward the file picker (which streams in batches).
@app.route("/scan", methods=["POST"])
def scan():
    jd_file = request.files.get("jd_file")
    if not jd_file or jd_file.filename == "":
        flash("Please upload a job description file.")
        return redirect(url_for("index"))

    jd_text = extract_text_from_file(jd_file.filename, jd_file.stream)
    if not jd_text.strip():
        flash("Could not read text from the job description file.")
        return redirect(url_for("index"))

    zip_file = request.files.get("resume_zip")
    if not zip_file or zip_file.filename == "":
        flash("Please upload a ZIP file of resumes.")
        return redirect(url_for("index"))
    resume_docs = extract_resumes_from_zip(zip_file.stream)
    resume_docs = [(n, t, r) for n, t, r in resume_docs if t.strip()]
    if not resume_docs:
        flash("No readable resumes were found (check file formats: PDF, DOCX, TXT).")
        return redirect(url_for("index"))

    keywords = top_keywords(jd_text, n=15)
    session_name = (request.form.get("session_name") or "").strip()[:120] or None
    criteria = engine.parse_criteria_json(request.form.get("criteria_json")) or engine.draft_requisition(jd_text)["criteria"]
    jd_role = (request.form.get("role_title") or "").strip()[:120] or guess_role(jd_text, jd_file.filename)
    start = time.time()

    candidates, claude_count = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(process_resume_doc, jd_text, keywords, n, t, r, criteria)
                   for n, t, r in resume_docs]
        for fut in concurrent.futures.as_completed(futures):
            cand, used_claude = fut.result()
            candidates.append(cand)
            if used_claude:
                claude_count += 1

    candidates.sort(key=lambda c: (c["score"], c.get("semantic_score", 0)), reverse=True)
    avg_score = round(sum(c["score"] for c in candidates) / len(candidates), 1)
    duration = round(time.time() - start, 1)
    engine_label = "Screen Genie Engine" if claude_count >= max(1, len(candidates) // 2) else "Rule-based"

    db = get_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO scans (jd_filename, jd_role, keywords, resume_count, avg_score, "
        "created_at, duration_seconds, engine, status, session_name, criteria, engine_version) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'done',%s,%s,%s) RETURNING id",
        (jd_file.filename, jd_role, json.dumps(keywords), len(candidates), avg_score,
         datetime.utcnow().isoformat(), duration, engine_label, session_name,
         json.dumps(criteria) if criteria else None, engine.ENGINE_VERSION),
    )
    scan_id = cur.fetchone()[0]
    for c in candidates:
        insert_candidate(cur, scan_id, c)
    db.commit()
    cur.close()
    return redirect(url_for("results", scan_id=scan_id))


@app.route("/results/<int:scan_id>")
def results(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at, "
        "duration_seconds, engine, session_name, criteria FROM scans WHERE id = %s",
        (scan_id,),
    )
    scan_row = fetchone_dict(cur)
    if not scan_row:
        flash("Scan not found.")
        return redirect(url_for("index"))

    cur.execute(
        "SELECT id, scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_skills, missing_skills, must_met, must_total, knockout_failed, "
        "experience_months, current_title FROM candidates "
        "WHERE scan_id = %s ORDER BY score DESC, COALESCE(semantic_score,0) DESC, id ASC",
        (scan_id,),
    )
    candidate_rows = fetchall_dict(cur)
    cur.close()

    candidates = []
    for c in candidate_rows:
        c["section_scores"] = json.loads(c["section_scores"])
        c["matched_skills"] = json.loads(c["matched_skills"])
        c["missing_skills"] = json.loads(c["missing_skills"])
        candidates.append(c)

    keywords = json.loads(scan_row["keywords"])
    criteria = json.loads(scan_row["criteria"]) if scan_row.get("criteria") else []
    return render_template("results.html", scan=scan_row, candidates=candidates, keywords=keywords,
                           criteria=criteria)


def _priority_for(status, score):
    """Recruiter priority derived from status/score."""
    if status == "Shortlist" or score >= 70:
        return "High"
    if status == "Review" or score >= 45:
        return "Medium"
    return "Low"


EXPORT_COLUMNS = [
    ("Rank", 6), ("Priority", 10), ("Score", 7), ("Grade", 7), ("Status", 11),
    ("Candidate Name", 24), ("Email", 30), ("Phone", 18), ("Experience", 16), ("Experience (months)", 12),
    ("Skills Matched", 14), ("Skills Missing", 14), ("Matched Skills", 46),
    ("Missing Skills", 46), ("Resume File", 30),
]


def _export_rows(scan_id):
    """Fetch scan + candidate rows shaped for export. Returns (scan, rows)."""
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, session_name, avg_score, resume_count, "
                "created_at FROM scans WHERE id = %s", (scan_id,))
    scan = fetchone_dict(cur)
    if not scan:
        cur.close()
        return None, None
    cur.execute(
        "SELECT score, grade, status, name, email, phone, filename, cand_years, experience_months, "
        "matched_skills, missing_skills FROM candidates "
        "WHERE scan_id = %s ORDER BY score DESC, COALESCE(semantic_score,0) DESC, id ASC",
        (scan_id,),
    )
    rows = fetchall_dict(cur)
    cur.close()

    out = []
    for i, r in enumerate(rows, 1):
        matched = json.loads(r["matched_skills"]); missing = json.loads(r["missing_skills"])
        yrs = r.get("cand_years")
        out.append({
            "rank": i,
            "priority": _priority_for(r["status"], r["score"]),
            "score": r["score"], "grade": r["grade"], "status": r["status"],
            "name": r["name"], "email": r["email"], "phone": r["phone"],
            "experience": (duration_filter(r["experience_months"]) if r.get("experience_months") is not None
                           else (f"{yrs} yrs" if yrs is not None else "—")),
            "experience_months": (r["experience_months"] if r.get("experience_months") is not None
                                  else (yrs * 12 if yrs is not None else None)),
            "n_matched": len(matched), "n_missing": len(missing),
            "matched": ", ".join(matched), "missing": ", ".join(missing),
            "file": r["filename"],
        })
    return scan, out


@app.route("/results/<int:scan_id>/export.xlsx")
def export_results_xlsx(scan_id):
    scan, rows = _export_rows(scan_id)
    if scan is None:
        abort(404)

    title = scan.get("session_name") or scan.get("jd_role") or scan.get("jd_filename") or "Screening"

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "Candidates"

        # ── Title / meta banner ──
        ncols = len(EXPORT_COLUMNS)
        ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
        c = ws.cell(row=1, column=1, value=f"Screen Genie — {title}")
        c.font = Font(bold=True, size=15, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="111114")
        c.alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[1].height = 26

        ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
        meta = (f"JD: {scan.get('jd_filename','')}   ·   Role: {scan.get('jd_role','')}   ·   "
                f"Candidates: {scan.get('resume_count',len(rows))}   ·   Avg score: {scan.get('avg_score','')}%   ·   "
                f"Date: {str(scan.get('created_at',''))[:10]}")
        c2 = ws.cell(row=2, column=1, value=meta)
        c2.font = Font(size=10, color="555555")
        ws.row_dimensions[2].height = 18

        # ── Header row ──
        header_row = 4
        thin = Side(style="thin", color="D9D9E3")
        border = Border(left=thin, right=thin, top=thin, bottom=thin)
        for col, (name, width) in enumerate(EXPORT_COLUMNS, 1):
            cell = ws.cell(row=header_row, column=col, value=name)
            cell.font = Font(bold=True, color="FFFFFF", size=11)
            cell.fill = PatternFill("solid", fgColor="4B4BE6")
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            cell.border = border
            ws.column_dimensions[get_column_letter(col)].width = width
        ws.row_dimensions[header_row].height = 24

        prio_fill = {"High": "E6F4EA", "Medium": "FDF3E0", "Low": "FDECEB"}
        prio_font = {"High": "137339", "Medium": "8A5A10", "Low": "A3271E"}

        for idx, r in enumerate(rows):
            rr = header_row + 1 + idx
            values = [r["rank"], r["priority"], r["score"], r["grade"], r["status"],
                      r["name"], r["email"], r["phone"], r["experience"], r["experience_months"],
                      r["n_matched"], r["n_missing"], r["matched"], r["missing"], r["file"]]
            for col, val in enumerate(values, 1):
                cell = ws.cell(row=rr, column=col, value=val)
                cell.border = border
                cell.alignment = Alignment(
                    horizontal="center" if col in (1, 2, 3, 4, 5, 9, 10, 11, 12) else "left",
                    vertical="top", wrap_text=col in (13, 14),
                )
                if col == 2:  # priority colouring
                    cell.fill = PatternFill("solid", fgColor=prio_fill[r["priority"]])
                    cell.font = Font(bold=True, color=prio_font[r["priority"]])
                if idx % 2 == 1 and col != 2:  # zebra striping
                    cell.fill = PatternFill("solid", fgColor="F7F7FB")

        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
        ws.auto_filter.ref = f"A{header_row}:{get_column_letter(ncols)}{header_row + len(rows)}"

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)
        safe = "".join(ch for ch in title if ch.isalnum() or ch in " -_")[:40].strip() or "screening"
        return Response(
            buf.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="ScreenGenie_{safe}.xlsx"'},
        )
    except ImportError:
        # openpyxl unavailable → structured CSV fallback with the same columns
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow([f"Screen Genie — {title}"])
        w.writerow([f"JD: {scan.get('jd_filename','')}", f"Role: {scan.get('jd_role','')}",
                    f"Candidates: {scan.get('resume_count',len(rows))}", f"Avg: {scan.get('avg_score','')}%"])
        w.writerow([])
        w.writerow([n for n, _ in EXPORT_COLUMNS])
        for r in rows:
            w.writerow([r["rank"], r["priority"], r["score"], r["grade"], r["status"],
                        r["name"], r["email"], r["phone"], r["experience"], r["experience_months"],
                        r["n_matched"], r["n_missing"], r["matched"], r["missing"], r["file"]])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename=ScreenGenie_scan_{scan_id}.csv"})


# Back-compat: old CSV link now redirects to the richer Excel export
@app.route("/results/<int:scan_id>/export.csv")
def export_results_csv(scan_id):
    return redirect(url_for("export_results_xlsx", scan_id=scan_id))


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_keywords, missing_keywords, matched_skills, missing_skills, "
        "extra_skills, reasons, suggestions, passed_checks, warning_checks, issue_checks, "
        "semantic_score, cand_years, criteria_results, experience_months, internship_months, "
        "knockout_failed, must_met, must_total, current_title, experience_detail, "
        "(file_data IS NOT NULL) AS has_file FROM candidates WHERE id = %s",
        (candidate_id,),
    )
    row = fetchone_dict(cur)
    if not row:
        flash("Candidate not found.")
        return redirect(url_for("index"))

    candidate = row
    candidate["section_scores"] = json.loads(candidate["section_scores"])
    candidate["matched_keywords"] = json.loads(candidate["matched_keywords"])
    candidate["missing_keywords"] = json.loads(candidate["missing_keywords"])
    candidate["matched_skills"] = json.loads(candidate["matched_skills"])
    candidate["missing_skills"] = json.loads(candidate["missing_skills"])
    candidate["extra_skills"] = json.loads(candidate["extra_skills"])
    candidate["reasons"] = json.loads(candidate["reasons"])
    candidate["suggestions"] = json.loads(candidate["suggestions"]) if candidate["suggestions"] else []
    candidate["passed_checks"] = json.loads(candidate["passed_checks"]) if candidate["passed_checks"] else []
    candidate["warning_checks"] = json.loads(candidate["warning_checks"]) if candidate["warning_checks"] else []
    candidate["issue_checks"] = json.loads(candidate["issue_checks"]) if candidate["issue_checks"] else []
    candidate["criteria_results"] = json.loads(candidate["criteria_results"]) if candidate.get("criteria_results") else []
    candidate["experience_detail"] = json.loads(candidate["experience_detail"]) if candidate.get("experience_detail") else None

    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at, session_name FROM scans WHERE id = %s", (candidate["scan_id"],))
    scan_row = fetchone_dict(cur)

    # Prev / next candidate within the same scan, ranked by score (desc).
    cur.execute(
        "SELECT id FROM candidates WHERE scan_id = %s ORDER BY score DESC, COALESCE(semantic_score,0) DESC, id ASC",
        (candidate["scan_id"],),
    )
    ordered_ids = [r[0] for r in cur.fetchall()]
    cur.close()

    prev_id = next_id = None
    rank = None
    if candidate["id"] in ordered_ids:
        idx = ordered_ids.index(candidate["id"])
        rank = idx + 1
        if idx > 0:
            prev_id = ordered_ids[idx - 1]
        if idx < len(ordered_ids) - 1:
            next_id = ordered_ids[idx + 1]

    return render_template(
        "candidate.html", candidate=candidate, scan=scan_row,
        prev_id=prev_id, next_id=next_id, rank=rank, total=len(ordered_ids),
    )


def _send_resume(candidate_id, as_attachment):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT filename, file_data FROM candidates WHERE id = %s", (candidate_id,))
    row = fetchone_dict(cur)
    cur.close()
    if not row or row["file_data"] is None:
        abort(404)
    mimetype, _ = mimetypes.guess_type(row["filename"])
    file_bytes = bytes(row["file_data"])
    return send_file(
        io.BytesIO(file_bytes),
        mimetype=mimetype or "application/octet-stream",
        as_attachment=as_attachment,
        download_name=row["filename"],
    )


@app.route("/candidate/<int:candidate_id>/view")
def candidate_view_file(candidate_id):
    resp = _send_resume(candidate_id, as_attachment=False)
    # Uploaded files are untrusted: never let one run scripts on our origin.
    resp.headers["Content-Security-Policy"] = "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:"
    return resp


@app.route("/candidate/<int:candidate_id>/download")
def candidate_download_file(candidate_id):
    return _send_resume(candidate_id, as_attachment=True)


@app.route("/history")
def history():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at, session_name FROM scans WHERE status IS DISTINCT FROM 'pending' ORDER BY created_at DESC")
    scans = fetchall_dict(cur)
    cur.execute("SELECT id, jd_filename, detected_role, overall_score, seniority_level, created_at FROM jd_analyses ORDER BY created_at DESC")
    jd_analyses = fetchall_dict(cur)
    cur.close()
    return render_template("history.html", scans=scans, jd_analyses=jd_analyses)


@app.route("/jd-analysis/<int:analysis_id>/delete", methods=["POST"])
def delete_jd_analysis(analysis_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM jd_fixes WHERE analysis_id = %s", (analysis_id,))
    cur.execute("DELETE FROM jd_analyses WHERE id = %s", (analysis_id,))
    db.commit()
    cur.close()
    flash("JD analysis deleted.")
    return redirect(url_for("history"))


@app.route("/history/<int:scan_id>/delete", methods=["POST"])
def delete_scan(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("DELETE FROM candidates WHERE scan_id = %s", (scan_id,))
    cur.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
    db.commit()
    cur.close()
    flash("Scan deleted.")
    return redirect(url_for("history"))


@app.route("/engine-health")
def engine_health():
    """Is Claude really screening, how fast, and what does it cost? (Day 1)"""
    from datetime import timedelta
    db = get_db()
    cur = db.cursor()
    since7 = (datetime.utcnow() - timedelta(days=7)).isoformat()
    since14 = (datetime.utcnow() - timedelta(days=14)).isoformat()

    cur.execute("""
        SELECT COUNT(*) AS runs,
               COUNT(*) FILTER (WHERE path = 'ai') AS ai,
               COUNT(*) FILTER (WHERE path <> 'ai') AS rule,
               COALESCE(SUM(cost_usd), 0) AS cost,
               COALESCE(AVG(cost_usd) FILTER (WHERE path = 'ai'), 0) AS cost_per_ai,
               percentile_cont(0.5) WITHIN GROUP (ORDER BY seconds) FILTER (WHERE path = 'ai') AS p50,
               percentile_cont(0.95) WITHIN GROUP (ORDER BY seconds) FILTER (WHERE path = 'ai') AS p95,
               COALESCE(AVG(input_tokens + cache_read_tokens + cache_write_tokens) FILTER (WHERE path = 'ai'), 0) AS avg_in,
               COALESCE(AVG(output_tokens) FILTER (WHERE path = 'ai'), 0) AS avg_out,
               COALESCE(SUM(cache_read_tokens), 0) AS cache_read,
               COALESCE(SUM(input_tokens + cache_read_tokens + cache_write_tokens), 0) AS all_in
        FROM engine_runs WHERE kind = 'resume' AND created_at >= %s
    """, (since7,))
    r = {k: (float(v) if v is not None and k not in ("runs","ai","rule") else v) for k, v in fetchone_dict(cur).items()}
    cur.execute("""
        SELECT COUNT(*) AS runs, COUNT(*) FILTER (WHERE path = 'ai') AS ai
        FROM engine_runs WHERE kind = 'requisition' AND created_at >= %s
    """, (since7,))
    q = fetchone_dict(cur)
    cur.execute("""
        SELECT SUBSTRING(created_at, 1, 10) AS day,
               COUNT(*) FILTER (WHERE kind = 'resume') AS resumes,
               COUNT(*) FILTER (WHERE kind = 'resume' AND path = 'ai') AS ai,
               COUNT(*) FILTER (WHERE kind = 'resume' AND path <> 'ai') AS rule,
               COALESCE(SUM(cost_usd), 0) AS cost
        FROM engine_runs WHERE created_at >= %s
        GROUP BY 1 ORDER BY 1 DESC
    """, (since14,))
    days = [dict(d, cost=float(d["cost"] or 0)) for d in fetchall_dict(cur)]
    cur.execute("""
        SELECT created_at, kind, path, filename, scan_id, error
        FROM engine_runs WHERE error IS NOT NULL
        ORDER BY id DESC LIMIT 25
    """)
    failures = fetchall_dict(cur)
    cur.close()

    usd_inr = float(os.environ.get("SG_USD_INR", "88"))
    runs = r["runs"] or 0
    stats = {
        "runs": runs,
        "ai": r["ai"] or 0,
        "rule": r["rule"] or 0,
        "ai_share": round(100.0 * (r["ai"] or 0) / runs) if runs else None,
        "p50": round(r["p50"], 1) if r["p50"] is not None else None,
        "p95": round(r["p95"], 1) if r["p95"] is not None else None,
        "cost_inr": round(float(r["cost"]) * usd_inr, 2),
        "cost_per_resume_inr": round(float(r["cost_per_ai"]) * usd_inr, 2) if r["ai"] else None,
        "avg_in": int(r["avg_in"] or 0),
        "avg_out": int(r["avg_out"] or 0),
        "cache_share": round(100.0 * r["cache_read"] / r["all_in"]) if r["all_in"] else None,
        "req_runs": q["runs"] or 0,
        "req_ai": q["ai"] or 0,
    }
    config = {
        "model": engine.MODEL,
        "effort": engine.EFFORT,
        "api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "parallel": os.environ.get("SG_PARALLEL", "5"),
        "timeout": engine.CALL_TIMEOUT,
        "version": engine.ENGINE_VERSION,
        "usd_inr": usd_inr,
        "price_in": engine.PRICE_IN,
        "price_out": engine.PRICE_OUT,
    }
    return render_template("engine_health.html", active="engine", stats=stats, days=days,
                           failures=failures, config=config)


@app.route("/dashboard")
def dashboard():
    db = get_db()
    cur = db.cursor()

    # All KPIs computed in a single SQL pass — no longer pulls every candidate
    # row into Python (which was the slow part as data grew).
    cur.execute("""
        SELECT
            COUNT(*)                                          AS total_resumes,
            COALESCE(ROUND(AVG(score)::numeric, 1), 0)        AS avg_score,
            COUNT(*) FILTER (WHERE status = 'Shortlist')      AS shortlisted,
            COUNT(*) FILTER (WHERE score < 40)                AS b0,
            COUNT(*) FILTER (WHERE score >= 40 AND score < 60) AS b1,
            COUNT(*) FILTER (WHERE score >= 60 AND score < 80) AS b2,
            COUNT(*) FILTER (WHERE score >= 80)               AS b3,
            COUNT(*) FILTER (WHERE grade = 'A')               AS ga,
            COUNT(*) FILTER (WHERE grade = 'B')               AS gb,
            COUNT(*) FILTER (WHERE grade = 'C')               AS gc,
            COUNT(*) FILTER (WHERE grade = 'D')               AS gd,
            COUNT(*) FILTER (WHERE grade = 'F')               AS gf
        FROM candidates
    """)
    agg = fetchone_dict(cur)

    cur.execute("SELECT COUNT(*) FROM scans WHERE status IS DISTINCT FROM 'pending'")
    total_scans = cur.fetchone()[0]

    total_resumes = agg["total_resumes"] or 0
    avg_score = float(agg["avg_score"]) if total_resumes else 0
    shortlisted = agg["shortlisted"] or 0
    buckets = {"0-39": agg["b0"], "40-59": agg["b1"], "60-79": agg["b2"], "80-100": agg["b3"]}
    grade_counts = {"A": agg["ga"], "B": agg["gb"], "C": agg["gc"], "D": agg["gd"], "F": agg["gf"]}

    cur.execute("SELECT id, scan_id, filename, name, score, grade, status FROM candidates ORDER BY score DESC, COALESCE(semantic_score,0) DESC, id ASC LIMIT 8")
    top_candidates = fetchall_dict(cur)
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans WHERE status IS DISTINCT FROM 'pending' ORDER BY created_at DESC LIMIT 5")
    recent_scans = fetchall_dict(cur)

    # ── JD Analysis KPIs (single SQL pass) ──
    cur.execute("""
        SELECT
            COUNT(*)                                              AS total_jds,
            COALESCE(ROUND(AVG(overall_score)::numeric, 1), 0)    AS avg_jd_score,
            COUNT(*) FILTER (WHERE overall_score >= 80)           AS excellent_jds,
            COUNT(*) FILTER (WHERE overall_score < 50)            AS weak_jds,
            COUNT(*) FILTER (WHERE overall_score < 40)            AS jb0,
            COUNT(*) FILTER (WHERE overall_score >= 40 AND overall_score < 60) AS jb1,
            COUNT(*) FILTER (WHERE overall_score >= 60 AND overall_score < 80) AS jb2,
            COUNT(*) FILTER (WHERE overall_score >= 80)           AS jb3
        FROM jd_analyses
    """)
    jagg = fetchone_dict(cur)

    # Improved-JD count (how many analyses were turned into a fixed JD)
    cur.execute("SELECT COUNT(DISTINCT analysis_id) FROM jd_fixes")
    jds_improved = cur.fetchone()[0] or 0

    total_jds = jagg["total_jds"] or 0
    jd_stats = {
        "total_jds": total_jds,
        "avg_jd_score": float(jagg["avg_jd_score"]) if total_jds else 0,
        "excellent_jds": jagg["excellent_jds"] or 0,
        "weak_jds": jagg["weak_jds"] or 0,
        "jds_improved": jds_improved,
    }
    jd_buckets = {"0-39": jagg["jb0"], "40-59": jagg["jb1"], "60-79": jagg["jb2"], "80-100": jagg["jb3"]}

    cur.execute(
        "SELECT id, jd_filename, detected_role, overall_score, seniority_level, created_at "
        "FROM jd_analyses ORDER BY created_at DESC LIMIT 5"
    )
    recent_jds = fetchall_dict(cur)
    cur.close()

    return render_template(
        "dashboard.html",
        total_resumes=total_resumes, total_scans=total_scans,
        shortlisted=shortlisted, avg_score=avg_score,
        buckets=buckets, grade_counts=grade_counts,
        top_candidates=top_candidates, recent_scans=recent_scans,
        jd_stats=jd_stats, jd_buckets=jd_buckets, recent_jds=recent_jds,
    )


# ── JD Analysis ──────────────────────────────────────────────────────────────

@app.route("/jd-analysis")
def jd_analysis():
    return render_template("jd_analysis.html")


@app.route("/jd-analyze", methods=["POST"])
def jd_analyze():
    jd_file = request.files.get("jd_file")
    description = request.form.get("description", "").strip()

    if not jd_file or jd_file.filename == "":
        flash("Please upload a JD file.")
        return redirect(url_for("jd_analysis"))

    jd_text = extract_text_from_file(jd_file.filename, jd_file.stream)
    if not jd_text.strip():
        flash("Could not read text from the JD file.")
        return redirect(url_for("jd_analysis"))

    document_loading_overlay = True  # noqa — triggers loading overlay in template
    result = analyze_jd_quality_with_claude(jd_text, description) or analyze_jd_quality(jd_text)

    scores = {
        "clarity": result["clarity_score"],
        "completeness": result["completeness_score"],
        "keyword_richness": result["keyword_richness_score"],
        "role_definition": result["role_definition_score"],
        "attractiveness": result["attractiveness_score"],
        "seniority_alignment": result["seniority_alignment_score"],
    }

    db = connect_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO jd_analyses (jd_filename, jd_text, description, overall_score, scores, "
        "missing_sections, missing_keywords, issues, strengths, suggestions, "
        "seniority_level, detected_role, created_at) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (jd_file.filename, jd_text, description, result["overall_score"],
         json.dumps(scores), json.dumps(result["missing_sections"]),
         json.dumps(result["missing_keywords"]), json.dumps(result["issues"]),
         json.dumps(result["strengths"]), json.dumps(result["suggestions"]),
         result["seniority_level"], result["detected_role"],
         datetime.utcnow().isoformat()),
    )
    analysis_id = cur.fetchone()[0]
    db.commit()
    cur.close()
    db.close()
    return redirect(url_for("jd_analysis_result", analysis_id=analysis_id))


@app.route("/jd-analysis/<int:analysis_id>")
def jd_analysis_result(analysis_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM jd_analyses WHERE id = %s", (analysis_id,))
    row = fetchone_dict(cur)
    if not row:
        flash("Analysis not found.")
        return redirect(url_for("jd_analysis"))
    cur.execute(
        "SELECT id FROM jd_fixes WHERE analysis_id = %s ORDER BY id DESC LIMIT 1",
        (analysis_id,),
    )
    fix_row = cur.fetchone()
    fix_id = fix_row[0] if fix_row else None
    cur.close()

    row["scores"] = json.loads(row["scores"])
    row["missing_sections"] = json.loads(row["missing_sections"])
    row["missing_keywords"] = json.loads(row["missing_keywords"])
    row["issues"] = json.loads(row["issues"])
    row["strengths"] = json.loads(row["strengths"])
    row["suggestions"] = json.loads(row["suggestions"])
    return render_template("jd_analysis.html", analysis=row, fix_id=fix_id)


@app.route("/jd-analysis/<int:analysis_id>/fix", methods=["POST"])
def jd_fix(analysis_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM jd_analyses WHERE id = %s", (analysis_id,))
    row = fetchone_dict(cur)
    cur.close()
    if not row:
        flash("Analysis not found.")
        return redirect(url_for("jd_analysis"))

    analysis_data = {
        "issues": json.loads(row["issues"]),
        "suggestions": json.loads(row["suggestions"]),
        "missing_keywords": json.loads(row["missing_keywords"]),
        "missing_sections": json.loads(row["missing_sections"]),
    }

    fixed_text = fix_jd_with_claude(row["jd_text"], row["description"], analysis_data)
    if not fixed_text:
        flash("Could not generate improved JD. Please try again.")
        return redirect(url_for("jd_analysis_result", analysis_id=analysis_id))

    db2 = connect_db()
    cur2 = db2.cursor()
    cur2.execute(
        "INSERT INTO jd_fixes (analysis_id, fixed_jd_text, created_at) "
        "VALUES (%s,%s,%s) RETURNING id",
        (analysis_id, fixed_text, datetime.utcnow().isoformat()),
    )
    fix_id = cur2.fetchone()[0]
    db2.commit()
    cur2.close()
    db2.close()
    return redirect(url_for("jd_fix_result", fix_id=fix_id))


@app.route("/jd-fix/<int:fix_id>")
def jd_fix_result(fix_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT f.id, f.fixed_jd_text, f.created_at, f.analysis_id, "
        "a.detected_role, a.jd_filename, a.overall_score "
        "FROM jd_fixes f JOIN jd_analyses a ON a.id = f.analysis_id "
        "WHERE f.id = %s",
        (fix_id,),
    )
    row = fetchone_dict(cur)
    cur.close()
    if not row:
        flash("Fix not found.")
        return redirect(url_for("jd_analysis"))
    return render_template("jd_fix.html", fix=row)


@app.route("/jd-fix/<int:fix_id>/download/pdf")
def jd_fix_download_pdf(fix_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT fixed_jd_text FROM jd_fixes WHERE id = %s", (fix_id,))
    row = fetchone_dict(cur)
    cur.close()
    if not row:
        abort(404)

    try:
        from fpdf import FPDF

        def _safe(text):
            return text.encode("latin-1", errors="replace").decode("latin-1")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.set_margins(18, 18, 18)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(150, 150, 150)
        pdf.cell(0, 6, "Generated by Screen Genie", new_x="LMARGIN", new_y="NEXT", align="R")
        pdf.ln(4)

        for line in row["fixed_jd_text"].split("\n"):
            stripped = line.strip()
            if not stripped:
                pdf.ln(3)
                continue
            if stripped.startswith("# ") or (stripped.isupper() and 3 < len(stripped) < 80):
                pdf.set_font("Helvetica", "B", 15)
                pdf.set_text_color(17, 17, 20)
                pdf.multi_cell(0, 9, _safe(stripped.lstrip("# ")))
                pdf.ln(2)
            elif stripped.startswith("## ") or (stripped.endswith(":") and len(stripped) < 55 and not stripped.startswith("-")):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(37, 99, 235)
                pdf.multi_cell(0, 8, _safe(stripped.lstrip("# ").rstrip(":") + ":"))
                pdf.set_text_color(17, 17, 20)
                pdf.ln(1)
            elif stripped.startswith(("- ", "* ")):
                pdf.set_font("Helvetica", size=11)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 6, _safe("  -  " + stripped.lstrip("-* ")))
            else:
                pdf.set_font("Helvetica", size=11)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 6, _safe(stripped))

        return Response(
            bytes(pdf.output()),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=improved_jd_{fix_id}.pdf"},
        )
    except Exception:
        return Response(
            row["fixed_jd_text"],
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename=improved_jd_{fix_id}.txt"},
        )


@app.route("/jd-fix/<int:fix_id>/download/json")
def jd_fix_download_json(fix_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT f.fixed_jd_text, f.created_at, a.jd_filename, a.detected_role, "
        "a.overall_score, a.missing_keywords "
        "FROM jd_fixes f JOIN jd_analyses a ON a.id = f.analysis_id "
        "WHERE f.id = %s",
        (fix_id,),
    )
    row = fetchone_dict(cur)
    cur.close()
    if not row:
        abort(404)

    sections = {}
    current_section = "header"
    current_lines = []
    for line in row["fixed_jd_text"].split("\n"):
        s = line.strip()
        if not s:
            continue
        is_heading = s.startswith("#") or (s.endswith(":") and len(s) < 60 and not s.startswith("-"))
        if is_heading:
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = s.lstrip("#").rstrip(":").strip()
            current_lines = []
        else:
            current_lines.append(s)
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()

    output = {
        "metadata": {
            "source_file": row["jd_filename"],
            "detected_role": row["detected_role"],
            "original_quality_score": row["overall_score"],
            "generated_at": row["created_at"],
            "generated_by": "Screen Genie Engine",
        },
        "full_text": row["fixed_jd_text"],
        "sections": sections,
        "added_keywords": json.loads(row["missing_keywords"]) if row["missing_keywords"] else [],
    }
    return Response(
        json.dumps(output, indent=2, ensure_ascii=False),
        mimetype="application/json",
        headers={"Content-Disposition": f"attachment; filename=improved_jd_{fix_id}.json"},
    )


# DB schema is created once per process, lazily, on the first request that
# needs it — NOT at import time. Running CREATE TABLE / ALTER / CREATE INDEX on
# every cold start added latency to every page and could crash the function if
# the DB was briefly slow. The flag ensures the (idempotent) init runs at most
# once per warm instance.
_DB_READY = False

def ensure_db_ready():
    global _DB_READY
    if _DB_READY:
        return
    try:
        init_db()
        _DB_READY = True
    except Exception as _e:
        import sys
        print(f"[screen-genie] DB init warning: {_e}", file=sys.stderr)


@app.before_request
def _before_request_db_init():
    ensure_db_ready()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
