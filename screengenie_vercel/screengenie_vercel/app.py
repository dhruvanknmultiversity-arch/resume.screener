import os
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

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "screen-genie-internal-dev-key")
app.config["MAX_CONTENT_LENGTH"] = 60 * 1024 * 1024  # 60 MB

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
    (Flask's `g`/get_db is request-scoped and must not be shared across threads)."""
    url = get_database_url()
    if not url or "://" not in url:
        raise RuntimeError(f"DATABASE_URL not set or invalid: '{url}'")
    p = parse_db_url(url)
    conn = pg8000.dbapi.connect(
        host=p["host"], port=p["port"], database=p["database"],
        user=p["user"], password=p["password"], ssl_context=True,
    )
    conn.autocommit = False
    return conn


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
    url = get_database_url()
    has_url = bool(url and "://" in url)
    env_keys = [k for k in os.environ if "DATABASE" in k or "POSTGRES" in k or "PG" in k]
    return {"status": "ok", "has_db_url": has_url, "db_env_keys": env_keys}


@app.route("/")
def index():
    return render_template("bulk_screener.html")


# ---------------------------------------------------------------------------
# Synchronous scan — runs entirely within the POST request.
# Works on Vercel (serverless) and locally. The browser shows a full-page
# loading overlay while the server processes; on completion it is redirected
# to /results/<scan_id> automatically.
# ---------------------------------------------------------------------------

def run_scan_sync(jd_file_name, jd_text, jd_role, keywords, resume_docs):
    """Run the full scan synchronously. Returns scan_id."""
    start = time.time()

    def _process_one(args):
        filename, text, raw = args
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
            "raw": raw,
        }, used_claude

    candidates = []
    claude_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(_process_one, doc) for doc in resume_docs]
        for fut in concurrent.futures.as_completed(futures):
            cand, used_claude = fut.result()
            candidates.append(cand)
            if used_claude:
                claude_count += 1

    candidates.sort(key=lambda c: c["score"], reverse=True)
    avg_score = round(sum(c["score"] for c in candidates) / len(candidates), 1)
    duration = round(time.time() - start, 1)
    engine = "Claude AI" if claude_count >= max(1, len(candidates) // 2) else "Rule-based"

    db = connect_db()
    cur = db.cursor()
    cur.execute(
        "INSERT INTO scans (jd_filename, jd_role, keywords, resume_count, avg_score, "
        "created_at, duration_seconds, engine) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (jd_file_name, jd_role, json.dumps(keywords), len(candidates), avg_score,
         datetime.utcnow().isoformat(), duration, engine),
    )
    scan_id = cur.fetchone()[0]
    for c in candidates:
        cur.execute(
            "INSERT INTO candidates (scan_id, filename, name, email, phone, score, grade, status, "
            "section_scores, matched_keywords, missing_keywords, matched_skills, missing_skills, "
            "extra_skills, reasons, file_data, suggestions, passed_checks, warning_checks, "
            "issue_checks, semantic_score) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (scan_id, c["filename"], c["name"], c["email"], c["phone"], c["score"], c["grade"],
             c["status"], json.dumps(c["section_scores"]), json.dumps(c["matched_keywords"]),
             json.dumps(c["missing_keywords"]), json.dumps(c["matched_skills"]),
             json.dumps(c["missing_skills"]), json.dumps(c["extra_skills"]),
             json.dumps(c["reasons"]), c["raw"],
             json.dumps(c["suggestions"]), json.dumps(c["passed_checks"]),
             json.dumps(c["warning_checks"]), json.dumps(c["issue_checks"]),
             c["semantic_score"]),
        )
    db.commit()
    cur.close()
    db.close()
    return scan_id


@app.route("/scan", methods=["POST"])
def scan():
    jd_file = request.files.get("jd_file")
    resume_mode = request.form.get("resume_mode", "files")

    if not jd_file or jd_file.filename == "":
        flash("Please upload a job description file.")
        return redirect(url_for("index"))

    jd_text = extract_text_from_file(jd_file.filename, jd_file.stream)
    if not jd_text.strip():
        flash("Could not read text from the job description file.")
        return redirect(url_for("index"))

    resume_docs = []
    if resume_mode == "zip":
        zip_file = request.files.get("resume_zip")
        if not zip_file or zip_file.filename == "":
            flash("Please upload a ZIP file of resumes.")
            return redirect(url_for("index"))
        resume_docs = extract_resumes_from_zip(zip_file.stream)
    else:
        files = request.files.getlist("resume_files")
        files = [f for f in files if f and f.filename]
        if not files:
            flash("Please select one or more resume files.")
            return redirect(url_for("index"))
        for f in files:
            raw = f.read()
            text = extract_text_from_file(f.filename, io.BytesIO(raw))
            resume_docs.append((f.filename, text, raw))

    resume_docs = [(name, text, raw) for name, text, raw in resume_docs if text.strip()]
    if not resume_docs:
        flash("No readable resumes were found (check file formats: PDF, DOCX, TXT).")
        return redirect(url_for("index"))

    keywords = top_keywords(jd_text, n=15)
    jd_role = guess_role(jd_text, jd_file.filename)

    try:
        scan_id = run_scan_sync(jd_file.filename, jd_text, jd_role, keywords, resume_docs)
    except Exception as e:
        flash(f"Scan failed: {e}")
        return redirect(url_for("index"))

    return redirect(url_for("results", scan_id=scan_id))


@app.route("/scan_progress/<job_id>")
def scan_progress(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify({"status": "missing"}), 404
        total = job["total"]
        done = job["done"]
        elapsed = time.time() - job["started_at"]
        status = job["status"]
        scan_id = job["scan_id"]
        error = job["error"]

    percent = round((done / total) * 100) if total else 0
    if done > 0 and status == "processing":
        eta = max(0, round((elapsed / done) * (total - done)))
    else:
        eta = None
    return jsonify({
        "status": status, "total": total, "done": done, "percent": percent,
        "elapsed": round(elapsed), "eta": eta, "scan_id": scan_id, "error": error,
    })


@app.route("/results/<int:scan_id>")
def results(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at, "
        "duration_seconds, engine FROM scans WHERE id = %s",
        (scan_id,),
    )
    scan_row = fetchone_dict(cur)
    if not scan_row:
        flash("Scan not found.")
        return redirect(url_for("index"))

    cur.execute(
        "SELECT id, scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_skills, missing_skills FROM candidates "
        "WHERE scan_id = %s ORDER BY score DESC",
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
    return render_template("results.html", scan=scan_row, candidates=candidates, keywords=keywords)


@app.route("/results/<int:scan_id>/export.csv")
def export_results_csv(scan_id):
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT jd_role FROM scans WHERE id = %s", (scan_id,))
    scan_row = fetchone_dict(cur)
    if not scan_row:
        abort(404)
    cur.execute(
        "SELECT score, grade, status, name, email, phone, filename, "
        "matched_skills, missing_skills FROM candidates "
        "WHERE scan_id = %s ORDER BY score DESC",
        (scan_id,),
    )
    rows = fetchall_dict(cur)
    cur.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Rank", "Score", "Grade", "Status", "Name", "Email", "Phone",
                     "Resume File", "Matched Skills", "Missing Skills"])
    for i, r in enumerate(rows, 1):
        writer.writerow([
            i, r["score"], r["grade"], r["status"], r["name"], r["email"], r["phone"],
            r["filename"],
            "; ".join(json.loads(r["matched_skills"])),
            "; ".join(json.loads(r["missing_skills"])),
        ])
    csv_data = buf.getvalue()
    fname = f"screen_genie_scan_{scan_id}.csv"
    return Response(
        csv_data, mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )


@app.route("/candidate/<int:candidate_id>")
def candidate_detail(candidate_id):
    db = get_db()
    cur = db.cursor()
    cur.execute(
        "SELECT id, scan_id, filename, name, email, phone, score, grade, status, "
        "section_scores, matched_keywords, missing_keywords, matched_skills, missing_skills, "
        "extra_skills, reasons, suggestions, passed_checks, warning_checks, issue_checks, "
        "semantic_score, (file_data IS NOT NULL) AS has_file FROM candidates WHERE id = %s",
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

    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans WHERE id = %s", (candidate["scan_id"],))
    scan_row = fetchone_dict(cur)

    # Prev / next candidate within the same scan, ranked by score (desc).
    cur.execute(
        "SELECT id FROM candidates WHERE scan_id = %s ORDER BY score DESC, id ASC",
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
    return _send_resume(candidate_id, as_attachment=False)


@app.route("/candidate/<int:candidate_id>/download")
def candidate_download_file(candidate_id):
    return _send_resume(candidate_id, as_attachment=True)


@app.route("/history")
def history():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans ORDER BY created_at DESC")
    scans = fetchall_dict(cur)
    cur.close()
    return render_template("history.html", scans=scans)


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


@app.route("/dashboard")
def dashboard():
    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans")
    scans = fetchall_dict(cur)
    cur.execute("SELECT id, scan_id, filename, name, score, grade, status FROM candidates")
    candidates = fetchall_dict(cur)

    total_resumes = len(candidates)
    total_scans = len(scans)
    shortlisted = sum(1 for c in candidates if c["status"] == "Shortlist")
    avg_score = round(sum(c["score"] for c in candidates) / total_resumes, 1) if total_resumes else 0

    buckets = {"0-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for c in candidates:
        s = c["score"]
        if s < 40: buckets["0-39"] += 1
        elif s < 60: buckets["40-59"] += 1
        elif s < 80: buckets["60-79"] += 1
        else: buckets["80-100"] += 1

    grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for c in candidates:
        grade_counts[c["grade"]] = grade_counts.get(c["grade"], 0) + 1

    cur.execute("SELECT id, scan_id, filename, name, score, grade, status FROM candidates ORDER BY score DESC LIMIT 8")
    top_candidates = fetchall_dict(cur)
    cur.execute("SELECT id, jd_filename, jd_role, keywords, resume_count, avg_score, created_at FROM scans ORDER BY created_at DESC LIMIT 5")
    recent_scans = fetchall_dict(cur)
    cur.close()

    return render_template(
        "dashboard.html",
        total_resumes=total_resumes, total_scans=total_scans,
        shortlisted=shortlisted, avg_score=avg_score,
        buckets=buckets, grade_counts=grade_counts,
        top_candidates=top_candidates, recent_scans=recent_scans,
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

        class _PDF(FPDF):
            def header(self):
                self.set_font("Helvetica", "B", 9)
                self.set_text_color(150, 150, 150)
                self.cell(0, 8, "Generated by Screen Genie", ln=True, align="R")
                self.ln(2)

        pdf = _PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=18)
        pdf.set_margins(18, 18, 18)

        for line in row["fixed_jd_text"].split("\n"):
            stripped = line.strip()
            if not stripped:
                pdf.ln(3)
                continue
            if stripped.startswith("# ") or (stripped.isupper() and len(stripped) < 80):
                pdf.set_font("Helvetica", "B", 15)
                pdf.set_text_color(17, 17, 20)
                pdf.multi_cell(0, 9, stripped.lstrip("# "))
                pdf.ln(2)
            elif stripped.startswith("## ") or (stripped.endswith(":") and len(stripped) < 55):
                pdf.set_font("Helvetica", "B", 12)
                pdf.set_text_color(37, 99, 235)
                pdf.multi_cell(0, 8, stripped.lstrip("# ").rstrip(":") + ":")
                pdf.set_text_color(17, 17, 20)
                pdf.ln(1)
            elif stripped.startswith(("- ", "• ", "* ")):
                pdf.set_font("Helvetica", size=11)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 6, "  •  " + stripped.lstrip("-•* "))
            else:
                pdf.set_font("Helvetica", size=11)
                pdf.set_text_color(30, 30, 30)
                pdf.multi_cell(0, 6, stripped)

        return Response(
            bytes(pdf.output()),
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=improved_jd_{fix_id}.pdf"},
        )
    except ImportError:
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
            "generated_by": "Screen Genie — Claude AI",
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


# Initialize DB on every cold start (Vercel serverless + local)
try:
    init_db()
except Exception as _e:
    import sys
    print(f"[screen-genie] DB init warning: {_e}", file=sys.stderr)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
