"""Screen Genie engine v2 — requirement-based screening with evidence.

Flow
----
1. ``draft_requisition(jd_text, context)`` turns a JD into an editable list of
   requirements (criteria): must-haves, nice-to-haves and knockouts, each with a
   weight and an optional minimum number of years.
2. ``evaluate_resume(jd_text, criteria, resume_text)`` judges one resume against
   every criterion. The model must quote the resume as evidence; a quote that is
   not actually in the resume is downgraded. Experience is computed here, in
   code, from the job dates the model extracts (overlaps counted once, study
   excluded), never taken from the model's own arithmetic.
3. ``score_evaluation`` turns verdicts into one documented score:

       score = 100 * sum(weight_i * value_i) / sum(weight_i)
       value: met = 1, partial = 0.5, not met = 0

   A failed knockout never lowers the score silently; it is flagged.

Bias guard: the candidate's name, email and phone are masked before the resume
text is sent to the model, and the prompt forbids using protected attributes.

Every model call has a rule-based fallback, so screening still works when the
API key is missing or the API fails (the scan is then labelled rule-based).
"""

import json
import os
import re
import sys
import threading
import time
from datetime import date

ENGINE_VERSION = "sg-engine-2.0"
MODEL = os.environ.get("SG_MODEL", "claude-sonnet-5-5")
EFFORT = os.environ.get("SG_EFFORT", "low")          # low | medium | high
CALL_TIMEOUT = float(os.environ.get("SG_TIMEOUT", "55"))

KIND_WEIGHT_DEFAULT = {"knockout": 3, "must": 3, "nice": 1}
VERDICT_VALUE = {"met": 1.0, "partial": 0.5, "not_met": 0.0}

_client = None
_client_lock = threading.Lock()
_semaphore = threading.Semaphore(int(os.environ.get("SG_PARALLEL", "5")))


def log(msg):
    print(f"[screengenie:engine] {msg}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Model access
# ---------------------------------------------------------------------------

def llm_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            import anthropic
            _client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                timeout=CALL_TIMEOUT,
                max_retries=0,
            )
    return _client


def _strip_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            text = text[a:b + 1]
    return text


def call_json(system, user_content, schema, max_tokens=4000, client=None):
    """One structured-output call. Returns the parsed dict, or raises.

    ``output_config`` is sent through ``extra_body`` so this works on any
    version of the anthropic SDK."""
    import anthropic
    client = client or _get_client()
    body = {"output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": EFFORT}}
    delays = [2, 5, 10]
    for attempt in range(len(delays) + 1):
        try:
            with _semaphore:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                    extra_body=body,
                )
            if getattr(resp, "stop_reason", None) == "refusal":
                raise RuntimeError("model refused")
            text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
            if not text:
                raise RuntimeError(f"no text block (stop_reason={getattr(resp, 'stop_reason', None)})")
            return json.loads(_strip_json(text))
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
            if attempt == len(delays):
                raise
            log(f"transient {type(e).__name__}, retrying in {delays[attempt]}s")
            time.sleep(delays[attempt])


# ---------------------------------------------------------------------------
# Criteria (requisition)
# ---------------------------------------------------------------------------

REQUISITION_SCHEMA = {
    "type": "object",
    "properties": {
        "role_title": {"type": "string"},
        "seniority": {"type": "string", "enum": ["Internship", "Entry", "Junior", "Mid", "Senior", "Lead", "Unknown"]},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "kind": {"type": "string", "enum": ["must", "nice", "knockout"]},
                    "weight": {"type": "integer", "enum": [1, 2, 3]},
                    "min_years": {"type": "number"},
                },
                "required": ["label", "kind", "weight", "min_years"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["role_title", "seniority", "criteria"],
    "additionalProperties": False,
}

REQUISITION_SYSTEM = """You turn a job description into a short, checkable screening checklist for recruiters.

Rules:
- 4 to 10 criteria. Each is ONE concrete, verifiable requirement a resume can show or not show
  (a skill, tool, qualification, certification, domain, or amount of experience).
- Never use soft or generic items (teamwork, communication, passion, hard-working, "projects",
  "solutions") unless the JD makes it a specific, checkable requirement.
- kind: "must" = clearly required; "nice" = preferred/bonus; "knockout" = only a hard legal or
  logistical condition stated in the JD (e.g. a required licence, work authorisation, a degree the
  JD calls mandatory). Use knockout rarely.
- weight: 3 = core to the role, 2 = important, 1 = minor.
- min_years: the minimum years the JD asks for this criterion, else 0.
- Group synonyms in one label (e.g. "PyTorch or TensorFlow").
- Never include age, gender, religion, caste, marital status, nationality, photo or similar
  protected attributes."""


def _clean_criteria(items):
    out, seen = [], set()
    for i, c in enumerate(items or []):
        label = re.sub(r"\s+", " ", str(c.get("label", ""))).strip()[:140]
        if not label or label.lower() in seen:
            continue
        seen.add(label.lower())
        kind = c.get("kind") if c.get("kind") in KIND_WEIGHT_DEFAULT else "must"
        try:
            weight = int(c.get("weight") or KIND_WEIGHT_DEFAULT[kind])
        except (TypeError, ValueError):
            weight = KIND_WEIGHT_DEFAULT[kind]
        weight = max(1, min(3, weight))
        try:
            min_years = float(c.get("min_years") or 0)
        except (TypeError, ValueError):
            min_years = 0.0
        min_years = max(0.0, min(30.0, min_years))
        out.append({"id": f"c{len(out) + 1}", "label": label, "kind": kind,
                    "weight": weight, "min_years": min_years})
        if len(out) >= 15:
            break
    return out


def parse_criteria_json(raw):
    """Validate criteria sent by the browser (the recruiter's edited list)."""
    try:
        data = json.loads(raw or "[]")
    except (TypeError, ValueError):
        return []
    return _clean_criteria(data if isinstance(data, list) else [])


def rule_requisition(jd_text):
    from parsing import extract_skills, required_years
    skills = extract_skills(jd_text)
    crit = [{"label": s, "kind": "must" if i < 5 else "nice", "weight": 2 if i < 5 else 1, "min_years": 0}
            for i, s in enumerate(skills[:9])]
    years = required_years(jd_text)
    if years:
        crit.insert(0, {"label": f"At least {years} years of relevant experience", "kind": "must",
                        "weight": 3, "min_years": years})
    return {"role_title": "", "seniority": "Unknown", "criteria": _clean_criteria(crit), "engine": "rule"}


def draft_requisition(jd_text, context="", client=None):
    if client is None and not llm_available():
        return rule_requisition(jd_text)
    try:
        user = (f"Hiring context from the recruiter (may be empty): {context.strip()[:1500]}\n\n"
                f"<job_description>\n{jd_text[:20000]}\n</job_description>\n\n"
                "Return the screening checklist.")
        data = call_json(REQUISITION_SYSTEM, user, REQUISITION_SCHEMA, max_tokens=2000, client=client)
        crit = _clean_criteria(data.get("criteria"))
        if not crit:
            raise RuntimeError("no criteria returned")
        return {"role_title": str(data.get("role_title", ""))[:120],
                "seniority": data.get("seniority", "Unknown"), "criteria": crit, "engine": "ai"}
    except Exception as e:
        log(f"requisition draft failed, using rule-based: {type(e).__name__}: {e}")
        return rule_requisition(jd_text)


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\-\s().]{8,}\d)(?!\d)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+", re.I)


def mask_pii(text, name=None):
    masked = _EMAIL_RE.sub("[EMAIL]", text)
    masked = _URL_RE.sub("[LINK]", masked)
    masked = _PHONE_RE.sub(lambda m: "[PHONE]" if sum(ch.isdigit() for ch in m.group(0)) >= 10 else m.group(0), masked)
    if name and name.strip() and name.strip().lower() not in ("not found", "unknown"):
        for part in sorted({p for p in re.split(r"\s+", name.strip()) if len(p) >= 3}, key=len, reverse=True):
            masked = re.sub(r"(?<![A-Za-z])" + re.escape(part) + r"(?![A-Za-z])", "[NAME]", masked, flags=re.I)
    return masked


# ---------------------------------------------------------------------------
# Experience maths
# ---------------------------------------------------------------------------

_MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun",
                                           "jul", "aug", "sep", "oct", "nov", "dec"])}


def parse_month(value, today=None, is_end=False):
    """'2024-03', '2024', 'Mar 2024', 'present' -> months since year 0, or None."""
    today = today or date.today()
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    if s in ("present", "current", "now", "till date", "ongoing", "today"):
        return today.year * 12 + today.month
    m = re.match(r"^(\d{4})[-/.](\d{1,2})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
    else:
        y_m = re.search(r"(19|20)\d{2}", s)
        if not y_m:
            return None
        y = int(y_m.group(0))
        name = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s)
        mo = _MONTHS[name.group(1)] if name else (12 if is_end else 1)
    if not (1 <= mo <= 12) or not (1970 <= y <= today.year + 1):
        return None
    return y * 12 + mo


def experience_months(roles, today=None):
    """Union of employment intervals in months. Overlapping jobs count once.

    Returns {"total": n, "work": n, "internship": n}; internships are counted
    separately so they never inflate full-time experience."""
    today = today or date.today()
    now = today.year * 12 + today.month

    def union(intervals):
        total, cur_s, cur_e = 0, None, None
        for s, e in sorted(intervals):
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            total += cur_e - cur_s
        return total

    work, intern = [], []
    for r in roles or []:
        s = parse_month(r.get("start"), today)
        e = parse_month(r.get("end"), today, is_end=True)
        if s is None:
            continue
        if e is None:
            e = s + 1
        e = min(e, now)
        if e < s:
            continue
        span = (s, e + 1)          # inclusive of the end month
        (intern if r.get("is_internship") else work).append(span)
    w, i = union(work), union(intern)
    return {"work": w, "internship": i, "total": union(work + intern)}


# ---------------------------------------------------------------------------
# Evidence check
# ---------------------------------------------------------------------------

def _norm(s):
    s = (s or "").lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[\"“”‘'`•·▪●◦•]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def evidence_found(quote, resume_text):
    q = _norm(quote).strip(" .,;:-")
    if len(q) < 3:
        return False
    hay = _norm(resume_text)
    if q in hay:
        return True
    # tolerate ellipses / small gaps: every chunk of 4+ words must appear in order
    chunks = [c.strip() for c in re.split(r"\.\.\.|…", q) if len(c.strip()) >= 3]
    pos = 0
    for c in chunks:
        idx = hay.find(c, pos)
        if idx == -1:
            return False
        pos = idx + len(c)
    return bool(chunks)


# ---------------------------------------------------------------------------
# Resume evaluation
# ---------------------------------------------------------------------------

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "current_title": {"type": "string"},
        "roles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "organization": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                    "is_internship": {"type": "boolean"},
                },
                "required": ["title", "organization", "start", "end", "is_internship"],
                "additionalProperties": False,
            },
        },
        "education": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"degree": {"type": "string"}, "institution": {"type": "string"}, "year": {"type": "string"}},
                "required": ["degree", "institution", "year"],
                "additionalProperties": False,
            },
        },
        "skills": {"type": "array", "items": {"type": "string"}},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "verdict": {"type": "string", "enum": ["met", "partial", "not_met"]},
                    "evidence": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "verdict", "evidence", "reason"],
                "additionalProperties": False,
            },
        },
        "summary": {"type": "array", "items": {"type": "string"}},
        "interview_questions": {"type": "array", "items": {"type": "string"}},
        "red_flags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["current_title", "roles", "education", "skills", "criteria", "summary",
                 "interview_questions", "red_flags"],
    "additionalProperties": False,
}

EVAL_SYSTEM = """You are a careful, fair technical recruiter. You check ONE resume against a fixed
checklist of requirements for ONE job. Personal identifiers are masked as [NAME], [EMAIL], [PHONE], [LINK].

For every criterion id you are given, return exactly one entry:
- verdict "met": the resume clearly shows it. "partial": some but incomplete evidence (e.g. coursework
  only, less depth or time than asked, a close-but-different tool). "not_met": no evidence.
- evidence: copy the most relevant text EXACTLY as it appears in the resume (one short phrase or
  sentence, max ~25 words). Do not paraphrase, fix spelling or join separate lines. Use "" when not_met.
- reason: one short sentence a recruiter can read.
- Do not credit a skill only because it appears in a list of keywords with nothing showing use of it;
  that is at most "partial".
- For a criterion with a minimum number of years, judge depth from the dated roles; the system checks
  the arithmetic separately.

Also extract:
- roles: every job, internship or freelance engagement with start/end as "YYYY-MM" (or "YYYY" if only the
  year is given; "present" if ongoing). Set is_internship for internships, trainee or apprentice roles.
  Do NOT include education, courses or academic projects as roles.
- education, skills (tools and technologies actually shown), current_title.
- summary: 2-4 sentences explaining the fit for a recruiter, most important first.
- interview_questions: 2-4 questions that probe the weakest or unverified requirements.
- red_flags: factual concerns only (overlapping full-time jobs, unexplained dates in the future,
  inconsistent titles). Never comment on gaps for study, health or family, and never on age, gender,
  religion, caste, marital status, nationality, appearance or other protected attributes.
Never use protected attributes in any judgment."""


def _criteria_block(criteria):
    lines = []
    for c in criteria:
        extra = f"; minimum {c['min_years']:g} years" if c.get("min_years") else ""
        lines.append(f"- id={c['id']} [{c['kind']}, weight {c['weight']}{extra}]: {c['label']}")
    return "\n".join(lines)


def _apply_years(criteria, verdicts, exp):
    """Replace model judgement on minimum-years criteria with computed months."""
    work_years = exp["work"] / 12.0
    all_years = exp["total"] / 12.0
    for c in criteria:
        need = c.get("min_years") or 0
        if not need:
            continue
        v = verdicts.get(c["id"])
        if v is None:
            continue
        model_verdict = v["verdict"]
        if work_years >= need:
            capped = "met"
        elif all_years >= need or work_years >= need * 0.5:
            capped = "partial"
        else:
            capped = "not_met"
        # the model may judge relevance as lower; never raise above the date check
        order = ["not_met", "partial", "met"]
        v["verdict"] = order[min(order.index(model_verdict), order.index(capped))]
        v["computed_years"] = round(work_years, 1)
        if v["verdict"] != model_verdict:
            v["reason"] = (f"{v['reason']} Dates show about {work_years:.1f} years of work"
                           f"{' plus internships' if exp['internship'] else ''} against {need:g} asked.").strip()


def score_evaluation(criteria, verdicts):
    total_w, got = 0.0, 0.0
    for c in criteria:
        v = verdicts.get(c["id"])
        if not v:
            continue
        w = float(c.get("weight") or KIND_WEIGHT_DEFAULT.get(c["kind"], 1))
        total_w += w
        got += w * VERDICT_VALUE.get(v["verdict"], 0.0)
    score = round(100 * got / total_w) if total_w else 0
    knockouts = [c["label"] for c in criteria if c["kind"] == "knockout"
                 and verdicts.get(c["id"], {}).get("verdict") == "not_met"]
    musts = [c for c in criteria if c["kind"] in ("must", "knockout")]
    musts_met = sum(1 for c in musts if verdicts.get(c["id"], {}).get("verdict") == "met")
    return {"score": max(0, min(100, score)), "knockouts_failed": knockouts,
            "must_total": len(musts), "must_met": musts_met}


def _finish(criteria, raw_items, resume_text, exp, engine):
    by_id = {}
    for it in raw_items:
        cid = str(it.get("id", ""))
        if cid in by_id:
            continue
        verdict = it.get("verdict") if it.get("verdict") in VERDICT_VALUE else "not_met"
        evidence = re.sub(r"\s+", " ", str(it.get("evidence", ""))).strip()[:300]
        verified = bool(evidence) and evidence_found(evidence, resume_text)
        reason = str(it.get("reason", "")).strip()[:300]
        if verdict == "met" and not verified:
            verdict = "partial"
            reason = (reason + " (Quote could not be found in the resume, so marked partial.)").strip()
        by_id[cid] = {"verdict": verdict, "evidence": evidence if verified else "", "reason": reason,
                      "verified": verified}
    for c in criteria:
        by_id.setdefault(c["id"], {"verdict": "not_met", "evidence": "", "reason": "Not assessed.", "verified": False})
    _apply_years(criteria, by_id, exp)
    s = score_evaluation(criteria, by_id)
    results = [dict(id=c["id"], label=c["label"], kind=c["kind"], weight=c["weight"],
                    min_years=c.get("min_years") or 0, **by_id[c["id"]]) for c in criteria]
    return dict(s, results=results, experience=exp, engine=engine)


def _find_line(terms, lines, low_lines):
    for t in sorted(terms, key=len, reverse=True):
        pat = re.compile(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])")
        for i, ll in enumerate(low_lines):
            if pat.search(ll):
                return lines[i]
    return None


def rule_evaluate(criteria, resume_text, today=None):
    """Fallback: literal/alias matching with a quoted resume line as evidence.

    A requirement naming several skills ("Python for machine learning") is met
    when every skill is found, partial when some are; alternatives joined by
    "or" need only one."""
    from parsing import ALIAS_TO_CANONICAL, extract_skills, rule_experience_months
    lines = [ln.strip() for ln in resume_text.splitlines() if ln.strip()]
    low_lines = [ln.lower() for ln in lines]

    def variants(skill):
        out = {skill}
        out.update(a for a, cc in ALIAS_TO_CANONICAL.items() if cc == skill)
        return out

    items = []
    for c in criteria:
        label = c["label"]
        alternatives = [a for a in re.split(r"\bor\b|/", label, flags=re.I) if a.strip()]
        best = ("not_met", None)
        for alt in alternatives:
            alt_low = alt.lower()

            def first_pos(sk):
                pos = [alt_low.find(v) for v in variants(sk) if alt_low.find(v) != -1]
                return min(pos) if pos else 10 ** 6
            skills = sorted(extract_skills(alt), key=first_pos)   # in the order the label names them
            if skills:
                hits = [_find_line(variants(sk), lines, low_lines) for sk in skills]
                found = list(dict.fromkeys(h for h in hits if h))
                if found:
                    # quote the line that mentions the most of this requirement's skills
                    def coverage(line):
                        ll = line.lower()
                        return sum(any(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", ll)
                                       for v in variants(sk)) for sk in skills)
                    # most skills covered first; ties go to the skill the label names first
                    quote = max(found, key=lambda ln: (coverage(ln), -found.index(ln)))
                    if len(found) == len(skills):
                        best = ("met", quote)
                        break
                    if best[0] == "not_met":
                        best = ("partial", quote)
            else:
                phrase = re.sub(r"[^a-z0-9+#.\s-]", " ", alt.lower()).strip()
                phrase = re.sub(r"^(at least|minimum|min|experience with|experience in|knowledge of|proficiency in)\s+", "", phrase)
                if len(phrase) >= 2 and not re.search(r"\byears?\b", phrase):
                    hit = _find_line({phrase}, lines, low_lines)
                    if hit:
                        best = ("met", hit)
                        break
        verdict, hit = best
        if c.get("min_years") and verdict == "not_met" and not extract_skills(label):
            items.append({"id": c["id"], "verdict": "met", "evidence": "", "reason": "Checked against job dates."})
        elif hit:
            items.append({"id": c["id"], "verdict": verdict, "evidence": hit[:200],
                          "reason": "Mentioned in the resume." if verdict == "met" else "Only part of this is mentioned."})
        else:
            items.append({"id": c["id"], "verdict": "not_met", "evidence": "", "reason": "No mention found in the resume."})
    exp = rule_experience_months(resume_text, today)
    out = _finish(criteria, items, resume_text, exp, "rule")
    out.update(current_title="", roles=[], education=[], skills=[], summary=[], interview_questions=[], red_flags=[])
    return out


def evaluate_resume(jd_text, criteria, resume_text, name=None, client=None, today=None):
    if not criteria:
        raise ValueError("no criteria")
    if client is None and not llm_available():
        return rule_evaluate(criteria, resume_text, today)
    masked = mask_pii(resume_text, name)
    try:
        user = [
            {"type": "text", "text": f"<job_description>\n{jd_text[:15000]}\n</job_description>\n\n"
                                     f"<checklist>\n{_criteria_block(criteria)}\n</checklist>",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"<resume>\n{masked[:30000]}\n</resume>\n\nEvaluate this resume against every checklist id."},
        ]
        data = call_json(EVAL_SYSTEM, user, EVAL_SCHEMA, max_tokens=5000, client=client)
    except Exception as e:
        log(f"resume evaluation failed, using rule-based: {type(e).__name__}: {e}")
        return rule_evaluate(criteria, resume_text, today)
    roles = [r for r in (data.get("roles") or []) if isinstance(r, dict)][:30]
    exp = experience_months(roles, today)
    out = _finish(criteria, data.get("criteria") or [], masked, exp, "ai")
    # evidence was checked against the masked text the model saw; show it as-is
    out.update(
        current_title=str(data.get("current_title", ""))[:120],
        roles=roles,
        education=[e for e in (data.get("education") or []) if isinstance(e, dict)][:10],
        skills=[str(s)[:60] for s in (data.get("skills") or [])][:40],
        summary=[str(s)[:400] for s in (data.get("summary") or [])][:4],
        interview_questions=[str(s)[:300] for s in (data.get("interview_questions") or [])][:4],
        red_flags=[str(s)[:300] for s in (data.get("red_flags") or [])][:4],
    )
    return out
