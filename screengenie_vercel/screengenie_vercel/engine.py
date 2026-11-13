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

import experience

ENGINE_VERSION = "sg-engine-2.0"
MODEL = os.environ.get("SG_MODEL", "claude-sonnet-5-5")
EFFORT = os.environ.get("SG_EFFORT", "low")          # low | medium | high
CALL_TIMEOUT = float(os.environ.get("SG_TIMEOUT", "55"))

# USD per million tokens (Claude Sonnet 5.5 list price); override with env vars.
PRICE_IN = float(os.environ.get("SG_PRICE_IN", "2.0"))
PRICE_OUT = float(os.environ.get("SG_PRICE_OUT", "10.0"))
PRICE_CACHE_READ = PRICE_IN * 0.1
PRICE_CACHE_WRITE = PRICE_IN * 1.25

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


def new_meta(kind):
    """Run record for the engine-health log: path, model, tokens, time, cost."""
    return {"kind": kind, "path": "rule", "model": None, "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "seconds": 0.0, "cost_usd": 0.0,
            "attempts": 0, "error": None}


def _record_usage(meta, resp, started):
    if meta is None:
        return
    usage = getattr(resp, "usage", None)
    get = lambda k: int(getattr(usage, k, 0) or 0) if usage is not None else 0
    meta.update(
        model=getattr(resp, "model", None) or MODEL,
        input_tokens=meta["input_tokens"] + get("input_tokens"),
        output_tokens=meta["output_tokens"] + get("output_tokens"),
        cache_read_tokens=meta["cache_read_tokens"] + get("cache_read_input_tokens"),
        cache_write_tokens=meta["cache_write_tokens"] + get("cache_creation_input_tokens"),
    )
    meta["seconds"] = round(time.time() - started, 2)
    meta["cost_usd"] = round((meta["input_tokens"] * PRICE_IN + meta["output_tokens"] * PRICE_OUT
                              + meta["cache_read_tokens"] * PRICE_CACHE_READ
                              + meta["cache_write_tokens"] * PRICE_CACHE_WRITE) / 1e6, 6)


def call_json(system, user_content, schema, max_tokens=4000, client=None, meta=None):
    """One structured-output call. Returns the parsed dict, or raises.

    ``output_config`` is sent through ``extra_body`` so this works on any
    version of the anthropic SDK. When ``meta`` (from ``new_meta``) is given,
    tokens, time and cost are added to it."""
    import anthropic
    client = client or _get_client()
    started = time.time()
    body = {"output_config": {"format": {"type": "json_schema", "schema": schema}, "effort": EFFORT}}
    delays = [2, 5, 10]
    for attempt in range(len(delays) + 1):
        try:
            if meta is not None:
                meta["attempts"] += 1
            with _semaphore:
                resp = client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user_content}],
                    extra_body=body,
                )
            _record_usage(meta, resp, started)
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


_SKILL_DISPLAY = {
    "sql": "SQL", "aws": "AWS", "gcp": "GCP", "etl": "ETL", "seo": "SEO", "sem": "SEM", "crm": "CRM",
    "gst": "GST", "tds": "TDS", "sap": "SAP", "nlp": "NLP", "css": "CSS", "html": "HTML", "php": "PHP",
    "r": "R", "c++": "C++", "c#": "C#", "ci/cd": "CI/CD", "power bi": "Power BI", "node.js": "Node.js",
    "next.js": "Next.js", "scikit-learn": "scikit-learn", "pytorch": "PyTorch", "tensorflow": "TensorFlow",
    "mysql": "MySQL", "postgresql": "PostgreSQL", "mongodb": "MongoDB", "javascript": "JavaScript",
    "typescript": "TypeScript", "graphql": "GraphQL", "rest api": "REST API", "nosql": "NoSQL",
    "a/b testing": "A/B testing", "ms office": "MS Office", "powerpoint": "PowerPoint", "word": "MS Word",
    "jira": "Jira", "testng": "TestNG", "junit": "JUnit", "quickbooks": "QuickBooks", "hr policies": "HR policies",
}
# generic soft skills never become checklist items on their own
_SOFT_SKILLS = {"communication", "teamwork", "leadership", "problem solving", "time management"}
_NICE_RE = re.compile(r"nice[\s-]to[\s-]have|good[\s-]to[\s-]have|\bpreferred\b|\bpreferably\b|\bbonus\b|"
                      r"\ba plus\b|\bplus point\b|\bdesirable\b|added advantage|\bwould be great\b", re.I)
_MUST_RE = re.compile(r"must[\s-]have|\brequired\b|\brequirements?\b|\bmandatory\b|\bessential\b|"
                      r"\bqualifications?\b|\bshould have\b|\bwhat you need\b", re.I)


def skill_display(canonical):
    return _SKILL_DISPLAY.get(canonical, canonical[:1].upper() + canonical[1:])


def rule_requisition(jd_text):
    """Fallback checklist: skills in the JD, grouped into "X or Y" where the JD offers a choice,
    marked nice-to-have when they sit under a "nice to have / preferred" heading or sentence."""
    from parsing import ALIAS_TO_CANONICAL, extract_skills, required_years

    def ordered_skills(text):
        low = " " + text.lower() + " "
        def pos(sk):
            ps = [m.start() for a, cc in ALIAS_TO_CANONICAL.items() if cc == sk
                  for m in [re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", low)] if m]
            return min(ps) if ps else 10 ** 6
        return sorted((s for s in extract_skills(text) if s not in _SOFT_SKILLS), key=pos)

    items, seen = [], set()
    mode = "must"
    for raw in (jd_text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        nice_hit, must_hit = _NICE_RE.search(line), _MUST_RE.search(line)
        if nice_hit and (not must_hit or nice_hit.start() > must_hit.start()):
            line_mode = "nice"
        elif must_hit:
            line_mode = "must"
        else:
            line_mode = mode
        # a heading such as "Nice to have:" with nothing after it sets the mode for the lines below
        if re.fullmatch(r"[^a-z0-9]*[\w\s/&()'-]{0,40}:?\s*", line, flags=re.I) and (nice_hit or must_hit):
            mode = line_mode
            continue
        if nice_hit or must_hit:
            mode = line_mode
        for clause in re.split(r"[;.\n]|,(?![^()]*\))", line):
            sk = ordered_skills(clause)
            if not sk:
                continue
            # skills joined by "or" / "either" in one clause are alternatives
            if len(sk) > 1 and re.search(r"\bor\b|\beither\b|\bany of\b", clause, re.I):
                groups = [sk]
            else:
                groups = [[x] for x in sk]
            for g in groups:
                key = tuple(sorted(g))
                if key in seen or any(x in {y for k in seen for y in k} for x in g) and len(g) == 1:
                    continue
                seen.add(key)
                items.append({"label": " or ".join(skill_display(x) for x in g), "kind": line_mode})

    musts = [i for i in items if i["kind"] == "must"][:7]
    nices = [i for i in items if i["kind"] == "nice"][:5]
    crit = [dict(i, weight=2, min_years=0) for i in musts] + [dict(i, weight=1, min_years=0) for i in nices]
    years = required_years(jd_text)
    if years:
        crit.insert(0, {"label": f"At least {years} years of relevant experience", "kind": "must",
                        "weight": 3, "min_years": years})
    return {"role_title": _guess_title(jd_text), "seniority": "Unknown",
            "criteria": _clean_criteria(crit), "engine": "rule"}


def _guess_title(jd_text):
    for raw in (jd_text or "").splitlines()[:5]:
        line = re.sub(r"^(job title|role|position)\s*[:\-]\s*", "", raw.strip(), flags=re.I)
        if 2 <= len(line) <= 80 and not line.endswith(".") and len(line.split()) <= 10:
            return line
    return ""


def draft_requisition(jd_text, context="", client=None):
    meta = new_meta("requisition")
    if client is None and not llm_available():
        meta["error"] = "ANTHROPIC_API_KEY not set or anthropic package missing"
        return dict(rule_requisition(jd_text), meta=meta)
    started = time.time()
    try:
        user = (f"Hiring context from the recruiter (may be empty): {context.strip()[:1500]}\n\n"
                f"<job_description>\n{jd_text[:20000]}\n</job_description>\n\n"
                "Return the screening checklist.")
        data = call_json(REQUISITION_SYSTEM, user, REQUISITION_SCHEMA, max_tokens=2000, client=client, meta=meta)
        crit = _clean_criteria(data.get("criteria"))
        if not crit:
            raise RuntimeError("no criteria returned")
        meta["path"] = "ai"
        return {"role_title": str(data.get("role_title", ""))[:120],
                "seniority": data.get("seniority", "Unknown"), "criteria": crit, "engine": "ai", "meta": meta}
    except Exception as e:
        log(f"requisition draft failed, using rule-based: {type(e).__name__}: {e}")
        meta.update(error=f"{type(e).__name__}: {e}"[:500], seconds=round(time.time() - started, 2))
        return dict(rule_requisition(jd_text), meta=meta)


# ---------------------------------------------------------------------------
# PII masking
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\-\s().]{8,}\d)(?!\d)")
_URL_RE = re.compile(r"https?://\S+|www\.\S+|linkedin\.com/\S+|github\.com/\S+", re.I)


def mask_pii(text, name=None):
    masked = _EMAIL_RE.sub("[EMAIL]", text)
    masked = _URL_RE.sub("[LINK]", masked)
    def _mask_phone(m):
        cand = m.group(0)
        n = sum(ch.isdigit() for ch in cand)
        if n < 10 or re.fullmatch(r"(?:\(?(?:19|20)\d{2}\)?[\s\-–—.()to]*){2,}", cand.strip(), flags=re.I):
            return cand            # too short for a phone, or a run of years: keep dates for the model
        return "[PHONE]"
    masked = _PHONE_RE.sub(_mask_phone, masked)
    if name and name.strip() and name.strip().lower() not in ("not found", "unknown"):
        for part in sorted({p for p in re.split(r"\s+", name.strip()) if len(p) >= 3}, key=len, reverse=True):
            masked = re.sub(r"(?<![A-Za-z])" + re.escape(part) + r"(?![A-Za-z])", "[NAME]", masked, flags=re.I)
    return masked


# ---------------------------------------------------------------------------
# Experience maths
# ---------------------------------------------------------------------------

def parse_month(value, today=None, is_end=False):
    """One date string -> month index (see experience.py for the rules)."""
    return experience.parse_month(value, today, is_end)


def experience_months(roles, today=None):
    """Union of employment intervals in months; overlaps count once, internships kept apart.
    Returns {"work", "internship", "total", "overlap", "approximate", "roles"}."""
    return experience.from_role_dicts(roles, today)


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
        if v["reason"] == "Checked against job dates." or not v["reason"]:
            v["reason"] = (f"Dates show about {work_years:.1f} years of full-time work"
                           f"{' plus internships' if exp['internship'] else ''} against {need:g} asked.")
        elif v["verdict"] != model_verdict:
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


def _is_years_only(c):
    """A requirement that is only an amount of experience ("3+ years"), judged from job dates."""
    if not c.get("min_years"):
        return False
    from parsing import extract_skills
    return not extract_skills(c["label"])


def _finish(criteria, raw_items, resume_text, exp, engine):
    years_only = {c["id"] for c in criteria if _is_years_only(c)}
    by_id = {}
    for it in raw_items:
        cid = str(it.get("id", ""))
        if cid in by_id:
            continue
        verdict = it.get("verdict") if it.get("verdict") in VERDICT_VALUE else "not_met"
        evidence = re.sub(r"\s+", " ", str(it.get("evidence", ""))).strip()[:300]
        verified = bool(evidence) and evidence_found(evidence, resume_text)
        reason = str(it.get("reason", "")).strip()[:300]
        if verdict == "met" and not verified and cid not in years_only:
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
        # split "X or Y" / "X/Y" into alternatives, but keep names like "A/B testing" or "CI/CD" whole
        slash_names = sorted((a for a in ALIAS_TO_CANONICAL if "/" in a), key=len, reverse=True)
        protected = label
        for k, a in enumerate(slash_names):
            protected = re.sub(re.escape(a), f"\x00{k}\x00", protected, flags=re.I)
        parts = [a for a in re.split(r"\bor\b|/", protected, flags=re.I) if a.strip()]
        alternatives = [re.sub(r"\x00(\d+)\x00", lambda m: slash_names[int(m.group(1))], a) for a in parts]
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
    meta = new_meta("resume")
    if client is None and not llm_available():
        meta["error"] = "ANTHROPIC_API_KEY not set or anthropic package missing"
        return dict(rule_evaluate(criteria, resume_text, today), meta=meta)
    masked = mask_pii(resume_text, name)
    started = time.time()
    try:
        user = [
            {"type": "text", "text": f"<job_description>\n{jd_text[:15000]}\n</job_description>\n\n"
                                     f"<checklist>\n{_criteria_block(criteria)}\n</checklist>",
             "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": f"<resume>\n{masked[:30000]}\n</resume>\n\nEvaluate this resume against every checklist id."},
        ]
        data = call_json(EVAL_SYSTEM, user, EVAL_SCHEMA, max_tokens=5000, client=client, meta=meta)
    except Exception as e:
        log(f"resume evaluation failed, using rule-based: {type(e).__name__}: {e}")
        meta.update(error=f"{type(e).__name__}: {e}"[:500], seconds=round(time.time() - started, 2))
        return dict(rule_evaluate(criteria, resume_text, today), meta=meta)
    meta["path"] = "ai"
    roles = [r for r in (data.get("roles") or []) if isinstance(r, dict)][:30]
    # Claude reads the roles; the months are counted here, and only from dates that are in the resume
    exp = experience.from_model_roles(roles, resume_text, today)
    scan = experience.from_text(resume_text, today)
    if not exp["work"] and not exp["internship"] and (scan["work"] or scan["internship"]):
        exp = scan                                  # Claude returned no usable roles: use the date scan
    elif abs(exp["work"] - scan["work"]) >= 12 and scan["roles"]:
        exp["cross_check"] = (f"A second reading of the dates gives {scan['work'] / 12:.1f} years; "
                              f"check the dates in this resume.")
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
        meta=meta,
    )
    return out
