"""Work-experience calculation for Screen Genie.

How experience is worked out (the same rules for the rule-based engine and for
roles that Claude extracts):

1. Every date range in the resume is found, in any common format:
   "Jan 2020 - Mar 2022", "January 2020 – Present", "01/2020 - 03/2022",
   "2020-01 to 2022-03", "01/06/2019 - 31/05/2021", "Jan'20 – Mar'22",
   "Jan-2020 to Mar-2022", "2019 - 2021", "2019-21", "since March 2021",
   "2021 onwards", "June 2021 - Till Date", and ranges split over two lines.
2. Each range is placed in the resume section it sits in. Ranges under
   Experience count as work; under Internships as internship; under Education,
   Projects, Certifications, Volunteering, Achievements and similar they are
   not counted. Before any recognisable heading, a line counts unless it reads
   like education (degree, CGPA, semester...), so a teacher at a school or a
   professor at a university is still counted.
3. A role is an internship only when its own title says so (intern, trainee,
   apprentice); a bullet such as "ran the internship programme" does not.
4. Months are counted inclusively ("Jan 2020 - Mar 2020" = 3 months; an ongoing
   role counts up to the current month). A year given without a month is read
   as mid-year, so "2019 - 2021" is about 24 months, not 36, and is marked
   approximate. Future months are never counted.
5. Overlapping roles count once (union of intervals). Internship months are
   kept separate and never added to full-time work.
"""
import re
from datetime import date

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8,
          "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MON = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?")
_Y4 = r"(?:19|20)\d{2}"
# one point in time; longest forms first
_DATE = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?[\s\-/.]+{_MON}[\s,\-/.']*{_Y4}"   # 1st Jan 2020 / 01-Jan-2020
    rf"|{_MON}[\s,\-/.]*\d{{1,2}}(?:st|nd|rd|th)?,\s*{_Y4}"             # Jan 5, 2020
    rf"|{_MON}[\s,\-/.']*{_Y4}"                                         # Jan 2020, Jan-2020, Jan, 2020
    rf"|{_MON}\s*['’]\s*\d{{2}}(?!\d)"                                  # Jan'20
    rf"|{_MON}-\d{{2}}(?![\d/])"                                        # Jan-20
    rf"|\d{{1,2}}[/.\-]\d{{1,2}}[/.\-]{_Y4}"                            # 01/06/2019 (day first)
    rf"|\d{{1,2}}[/.\-]{_Y4}"                                           # 06/2019
    rf"|{_Y4}[/.\-]\d{{1,2}}(?![\d])"                                   # 2019-06
    rf"|{_Y4})"                                                         # 2019
)
_PRESENT = (r"(?:present|current(?:ly)?|now|today|ongoing|continuing|till\s+(?:date|now|present)|"
            r"to\s+date|to\s+present|until\s+(?:now|present)|running|date)")
_SEP = r"\s*(?:-+|–|—|~|\bto\b|\btill\b|\buntil\b|\bthrough\b|\bthru\b|\bupto\b|\bup\s+to\b)\s*"
RANGE_RE = re.compile(rf"(?P<start>{_DATE}){_SEP}(?P<end>{_DATE}|{_PRESENT})(?![a-z])", re.I)
# "2019-21", "2019 – 21"
SHORT_RANGE_RE = re.compile(rf"(?<![\d/.\-])(?P<y1>{_Y4})\s*[-–—]\s*(?P<y2>\d{{2}})(?![\d/.\-])")
# "since March 2021", "2021 onwards", "from Jan 2021 (current)"
OPEN_RANGE_RE = re.compile(
    rf"(?:\bsince\s+(?P<s1>{_DATE})|(?P<s2>{_DATE})\s*(?:onwards?|-\s*$|–\s*$)|"
    rf"\bfrom\s+(?P<s3>{_DATE})\s*(?:\(?\s*(?:{_PRESENT})\s*\)?|onwards?))", re.I)

_INTERN_RE = re.compile(r"\b(intern|interns?hip|summer\s+intern|trainee|apprentice(?:ship)?|articleship)\b", re.I)
_EDU_STRONG_RE = re.compile(
    r"\b(b\.?\s?tech|b\.?\s?e\b|m\.?\s?tech|bca|mca|mba|pgdm|bba|b\.?\s?sc|m\.?\s?sc|b\.?\s?com|m\.?\s?com|b\.?\s?a\b|"
    r"m\.?\s?a\b|ph\.?\s?d|bachelor'?s?|master'?s?|degree|diploma|cgpa|gpa|sgpa|percentage|hsc|ssc|10th|12th|"
    r"class\s+(?:x|xii|10|12)|graduat\w*|semesters?|coursework|course|certification|certificate|"
    r"capstone|thesis|dissertation|hackathon|board\s+of|cbse|icse)\b", re.I)
_INSTITUTION_RE = re.compile(r"\b(university|college|school|institute|academy|iit|nit|iim)\b", re.I)
_PROJECT_RE = re.compile(r"\b(project|projects)\b", re.I)
_JOB_WORD_RE = re.compile(
    r"\b(engineer|developer|analyst|manager|executive|associate|consultant|lead|officer|specialist|designer|"
    r"scientist|support|administrator|assistant|teacher|professor|lecturer|faculty|tutor|trainer|instructor|"
    r"counsel+or|coordinator|principal|head|director|representative|advisor|adviser|mentor|recruiter|"
    r"accountant|auditor|architect|owner|founder|co-founder|partner|president|vp|intern|trainee|apprentice|"
    r"staff|employee|worked|working|employed|joined|freelance|freelancer|contractor|researcher|"
    r"fellow|operator|technician|agent|clerk|cashier|supervisor|editor|writer|marketer|sales)\b", re.I)

# section headings -> what ranges under them count as (checked in this order)
_SECTIONS = [
    ("internship", r"(?:summer\s+|industrial\s+)?internships?(?:\s+(?:experience|details))?|industrial training|"
                   r"summer training|articleship"),
    ("excluded", r"(?:academic|personal|key|major|minor|college|university|selected|notable|side|capstone)?\s*"
                 r"projects?(?:\s+(?:undertaken|done|details))?|certifications?|certificates?|courses?|trainings?|"
                 r"licen[cs]es|workshops?|volunteer(?:ing)?(?:\s+(?:experience|work))?|social work|"
                 r"extra[- ]?curricular(?:\s+activities)?|co[- ]?curricular(?:\s+activities)?|activities|"
                 r"achievements?|awards?|honou?rs?|publications?|positions? of responsibility|"
                 r"leadership(?:\s+roles?)?|hobbies|interests|languages?|references?|declaration|"
                 r"personal\s+(?:details|information|profile)"),
    ("education", r"education(?:al)?(?:\s+(?:background|qualifications?|details))?|academics?|academic\s+"
                  r"(?:background|qualifications?|details|record|profile|credentials)|qualifications?|scholastic.*"),
    ("work", r"(?:[a-z']+\s+){0,3}(?:experience|exp\.?)(?:\s+(?:details|summary|record|profile))?|"
             r"employment(?:\s+(?:history|details|record))?|work(?:\s+(?:history|details|profile))?|"
             r"career(?:\s+(?:history|profile|progression|details|journey))?|positions? held|"
             r"professional\s+(?:background|journey|history)|previous employers|where i'?ve worked|"
             r"employers?"),
    ("neutral", r"(?:professional\s+|career\s+)?(?:summary|profile|objective)|about me|skills?|"
                r"technical skills|key skills|core competencies|tools|strengths"),
]
_SECTION_RES = [(k, re.compile(rf"^(?:{p})$", re.I)) for k, p in _SECTIONS]


def _ym(y, m):
    return y * 12 + (m - 1)


def format_duration(months, long=False):
    """12 -> "1 yr", 113 -> "9 yrs 5 mos", 3 -> "3 mos", 0 -> "0 mos".
    long=True gives "9 years 5 months" for sentences."""
    if months is None:
        return "—"
    months = max(0, int(round(months)))
    y, m = divmod(months, 12)
    if long:
        yt = f"{y} year{'s' if y != 1 else ''}"
        mt = f"{m} month{'s' if m != 1 else ''}"
    else:
        yt = f"{y} yr{'s' if y != 1 else ''}"
        mt = f"{m} mo{'s' if m != 1 else ''}"
    if y and m:
        return f"{yt} {mt}"
    return yt if y else mt


def format_years(years, long=True):
    """A requirement like 3 or 2.5 years -> "3 years" / "2 years 6 months"."""
    return format_duration(round(float(years) * 12), long=long)


def ym_label(v):
    return f"{list(MONTHS)[v % 12].title()} {v // 12}"


def _parse_point(text, today, is_end):
    """One date -> (month index, precise). Year-only -> mid-year, marked not precise."""
    s = text.strip().lower().rstrip(".,")
    now = _ym(today.year, today.month)
    if re.fullmatch(_PRESENT, s, re.I):
        return now, True
    mon = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s)
    y4 = re.search(_Y4, s)
    if mon:
        if y4:
            y = int(y4.group(0))
        else:
            yy = re.search(r"(\d{2})\s*$", s)
            if not yy:
                return None
            y = 2000 + int(yy.group(1)) if int(yy.group(1)) <= today.year % 100 + 1 else 1900 + int(yy.group(1))
        return _ym(y, MONTHS[mon.group(1)]), True
    m = re.fullmatch(rf"(\d{{1,2}})[/.\-](\d{{1,2}})[/.\-]({_Y4})", s)
    if m:                                    # day/month/year (Indian and UK order)
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 and d <= 12:              # tolerate month/day/year
            d, mo = mo, d
        return (_ym(y, mo), True) if 1 <= mo <= 12 else None
    m = re.fullmatch(rf"(\d{{1,2}})[/.\-]({_Y4})", s)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        return (_ym(y, mo), True) if 1 <= mo <= 12 else None
    m = re.fullmatch(rf"({_Y4})[/.\-](\d{{1,2}})", s)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return (_ym(y, mo), True) if 1 <= mo <= 12 else None
    if y4 and re.fullmatch(_Y4, s):
        y = int(y4.group(0))
        return _ym(y, 6 if is_end else 7), False     # year only: mid-year
    return None


def parse_month(value, today=None, is_end=False):
    """Compatibility helper: one date string -> month index, or None."""
    today = today or date.today()
    if value is None or not str(value).strip():
        return None
    p = _parse_point(str(value), today, is_end)
    if p is None:
        # "2024-03-15" style or free text with a date in it
        m = re.search(_DATE, str(value), re.I)
        p = _parse_point(m.group(0), today, is_end) if m else None
    return p[0] if p else None


def _valid_year(v, today):
    return _ym(1960, 1) <= v <= _ym(today.year + 1, 12)


def find_ranges(line, today):
    """All date ranges in one line: list of dicts with start, end, precise, text."""
    out, taken = [], []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for m in RANGE_RE.finditer(line):
        s = _parse_point(m.group("start"), today, False)
        e = _parse_point(m.group("end"), today, True)
        if not s or not e or not free(*m.span()):
            continue
        taken.append(m.span())
        out.append({"start": s[0], "end": e[0], "precise": s[1] and e[1], "text": m.group(0).strip()})
    for m in SHORT_RANGE_RE.finditer(line):
        if not free(*m.span()):
            continue
        y1 = int(m.group("y1"))
        y2 = (y1 // 100) * 100 + int(m.group("y2"))
        if y2 < y1:
            y2 += 100
        if y2 - y1 > 15:
            continue
        taken.append(m.span())
        out.append({"start": _ym(y1, 7), "end": _ym(y2, 6), "precise": False, "text": m.group(0).strip()})
    for m in OPEN_RANGE_RE.finditer(line):
        if not free(*m.span()):
            continue
        st = m.group("s1") or m.group("s2") or m.group("s3")
        s = _parse_point(st, today, False)
        if not s:
            continue
        # "Jan 2020 -" at the end of a line is only open-ended if the caller found no end on the next line
        dangling = bool(m.group("s2")) and not re.search(r"onwards?", m.group(0), re.I)
        taken.append(m.span())
        out.append({"start": s[0], "end": _ym(today.year, today.month), "precise": s[1],
                    "text": m.group(0).strip(), "dangling": dangling})
    return [r for r in out if _valid_year(r["start"], today)]


def _heading_kind(t):
    t = re.sub(r"\s*&\s*|\s+and\s+", " & ", t.lower())
    t = re.sub(r"\s*\([^)]*\)\s*", " ", t)              # "Professional Experience (5 years)"
    t = re.sub(r"[^a-z&/,' ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    for kind, rx in _SECTION_RES:
        if rx.match(t):
            return kind
    first = re.split(r" & |/|,", t)[0].strip()             # "Work Experience & Internships"
    if first != t:
        for kind, rx in _SECTION_RES:
            if rx.match(first):
                return kind
    return None


def _clean_heading(line):
    t = re.sub(r"[\s:|•·\-–—_=*#>]+$", "", re.sub(r"^[\s:|•·\-–—_=*#>\d.]+", "", line)).strip()
    return t.title() if t.isupper() else t


def _heading(line):
    """Section kind if this whole line is a heading, else None."""
    t = re.sub(r"[\s:|•·\-–—_=*#>]+$", "", re.sub(r"^[\s:|•·\-–—_=*#>\d.]+", "", line)).strip()
    if not t or len(t) > 50 or len(re.sub(r"\([^)]*\)", "", t).split()) > 5 or re.search(r"\d{4}", t):
        return None
    return _heading_kind(t)


def _inline_heading(line):
    """'Experience: Analyst at X, Jan 2020 - Present' -> ('work', rest)."""
    m = re.match(r"^\s*([A-Za-z][A-Za-z '&/]{2,40}?)\s*[:–—|]\s+(.+)$", line)
    if m:
        kind = _heading_kind(m.group(1))
        if kind:
            return kind, m.group(2)
    return None, line


def _looks_like_heading(line):
    """Short line in capitals or Title Case ending without a full stop, e.g. "WHERE I'VE WORKED"."""
    t = line.strip().rstrip(":").strip()
    if not t or len(t) > 40 or len(t.split()) > 5 or _is_bullet(t) or t.endswith(".") or re.search(r"\d", t):
        return False
    letters = re.sub(r"[^A-Za-z]", "", t)
    return len(letters) >= 4 and (t.isupper() or (line.strip().endswith(":")))


def _is_bullet(line):
    return bool(re.match(r"^\s*(?:[-•*▪●◦➢➤✓►]|\d+[.)])\s+", line))


def extract_roles(resume_text, today=None):
    """Find dated roles in raw resume text, each tagged work / internship / excluded with a reason."""
    today = today or date.today()
    lines = [ln.rstrip() for ln in (resume_text or "").splitlines()]
    roles = []
    section, section_name = None, ""
    i = 0
    while i < len(lines):
        if "\f" in lines[i]:
            section, section_name = None, ""        # a new column of a two-column page
            i += 1
            continue
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        h = _heading(line)
        if h:
            section = None if h == "neutral" else h
            section_name = _clean_heading(line)
            i += 1
            continue
        if not find_ranges(line, today) and _looks_like_heading(line) and section in ("education", "excluded"):
            # an unrecognised heading after Education/Projects ("Where I've worked"): stop
            # treating what follows as education, and judge each line on its own words
            section = None
            i += 1
            continue
        ih, rest = _inline_heading(line)
        if ih:
            section = None if ih == "neutral" else ih
            section_name = _clean_heading(line.split(":")[0])
            line = rest
        ranges = find_ranges(line, today)
        joined = False
        # a range split over two lines: "Jan 2020 -" / "Present", or "From Jan 2020" / "To Mar 2022"
        if i + 1 < len(lines) and (not ranges or any(r.get("dangling") for r in ranges)):
            nxt = lines[i + 1].strip()
            two = find_ranges(line + " " + nxt, today)
            two = [r for r in two if not r.get("dangling")]
            if two and len(two) > len([r for r in ranges if not r.get("dangling")]):
                if not find_ranges(nxt, today) or re.match(rf"^\s*(?:{_SEP.strip()})?\s*(?:{_PRESENT}|{_DATE})\s*$",
                                                           nxt, re.I):
                    ranges, joined = two, True
        if not ranges:
            i += 1
            continue
        # the role's own words: this line, plus the title line above or below when it is not a bullet
        ctx = [line]
        prev = next((lines[j].strip() for j in range(i - 1, max(-1, i - 3), -1) if lines[j].strip()), "")
        nxt_i = i + (2 if joined else 1)
        nxt = lines[nxt_i].strip() if nxt_i < len(lines) else ""
        bare = len(re.sub(r"[^A-Za-z]", "", RANGE_RE.sub("", line))) < 12
        if prev and not _is_bullet(prev) and not find_ranges(prev, today) and not _heading(prev):
            ctx.append(prev)
        if bare and nxt and not _is_bullet(nxt) and not find_ranges(nxt, today) and not _heading(nxt):
            ctx.append(nxt)
        context = " | ".join(ctx)
        for r in ranges:
            r = dict(r)
            r.pop("dangling", None)
            kind, reason = "work", ""
            if section == "education" and not (_JOB_WORD_RE.search(context) and not _EDU_STRONG_RE.search(context)
                                                   and not _INSTITUTION_RE.search(context)):
                kind, reason = "excluded", f"Under “{section_name or 'Education'}”"
            elif section == "excluded":
                kind, reason = "excluded", f"Under “{section_name or 'Projects'}”"
            elif section == "internship" or _INTERN_RE.search(context):
                kind = "internship"
            elif section is None:
                if _EDU_STRONG_RE.search(context) and not _JOB_WORD_RE.search(context):
                    kind, reason = "excluded", "Looks like education"
                elif _PROJECT_RE.search(context) and not _JOB_WORD_RE.search(context):
                    kind, reason = "excluded", "Looks like a project"
                elif _INSTITUTION_RE.search(context) and not _JOB_WORD_RE.search(context):
                    kind, reason = "excluded", "Looks like education"
            r.update(kind=kind, reason=reason, title=_title_from(context, r["text"]))
            roles.append(r)
        i += 2 if joined else 1
    return roles


def _title_from(context, date_text):
    t = context.split(" | ")
    first = t[0].replace(date_text, " ")
    if len(re.sub(r"[^A-Za-z]", "", first)) < 3 and len(t) > 1:
        first = t[1]
    first = re.sub(r"\s+", " ", re.sub(r"[|•·()\[\]]+", " ", first)).strip(" ,-–—:")
    return first[:90]


def _union(intervals):
    """Total months covered by inclusive [s, e] month intervals, overlaps counted once."""
    total, cs, ce = 0, None, None
    for s, e in sorted(intervals):
        if ce is None or s > ce + 1:
            if ce is not None:
                total += ce - cs + 1
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if ce is not None:
        total += ce - cs + 1
    return total


def compute(roles, today=None):
    """roles: dicts with start/end month indexes (from extract_roles) -> totals and a breakdown."""
    today = today or date.today()
    now = _ym(today.year, today.month)
    work, intern, rows = [], [], []
    for r in roles:
        s, e = r["start"], min(r["end"], now)
        row = dict(r, months=0)
        if e < s and not r.get("precise", True) and r["end"] // 12 == s // 12:
            e = s + 5                       # "2015 - 2015": a role inside one year, read as about 6 months
        if s > now:
            row.update(kind="excluded", reason="Starts in the future")
        elif e < s:
            row.update(kind="excluded", reason="End date is before start date")
        else:
            row["months"] = e - s + 1
            row["end"] = e
            if r["kind"] == "work":
                work.append((s, e))
            elif r["kind"] == "internship":
                intern.append((s, e))
        rows.append(row)
    w, it = _union(work), _union(intern)
    raw_work = sum(r["months"] for r in rows if r["kind"] == "work")
    for r in rows:
        r["from"] = ym_label(r["start"])
        r["to"] = "Present" if r["end"] >= now else ym_label(r["end"])
    return {"work": w, "internship": it, "total": _union(work + intern),
            "overlap": max(0, raw_work - w),
            "approximate": any(not r["precise"] for r in rows if r["kind"] != "excluded"),
            "roles": rows}


_STATED_RE = re.compile(
    r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs|year)\s+(?:of\s+)?(?:[a-z/&\-]+\s+){0,3}?(?:experience|exp)\b", re.I)


def stated_years(resume_text):
    vals = [float(m.group(1)) for m in _STATED_RE.finditer(resume_text or "") if float(m.group(1)) <= 45]
    return max(vals) if vals else None


def from_text(resume_text, today=None):
    """Full calculation from raw resume text, with what the resume itself claims."""
    today = today or date.today()
    out = compute(extract_roles(resume_text, today), today)
    out["stated_years"] = stated_years(resume_text)
    return out


def from_role_dicts(roles, today=None):
    """Roles given as {"start": "2019-03", "end": "present", "is_internship": bool} -> totals."""
    today = today or date.today()
    out = []
    for r in roles or []:
        s = _parse_point(str(r.get("start") or ""), today, False) if r.get("start") else None
        if not s:
            continue
        e = _parse_point(str(r.get("end") or ""), today, True) if r.get("end") else None
        e = e or s
        out.append({"start": s[0], "end": e[0], "precise": s[1] and e[1], "text": f"{r.get('start')} – {r.get('end')}",
                    "kind": "internship" if r.get("is_internship") else "work", "reason": "",
                    "title": str(r.get("title") or "")[:90]})
    return compute(out, today)


def from_model_roles(model_roles, resume_text, today=None):
    """Roles that Claude extracted -> same calculation. A role is kept only if its dates
    can be found in the resume, so an invented or misread date never counts."""
    today = today or date.today()
    text_low = (resume_text or "").lower()
    years_in_text = set(re.findall(_Y4, resume_text or ""))
    present_in_text = bool(re.search(_PRESENT, text_low))
    roles = []
    for mr in model_roles or []:
        s = _parse_point(str(mr.get("start", "")), today, False) if mr.get("start") else None
        end_raw = str(mr.get("end", "") or "")
        e = _parse_point(end_raw, today, True) if end_raw else None
        title = " ".join(x for x in [str(mr.get("title", "")).strip(), str(mr.get("organization", "")).strip()] if x)
        if not s:
            continue
        if e is None:
            e = s
        reason = ""
        kind = "internship" if mr.get("is_internship") else "work"
        s_year = str(s[0] // 12)
        if s_year not in years_in_text:
            kind, reason = "excluded", f"Start year {s_year} not found in the resume"
        elif end_raw.strip().lower() not in ("present", "current", "now", "ongoing") \
                and str(e[0] // 12) not in years_in_text:
            kind, reason = "excluded", f"End year {e[0] // 12} not found in the resume"
        elif end_raw.strip().lower() in ("present", "current", "now", "ongoing") and not present_in_text:
            e = s
            reason = "Resume does not say this role is ongoing; counted as one month"
        roles.append({"start": s[0], "end": e[0], "precise": s[1] and e[1], "text": f"{mr.get('start')} – {end_raw}",
                      "kind": kind, "reason": reason, "title": title[:90]})
    out = compute(roles, today)
    out["stated_years"] = stated_years(resume_text)
    return out


def summary_note(exp):
    """One plain sentence for recruiters when something about the dates needs a look."""
    notes = []
    years = exp["work"] / 12.0
    st = exp.get("stated_years")
    if st is not None and exp["work"] and abs(st - years) >= 1.5:
        notes.append(f"Resume states {st:g}+ years; dated roles add up to {format_duration(exp['work'], long=True)}.")
    if st is not None and not exp["work"] and not exp["internship"]:
        notes.append(f"Resume states {st:g}+ years but gives no dated roles to check it.")
    return " ".join(notes)
