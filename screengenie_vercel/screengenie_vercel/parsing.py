"""
Screen Genie — resume screening engine.

Designed to give recruiter-grade accuracy without any external API call.
Combines:
  - large, normalized skill / tool / qualification vocabulary (with aliases)
  - phrase-aware skill extraction (multi-word terms)
  - aggressive noise / generic-word filtering for JD keywords
  - TF-IDF + ngrams semantic similarity
  - experience-years extraction with multiple patterns (date ranges + "X years")
  - education detection with degree normalization
  - explicit Passed / Warnings / Issues counters
  - actionable AI Suggestions (improvement hints)
"""

import re
import os
import sys
import time
import json
import threading
import zipfile
import io
import pdfplumber
import docx
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------
# Noise / stopwords
# ----------------------------------------------------------------------
#
# These words are filtered from JD-keyword extraction. They are either
# generic English, common HR/JD boilerplate, or words that pollute the
# top-keywords list (company names, locations, soft phrases). This is
# what was making earlier scoring inaccurate.

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been
before being below between both but by can't cannot could couldn't did didn't do does
doesn't doing don't down during each few for from further had hadn't has hasn't have
haven't having he he'd he'll he's her here here's hers herself him himself his how
how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out
over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll
they're they've this those through to too under until up very was wasn't we we'd
we'll we're we've were weren't what what's when when's where where's which while who
who's whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves

work experience years year strong excellent ability skills team teams job role roles
responsibilities requirements requirement preferred required candidate candidates
company companies please looking must building good great new etc using use used
based within across also will may can shall should one two three four five six seven
eight nine ten plus min max etc title position open opening hire hiring location
locations onsite remote hybrid fresher internship intern junior senior lead
description detail details summary objective profile overview
""".split())

# Words that are JD boilerplate / role text but not informative as keywords.
NOISE_TERMS = set("""
job title experience required preferred responsibilities key skills knowledge
location company about role roles responsibilities requirement qualifications
qualification overview description type fulltime full part time benefits salary
employment status industry domain dept department team member members manager
managers candidate candidates apply applicant applicants opportunity opportunities
required must mandatory should know preferably ideal ideally minimum mininum maximum
preferred preferred. plus etc. e.g. eg ie i.e. years yrs year months day days
fresher freshers experience pune mumbai delhi bangalore bengaluru hyderabad chennai
gurgaon noida onsite offline remote hybrid india office hours time mode work-from
multiversity nuverse intern internship trainee fresher fulltime full-time part-time
contract permanent
""".split())

ALL_NOISE = STOPWORDS | NOISE_TERMS


# ----------------------------------------------------------------------
# Skill vocabulary with aliases
# ----------------------------------------------------------------------
#
# Each canonical skill maps to a list of aliases. Matching uses the union
# of aliases so a resume that says "ML" still matches "Machine Learning".

SKILL_ALIASES = {
    # ---- Programming languages ----
    "python": ["python", "python3", "py", "python 3"],
    "java": ["java", "core java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "c sharp", "csharp", "dotnet", ".net"],
    "go": ["golang", "go lang"],
    "rust": ["rust"],
    "php": ["php"],
    "ruby": ["ruby", "ruby on rails", "rails"],
    "kotlin": ["kotlin"],
    "swift": ["swift"],
    "r": ["r programming", "r language"],
    "scala": ["scala"],
    "sql": ["sql", "t-sql", "pl/sql", "tsql"],

    # ---- Web ----
    "html": ["html", "html5"],
    "css": ["css", "css3", "sass", "scss"],
    "react": ["react", "reactjs", "react.js", "react native"],
    "angular": ["angular", "angularjs"],
    "vue": ["vue", "vuejs", "vue.js"],
    "node.js": ["node", "nodejs", "node.js"],
    "next.js": ["next", "nextjs", "next.js"],
    "django": ["django"],
    "flask": ["flask"],
    "spring": ["spring", "spring boot", "springboot"],
    "express": ["express", "expressjs", "express.js"],
    "rest api": ["rest api", "restful", "rest", "restful api"],
    "graphql": ["graphql"],

    # ---- Data / ML / AI ----
    "machine learning": ["machine learning", "ml", "supervised learning", "unsupervised learning"],
    "deep learning": ["deep learning", "dl", "neural network", "neural networks"],
    "nlp": ["nlp", "natural language processing"],
    "computer vision": ["computer vision", "cv"],
    "data analysis": ["data analysis", "data analytics", "analytics"],
    "data science": ["data science", "data scientist"],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "scikit-learn": ["scikit", "scikit-learn", "sklearn"],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "etl": ["etl", "data pipeline", "data pipelines"],
    "spark": ["spark", "apache spark", "pyspark"],
    "hadoop": ["hadoop"],
    "airflow": ["airflow"],

    # ---- Databases ----
    "mysql": ["mysql"],
    "postgresql": ["postgres", "postgresql", "psql"],
    "mongodb": ["mongo", "mongodb"],
    "redis": ["redis"],
    "oracle": ["oracle db", "oracle database"],
    "sqlite": ["sqlite"],
    "nosql": ["nosql"],
    "elasticsearch": ["elasticsearch", "elastic search"],

    # ---- Cloud / DevOps ----
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "ci/cd": ["ci/cd", "cicd", "ci cd", "continuous integration"],
    "jenkins": ["jenkins"],
    "git": ["git", "github", "gitlab", "bitbucket"],
    "linux": ["linux", "unix", "ubuntu"],
    "terraform": ["terraform"],
    "ansible": ["ansible"],

    # ---- Testing / QA ----
    "selenium": ["selenium", "selenium webdriver"],
    "automation testing": ["automation testing", "test automation"],
    "manual testing": ["manual testing"],
    "api testing": ["api testing", "postman"],
    "junit": ["junit"],
    "testng": ["testng"],
    "cypress": ["cypress"],
    "playwright": ["playwright"],
    "jira": ["jira"],
    "agile": ["agile", "agile methodology"],
    "scrum": ["scrum"],
    "test cases": ["test case", "test cases", "test plan", "test plans"],
    "bug tracking": ["bug tracking", "defect tracking"],

    # ---- Sales / Marketing ----
    "sales": ["sales", "b2b sales", "b2c sales", "field sales"],
    "lead generation": ["lead generation", "lead gen", "prospecting"],
    "crm": ["crm", "customer relationship management"],
    "salesforce": ["salesforce"],
    "negotiation": ["negotiation"],
    "cold calling": ["cold calling", "tele calling", "telecalling"],
    "digital marketing": ["digital marketing"],
    "seo": ["seo", "search engine optimization"],
    "sem": ["sem", "search engine marketing", "google ads"],
    "content writing": ["content writing", "copywriting", "content creation"],
    "social media": ["social media marketing", "social media", "smm"],
    "email marketing": ["email marketing", "email campaigns"],
    "market research": ["market research", "market analysis"],
    "brand management": ["brand management", "branding"],
    "campaign management": ["campaign management", "campaigns"],
    "google analytics": ["google analytics", "ga"],

    # ---- Finance / Accounting ----
    "accounting": ["accounting", "accountancy", "accountant"],
    "bookkeeping": ["bookkeeping", "book keeping"],
    "financial analysis": ["financial analysis", "fin analysis"],
    "financial modeling": ["financial modeling", "financial modelling"],
    "budgeting": ["budgeting", "budget planning"],
    "forecasting": ["forecasting"],
    "taxation": ["taxation", "tax filing", "income tax", "direct tax", "indirect tax"],
    "audit": ["audit", "auditing", "internal audit", "statutory audit"],
    "reconciliation": ["reconciliation", "bank reconciliation"],
    "tally": ["tally", "tally erp", "tally prime"],
    "sap": ["sap", "sap fico"],
    "quickbooks": ["quickbooks"],
    "accounts payable": ["accounts payable", "ap"],
    "accounts receivable": ["accounts receivable", "ar"],
    "gst": ["gst", "goods and services tax"],
    "tds": ["tds", "tax deducted at source"],
    "payroll": ["payroll", "payroll processing"],

    # ---- HR / Operations ----
    "recruitment": ["recruitment", "recruiting", "hiring"],
    "talent acquisition": ["talent acquisition", "ta"],
    "onboarding": ["onboarding"],
    "employee engagement": ["employee engagement"],
    "hr policies": ["hr policies", "hr policy"],
    "performance management": ["performance management", "appraisal", "appraisals"],
    "compensation": ["compensation", "comp and ben"],
    "training and development": ["training and development", "l&d", "learning and development"],
    "vendor management": ["vendor management"],
    "supply chain": ["supply chain", "scm"],
    "inventory management": ["inventory management", "inventory"],
    "logistics": ["logistics"],
    "procurement": ["procurement", "purchasing"],
    "operations": ["operations management", "operations"],
    "process improvement": ["process improvement", "process optimization"],
    "quality control": ["quality control", "qc", "quality assurance"],
    "six sigma": ["six sigma", "lean six sigma"],

    # ---- General / soft skills ----
    "communication": ["communication", "verbal communication", "written communication"],
    "leadership": ["leadership", "team lead", "team leadership"],
    "project management": ["project management", "pmp"],
    "presentation": ["presentation", "presentation skills"],
    "stakeholder management": ["stakeholder management"],
    "problem solving": ["problem solving", "analytical thinking"],
    "time management": ["time management"],
    "customer service": ["customer service", "customer support"],
    "teamwork": ["teamwork", "team work", "collaboration"],
    "documentation": ["documentation", "technical writing"],
    "ms office": ["ms office", "microsoft office", "ms-office"],
    "powerpoint": ["powerpoint", "ms powerpoint", "ppt"],
    "word": ["ms word", "microsoft word"],
}

# Flattened (alias -> canonical) for fast lookup
ALIAS_TO_CANONICAL = {}
for canonical, aliases in SKILL_ALIASES.items():
    for a in aliases:
        ALIAS_TO_CANONICAL[a.lower()] = canonical


DEGREE_PATTERNS = [
    ("MBA", [r"\bmba\b", r"master of business administration"]),
    ("BTech", [r"\bb\.?\s?tech\b", r"bachelor of technology"]),
    ("MTech", [r"\bm\.?\s?tech\b", r"master of technology"]),
    ("BE", [r"\bb\.?\s?e\.?\b", r"bachelor of engineering"]),
    ("ME", [r"\bm\.?\s?e\.?\b", r"master of engineering"]),
    ("BCA", [r"\bbca\b", r"bachelor of computer applications?"]),
    ("MCA", [r"\bmca\b", r"master of computer applications?"]),
    ("BSc", [r"\bb\.?\s?sc\.?\b", r"bachelor of science"]),
    ("MSc", [r"\bm\.?\s?sc\.?\b", r"master of science"]),
    ("BCom", [r"\bb\.?\s?com\.?\b", r"bachelor of commerce"]),
    ("MCom", [r"\bm\.?\s?com\.?\b", r"master of commerce"]),
    ("CA", [r"\bca\b(?!.{0,3}@)", r"chartered accountant"]),
    ("CPA", [r"\bcpa\b"]),
    ("PhD", [r"\bph\.?d\.?\b", r"doctorate"]),
    ("Diploma", [r"\bdiploma\b"]),
    ("Bachelor", [r"\bbachelor['s]*\b"]),
    ("Master", [r"\bmaster['s]*\b"]),
]


SECTION_HEADERS = {
    "summary": ["summary", "objective", "profile", "about me", "career objective"],
    "experience": ["experience", "employment", "work history", "work experience",
                   "professional experience", "career"],
    "education": ["education", "academic", "academics", "qualification",
                  "educational qualification"],
    "skills": ["skills", "technical skills", "core competencies", "expertise",
               "key skills", "skill set", "technologies"],
    "projects": ["projects", "personal projects", "academic projects", "certifications",
                 "certificates", "achievements", "publications", "training"],
}


EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s()]{8,}\d)")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#]*(?:[.\-][A-Za-z0-9+#]+)*")
YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:years|yrs|year)\s*(?:of)?\s*(?:experience|exp)?", re.I)
DATE_RANGE_RE = re.compile(
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(\d{4})\s*[-–—to]+\s*"
    r"(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*)?(\d{4}|present|current|now)",
    re.I,
)


# ----------------------------------------------------------------------
# File extraction
# ----------------------------------------------------------------------

def extract_text_from_file(filename, fileobj):
    """Extract plain text from a PDF / DOCX / TXT file-like object."""
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            text = []
            with pdfplumber.open(fileobj) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text.append(page_text)
            return "\n".join(text)
        elif name.endswith(".docx"):
            document = docx.Document(fileobj)
            return "\n".join(p.text for p in document.paragraphs)
        elif name.endswith(".txt"):
            return fileobj.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""
    return ""


def extract_resumes_from_zip(fileobj):
    """Return list of (filename, text, raw_bytes) for every supported file in a ZIP."""
    results = []
    with zipfile.ZipFile(fileobj) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            base = info.filename.rsplit("/", 1)[-1]
            if base.startswith(".") or base.startswith("__"):
                continue
            lower = base.lower()
            if not (lower.endswith(".pdf") or lower.endswith(".docx") or lower.endswith(".txt")):
                continue
            with z.open(info) as f:
                data = f.read()
            text = extract_text_from_file(base, io.BytesIO(data))
            results.append((base, text, data))
    return results


# ----------------------------------------------------------------------
# Contact extraction
# ----------------------------------------------------------------------

def extract_contact_info(text):
    email_match = EMAIL_RE.search(text)
    email = email_match.group(0) if email_match else None

    phone_match = PHONE_RE.search(text)
    phone = phone_match.group(0).strip() if phone_match else None

    name = None
    for line in text.splitlines()[:30]:
        line = line.strip()
        if not line:
            continue
        if EMAIL_RE.search(line) or PHONE_RE.search(line):
            continue
        # name heuristic: 1-4 words, short, mostly letters
        words = line.split()
        if 1 <= len(words) <= 4 and len(line) <= 50 and re.match(r"^[A-Za-z.\-' ]+$", line):
            # avoid section headers like "EXPERIENCE", "SUMMARY"
            if line.lower() in {s for aliases in SECTION_HEADERS.values() for s in aliases}:
                continue
            if line.lower() in ALL_NOISE:
                continue
            name = line.title()
            break

    return {
        "name": name or "Unknown Candidate",
        "email": email or "Not found",
        "phone": phone or "Not found",
    }


# ----------------------------------------------------------------------
# Tokenizing / keyword extraction
# ----------------------------------------------------------------------

def tokenize(text):
    out = []
    for w in WORD_RE.findall(text):
        w = w.lower().strip(".-")
        if w:
            out.append(w)
    return out


def normalize_text(text):
    """lowercase, collapse whitespace, remove punctuation noise."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9.+#/&\-\s]", " ", text.lower())).strip()


def top_keywords(jd_text, n=15):
    """Pick informative, non-generic keywords from the JD.

    The previous version pulled in *any* frequent word — that's how location
    / company / boilerplate (e.g. "multiversity", "pune", "fresher") were
    ending up in the keyword set and dragging accuracy down. This version
    filters against ALL_NOISE and prefers terms that look like real skills
    (in our vocabulary or 4+ chars and not a common verb).
    """
    tokens = tokenize(jd_text)
    freq = {}
    for t in tokens:
        if t in ALL_NOISE or len(t) < 3 or t.isdigit():
            continue
        # Prefer canonical skills detected in JD (heavy boost)
        weight = 3 if t in ALIAS_TO_CANONICAL else 1
        freq[t] = freq.get(t, 0) + weight

    # also seed with skills found by phrase matching
    jd_skills = extract_skills(jd_text)
    for s in jd_skills:
        freq[s] = freq.get(s, 0) + 5

    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    seen = set()
    out = []
    for word, _ in ranked:
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
        if len(out) >= n:
            break
    return out


def extract_skills(text):
    """Find the canonical skills present in `text` (phrase-aware, alias-aware)."""
    norm = " " + normalize_text(text) + " "
    found = set()
    for alias, canonical in ALIAS_TO_CANONICAL.items():
        # Use word boundary that respects multi-word phrases
        pattern = r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])"
        if re.search(pattern, norm):
            found.add(canonical)
    return sorted(found)


# ----------------------------------------------------------------------
# Section detection
# ----------------------------------------------------------------------

def detect_sections(resume_text):
    lower = resume_text.lower()
    found = {}
    for section, aliases in SECTION_HEADERS.items():
        found[section] = any(re.search(r"(?<![a-z])" + re.escape(a) + r"(?![a-z])", lower)
                             for a in aliases)
    return found


# ----------------------------------------------------------------------
# Experience extraction
# ----------------------------------------------------------------------

def required_years(jd_text):
    matches = [int(m.group(1)) for m in YEARS_RE.finditer(jd_text)]
    return max(matches) if matches else None


def candidate_years(resume_text):
    """Estimate years of experience from the resume.

    Strategy: take the larger of (highest "X years" statement, sum of date
    ranges in employment history). Falls back to None if neither is found.
    """
    text_years = [int(m.group(1)) for m in YEARS_RE.finditer(resume_text)]
    explicit = max(text_years) if text_years else 0

    current_year = datetime.utcnow().year
    range_total = 0
    for m in DATE_RANGE_RE.finditer(resume_text):
        start = int(m.group(2))
        end_str = m.group(4).lower() if m.group(4) else ""
        if end_str.isdigit():
            end = int(end_str)
        elif end_str in ("present", "current", "now"):
            end = current_year
        else:
            continue
        if 1980 < start <= end <= current_year + 1:
            range_total += max(0, end - start)

    if range_total == 0 and explicit == 0:
        return None
    return max(range_total, explicit)


# ----------------------------------------------------------------------
# Education
# ----------------------------------------------------------------------

def extract_degrees(text):
    lower = text.lower()
    found = []
    for label, patterns in DEGREE_PATTERNS:
        for p in patterns:
            if re.search(p, lower):
                if label not in found:
                    found.append(label)
                break
    return found


# ----------------------------------------------------------------------
# Sub-scores
# ----------------------------------------------------------------------

def cosine_score(jd_text, resume_text):
    if not jd_text.strip() or not resume_text.strip():
        return 0.0
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4000)
        tfidf = vec.fit_transform([jd_text, resume_text])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except ValueError:
        return 0.0


def score_format(resume_text):
    sections = detect_sections(resume_text)
    section_score = sum(1 for v in sections.values() if v) / len(sections) * 100

    word_count = len(tokenize(resume_text))
    if word_count < 80:
        length_score = 30
    elif word_count < 200:
        length_score = 70
    elif word_count <= 1200:
        length_score = 100
    elif word_count <= 1800:
        length_score = 80
    else:
        length_score = 60

    # Penalize resumes with no contact info findable (rough format hint)
    score = round(0.7 * section_score + 0.3 * length_score)
    return score, sections, word_count


def score_experience(jd_text, resume_text):
    req = required_years(jd_text)
    have = candidate_years(resume_text)
    sections = detect_sections(resume_text)
    has_exp_section = sections.get("experience", False)

    if req:
        if have is not None:
            ratio = have / req
            if ratio >= 1.0:
                score = 100
            elif ratio >= 0.8:
                score = 80
            elif ratio >= 0.5:
                score = 55
            else:
                score = 30
        else:
            score = 45 if has_exp_section else 15
    else:
        if have is not None:
            score = min(100, 55 + have * 8)
        else:
            score = 55 if has_exp_section else 25

    return round(score), req, have, has_exp_section


def score_education(jd_text, resume_text):
    resume_degrees = extract_degrees(resume_text)
    jd_degrees = extract_degrees(jd_text)

    if not resume_degrees:
        return 25, resume_degrees, jd_degrees

    if jd_degrees:
        overlap = any(d in resume_degrees for d in jd_degrees)
        return (100 if overlap else 60), resume_degrees, jd_degrees

    return 85, resume_degrees, jd_degrees


def score_contact(contact):
    score = 0
    if contact["name"] != "Unknown Candidate":
        score += 34
    if contact["email"] != "Not found":
        score += 33
    if contact["phone"] != "Not found":
        score += 33
    return score


# ----------------------------------------------------------------------
# Main analyzer
# ----------------------------------------------------------------------

def analyze_resume(jd_text, resume_text, jd_keywords):
    contact = extract_contact_info(resume_text)

    # Keyword coverage
    norm_resume = normalize_text(resume_text)
    matched_keywords = [k for k in jd_keywords if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", norm_resume)]
    missing_keywords = [k for k in jd_keywords if k not in matched_keywords]
    keyword_score = round((len(matched_keywords) / len(jd_keywords)) * 100) if jd_keywords else 0

    # Skills alignment (canonical, alias-aware)
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched_skills = [s for s in jd_skills if s in resume_skills]
    missing_skills = [s for s in jd_skills if s not in resume_skills]
    extra_skills = [s for s in resume_skills if s not in jd_skills]

    if jd_skills:
        skills_score = round((len(matched_skills) / len(jd_skills)) * 100)
    else:
        # if we can't pick any structured skills from JD, fall back on semantic similarity
        skills_score = round(cosine_score(jd_text, resume_text) * 100)

    # Semantic match (TF-IDF + bigrams) — represents "Resume vs JD Match %"
    cos = cosine_score(jd_text, resume_text)
    semantic_score = round(cos * 100)

    # Other section scores
    exp_score, req_years, cand_years, has_exp_section = score_experience(jd_text, resume_text)
    edu_score, resume_degrees, jd_degrees = score_education(jd_text, resume_text)
    fmt_score, sections, word_count = score_format(resume_text)
    contact_score = score_contact(contact)

    # Weighted overall
    overall = (
        0.30 * skills_score
        + 0.15 * keyword_score
        + 0.15 * semantic_score
        + 0.18 * exp_score
        + 0.10 * edu_score
        + 0.07 * fmt_score
        + 0.05 * contact_score
    )
    overall = round(max(0, min(100, overall)))

    section_scores = {
        "keywords": keyword_score,
        "format": fmt_score,
        "experience": exp_score,
        "skills": skills_score,
        "education": edu_score,
        "contact": contact_score,
    }

    # Passed / Warnings / Issues — like JobMorph's pills
    passed, warnings, issues = audit_resume(section_scores, sections, contact,
                                            req_years, cand_years, missing_skills)

    reasons = build_reasoning(overall, section_scores, matched_skills, missing_skills,
                              extra_skills, req_years, cand_years, has_exp_section,
                              resume_degrees, jd_degrees, sections)

    suggestions = build_suggestions(section_scores, missing_skills, missing_keywords,
                                    sections, req_years, cand_years, contact, resume_degrees)

    return {
        "score": overall,
        "semantic_score": semantic_score,
        "section_scores": section_scores,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_skills": extra_skills,
        "req_years": req_years,
        "cand_years": cand_years,
        "resume_degrees": resume_degrees,
        "sections": sections,
        "word_count": word_count,
        "reasons": reasons,
        "suggestions": suggestions,
        "passed_checks": passed,
        "warning_checks": warnings,
        "issue_checks": issues,
        "contact": contact,
    }


def audit_resume(sec, sections, contact, req_years, cand_years, missing_skills):
    """Walk a checklist of resume health checks. Each check goes into one of three buckets."""
    checks = []
    # Format / structure
    for s in ("summary", "experience", "education", "skills"):
        if sections.get(s):
            checks.append(("pass", f"{s.title()} section present"))
        else:
            checks.append(("issue", f"Missing {s} section"))

    # Contact details
    if contact["email"] != "Not found":
        checks.append(("pass", "Email address found"))
    else:
        checks.append(("issue", "Email address missing"))
    if contact["phone"] != "Not found":
        checks.append(("pass", "Phone number found"))
    else:
        checks.append(("warning", "Phone number not detected"))

    # Skills
    if sec["skills"] >= 75:
        checks.append(("pass", "Strong skills alignment with the JD"))
    elif sec["skills"] >= 45:
        checks.append(("warning", "Several required skills missing"))
    else:
        checks.append(("issue", "Skills do not align with the role"))

    # Keywords
    if sec["keywords"] >= 60:
        checks.append(("pass", "Resume covers most JD-specific terminology"))
    elif sec["keywords"] >= 30:
        checks.append(("warning", "Resume covers only some JD terms"))
    else:
        checks.append(("issue", "Resume covers very few JD terms"))

    # Experience
    if req_years and cand_years is not None:
        if cand_years >= req_years:
            checks.append(("pass", f"Meets {req_years}+ years of experience"))
        else:
            checks.append(("issue", f"Below {req_years}+ years of required experience"))
    elif req_years and cand_years is None:
        checks.append(("warning", "Years of experience not clearly stated"))

    # Education
    if sec["education"] >= 80:
        checks.append(("pass", "Education matches requirements"))
    elif sec["education"] >= 50:
        checks.append(("warning", "Degree found but may not exactly match JD"))
    else:
        checks.append(("issue", "No matching qualification found"))

    passed = [m for k, m in checks if k == "pass"]
    warnings = [m for k, m in checks if k == "warning"]
    issues = [m for k, m in checks if k == "issue"]
    return passed, warnings, issues


def build_reasoning(overall, sec, matched_skills, missing_skills, extra_skills,
                    req_years, cand_years, has_exp_section, resume_degrees, jd_degrees, sections):
    reasons = []

    if sec["skills"] >= 75:
        reasons.append(
            f"Strong skills match — the resume demonstrates {len(matched_skills)} of the skills required by the JD"
            + (f" ({', '.join(matched_skills[:6])})." if matched_skills else ".")
        )
    elif sec["skills"] >= 40:
        reasons.append(
            "Partial skills alignment — some core skills are present but the resume is missing "
            f"{', '.join(missing_skills[:6]) if missing_skills else 'several key areas'}."
        )
    else:
        reasons.append(
            "Weak skills alignment — the resume does not demonstrate "
            f"{', '.join(missing_skills[:6]) if missing_skills else 'the core skills'} required by this role."
        )

    if req_years:
        if cand_years is not None and cand_years >= req_years:
            reasons.append(f"Meets the experience requirement ({cand_years}+ years vs {req_years}+ years required).")
        elif cand_years is not None:
            reasons.append(f"Below the required experience level ({cand_years} years found vs {req_years}+ required).")
        else:
            reasons.append(f"Could not confirm {req_years}+ years of experience — the resume does not clearly state it.")
    elif cand_years:
        reasons.append(f"Resume indicates approximately {cand_years} years of relevant experience.")
    elif not has_exp_section:
        reasons.append("No clear work-experience section was found in the resume.")

    if sec["education"] >= 80:
        reasons.append(f"Education matches the requirement ({', '.join(resume_degrees[:2]) if resume_degrees else 'qualification found'}).")
    elif sec["education"] >= 50:
        reasons.append("A degree was found, but it may not exactly match the qualification preferred for this role.")
    else:
        reasons.append("No recognizable educational qualification was detected in the resume.")

    missing_sections = [s for s, present in sections.items() if not present]
    if sec["format"] >= 80:
        reasons.append("Resume is well-structured with clear sections (ATS-friendly).")
    elif missing_sections:
        reasons.append(f"Resume formatting could be improved — missing sections: {', '.join(missing_sections)}.")

    if extra_skills and overall >= 50:
        reasons.append(f"Additional relevant skills on the resume: {', '.join(extra_skills[:6])}.")

    return reasons


def build_suggestions(sec, missing_skills, missing_keywords, sections, req_years,
                      cand_years, contact, resume_degrees):
    """Concrete, actionable improvement suggestions (analogous to JobMorph's AI Suggestions)."""
    s = []
    if missing_skills:
        s.append(f"Add evidence of these skills: {', '.join(missing_skills[:6])}.")
    if missing_keywords and len(missing_keywords) >= 3:
        s.append(f"Mention these JD terms explicitly: {', '.join(missing_keywords[:6])}.")
    missing_sections = [x for x, present in sections.items() if not present]
    if missing_sections:
        s.append(f"Add a clear {', '.join(missing_sections)} section to improve ATS parsing.")
    if req_years and (cand_years is None or cand_years < req_years):
        s.append(f"Quantify total years of experience clearly (JD requires {req_years}+ years).")
    if contact["phone"] == "Not found":
        s.append("Include a phone number near the top of the resume.")
    if contact["email"] == "Not found":
        s.append("Include a professional email address.")
    if not resume_degrees:
        s.append("Spell out the highest qualification (e.g., 'B.Tech in Computer Science').")
    if sec["format"] < 60:
        s.append("Use standard section headings (Summary, Experience, Skills, Education).")
    return s[:10]


# ---------------------------------------------------------------------------
# Claude integration — bulk-safe, cost-efficient
#
# Three mechanisms keep 200-resume batches reliable and cheap:
#
#  1. PROMPT CACHING  — the system prompt + full JD text are marked with
#     cache_control. After the first resume, every subsequent call reads those
#     tokens from cache at ~10% of the normal input cost. On a 200-resume scan
#     the JD is typically 500-2 000 tokens; caching it saves ~90% of that cost
#     for resumes 2-200.
#
#  2. CONCURRENCY LIMITER — a global semaphore caps parallel API calls at 5.
#     The Anthropic API has per-minute token and request-rate limits; firing
#     200 threads simultaneously would hit those limits and cause errors. Five
#     concurrent calls keeps throughput high while staying comfortably under
#     the default rate limits.
#
#  3. RETRY WITH EXPONENTIAL BACKOFF — rate-limit (429) and transient server
#     (529) errors are retried up to 4 times with doubling delays (2 s → 4 s
#     → 8 s → 16 s). Any other error falls through to the rule-based engine
#     rather than crashing the whole scan.
# ---------------------------------------------------------------------------

_CLAUDE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "score": {"type": "integer"},
        "semantic_score": {"type": "integer"},
        "skills_score": {"type": "integer"},
        "experience_score": {"type": "integer"},
        "education_score": {"type": "integer"},
        "matched_skills": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "extra_skills": {"type": "array", "items": {"type": "string"}},
        "req_years": {"type": ["integer", "null"]},
        "cand_years": {"type": ["integer", "null"]},
        "reasons": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "passed_checks": {"type": "array", "items": {"type": "string"}},
        "warning_checks": {"type": "array", "items": {"type": "string"}},
        "issue_checks": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "score", "semantic_score", "skills_score", "experience_score",
        "education_score", "matched_skills", "missing_skills", "extra_skills",
        "req_years", "cand_years", "reasons", "suggestions",
        "passed_checks", "warning_checks", "issue_checks",
    ],
}

_CLAUDE_SYSTEM_PROMPT = """You are a senior technical recruiter screening candidates for a specific role.
You read a job description (JD) and a candidate's resume and produce a strict, \
discriminating evaluation that a hiring manager can trust.

Principles:
- Be evidence-based. Only credit a skill if the resume actually shows it (projects, \
work, tools used) — not just because a word appears once.
- Be discriminating. Do NOT cluster every candidate around 60-70. A weak/irrelevant \
candidate should clearly score low (below 45); a strong, directly relevant candidate \
should clearly score high (80+).
- Distinguish "required" vs "nice to have" skills in the JD and weight required ones more.
- Judge experience by both years AND relevance/depth, not just keywords.
- Return ONLY the JSON object requested — no markdown fences, no commentary."""

# Shared client — one connection pool reused across all threads
_claude_client_lock = threading.Lock()
_claude_client = None

# Max 5 parallel API calls — keeps us under Anthropic's default rate limits
_claude_semaphore = threading.Semaphore(5)


def _get_claude_client():
    global _claude_client
    with _claude_client_lock:
        if _claude_client is None:
            import anthropic
            _claude_client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
                timeout=180.0,
                max_retries=0,  # we handle retries ourselves
            )
    return _claude_client


def _claude_debug(msg):
    if os.environ.get("SCREENGENIE_DEBUG"):
        print(f"[screengenie:claude] {msg}", file=sys.stderr)


def _strip_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return text


def _call_claude_with_retry(client, **kwargs):
    """Call client.messages.create with exponential backoff on rate-limit errors."""
    import anthropic
    delays = [2, 4, 8, 16]
    for attempt, delay in enumerate(delays + [None]):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            if delay is None:
                raise
            _claude_debug(f"rate limit hit, retrying in {delay}s (attempt {attempt + 1}/4)")
            time.sleep(delay)
        except anthropic.InternalServerError as e:
            if delay is None:
                raise
            _claude_debug(f"server error ({e}), retrying in {delay}s (attempt {attempt + 1}/4)")
            time.sleep(delay)


def analyze_resume_with_claude(jd_text, resume_text, jd_keywords):
    """Claude-powered resume analysis — bulk-safe, cached, rate-limit resilient.

    Returns the same dict structure as analyze_resume(), or None on failure
    (caller falls back to the rule-based engine automatically).
    Set SCREENGENIE_DEBUG=1 to see per-resume diagnostics.
    """
    try:
        import anthropic
    except ImportError:
        _claude_debug("anthropic package not installed — run: pip install -U anthropic")
        return None

    if not os.environ.get("ANTHROPIC_API_KEY"):
        _claude_debug("ANTHROPIC_API_KEY not set")
        return None

    # ---- Exact/mechanical fields computed in Python (not delegated to Claude) ----
    contact = extract_contact_info(resume_text)
    contact_score = score_contact(contact)
    sections = detect_sections(resume_text)
    word_count = len(tokenize(resume_text))
    resume_degrees = extract_degrees(resume_text)
    fmt_score, _, _ = score_format(resume_text)

    norm_resume = normalize_text(resume_text)
    matched_keywords = [
        k for k in jd_keywords
        if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", norm_resume)
    ]
    missing_keywords = [k for k in jd_keywords if k not in matched_keywords]
    keyword_score = round((len(matched_keywords) / len(jd_keywords)) * 100) if jd_keywords else 0

    keywords_hint = ", ".join(jd_keywords) if jd_keywords else "none extracted"

    # Full documents — 40k-char cap only against corrupted giant files
    jd_for_model = jd_text[:40000]
    resume_for_model = resume_text[:40000]

    # ---- Prompt layout for caching ----
    # The system prompt + JD block are identical for every resume in a scan.
    # We mark them with cache_control so after the FIRST resume the tokens are
    # served from cache at ~10% cost for all subsequent ones.
    system_blocks = [
        {
            "type": "text",
            "text": _CLAUDE_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    jd_block = {
        "type": "text",
        "text": (
            f"=== JOB DESCRIPTION ===\n{jd_for_model}\n\n"
            f"Pre-extracted JD keywords: {keywords_hint}"
        ),
        "cache_control": {"type": "ephemeral"},  # cached — same for all 200 resumes
    }

    resume_block = {
        "type": "text",
        "text": (
            f"=== CANDIDATE RESUME ===\n{resume_for_model}\n\n"
            "Evaluate how well this resume matches the job description above.\n"
            "Return a strict recruiter evaluation as a JSON object:\n"
            "- score (0-100): overall fit. Weights: skills 40%, experience 25%, "
            "education 15%, JD relevance 20%.\n"
            "- semantic_score: conceptual alignment of resume language with JD.\n"
            "- skills_score: % of JD REQUIRED skills the candidate clearly demonstrates.\n"
            "- experience_score: years + depth vs JD requirement.\n"
            "- education_score: qualification vs JD requirement.\n"
            "- matched_skills / missing_skills / extra_skills: skill lists.\n"
            "- req_years / cand_years: integers or null.\n"
            "- reasons: 3-5 specific sentences citing concrete evidence.\n"
            "- suggestions: up to 6 actionable improvements for THIS candidate.\n"
            "- passed_checks / warning_checks / issue_checks: ATS audit bullets.\n"
            "Be strict — missing core skills → score below 45. Strong match → 80+."
        ),
        # No cache_control here — this changes per resume
    }

    messages = [{"role": "user", "content": [jd_block, resume_block]}]

    call_kwargs = dict(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=system_blocks,
        messages=messages,
    )
    try:
        call_kwargs_with_schema = {
            **call_kwargs,
            "output_config": {"format": {"type": "json_schema", "schema": _CLAUDE_JSON_SCHEMA}},
        }
    except Exception:
        call_kwargs_with_schema = call_kwargs

    try:
        client = _get_claude_client()
        with _claude_semaphore:  # max 5 parallel calls
            try:
                resp = _call_claude_with_retry(client, **call_kwargs_with_schema)
            except TypeError:
                _claude_debug("SDK lacks output_config; using prompt-only JSON mode")
                resp = _call_claude_with_retry(client, **call_kwargs)

        result_text = next(
            (b.text for b in resp.content if getattr(b, "type", None) == "text"), ""
        )
        if not result_text:
            _claude_debug("no text block in response")
            return None

        # Log cache efficiency when debug mode is on
        u = resp.usage
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        uncached = getattr(u, "input_tokens", 0) or 0
        _claude_debug(
            f"{contact['name']}: tokens in={uncached} cached={cached} out={u.output_tokens}"
        )

        data = json.loads(_strip_json(result_text))

    except Exception as e:
        _claude_debug(f"Claude failed: {type(e).__name__}: {e}")
        return None

    def _clamp(v):
        try:
            return max(0, min(100, int(round(float(v)))))
        except (TypeError, ValueError):
            return 0

    overall = _clamp(data.get("score", 0))
    skills_score = _clamp(data.get("skills_score", 0))
    exp_score = _clamp(data.get("experience_score", 0))
    edu_score = _clamp(data.get("education_score", 0))
    semantic_score = _clamp(data.get("semantic_score", 0))

    _claude_debug(
        f"score={overall} skills={skills_score} exp={exp_score} edu={edu_score} "
        f"sem={semantic_score}"
    )

    return {
        "score": overall,
        "semantic_score": semantic_score,
        "section_scores": {
            "keywords": keyword_score,
            "format": fmt_score,
            "experience": exp_score,
            "skills": skills_score,
            "education": edu_score,
            "contact": contact_score,
        },
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "matched_skills": data.get("matched_skills", []),
        "missing_skills": data.get("missing_skills", []),
        "extra_skills": data.get("extra_skills", []),
        "req_years": data.get("req_years"),
        "cand_years": data.get("cand_years"),
        "resume_degrees": resume_degrees,
        "sections": sections,
        "word_count": word_count,
        "reasons": data.get("reasons", []),
        "suggestions": data.get("suggestions", [])[:10],
        "passed_checks": data.get("passed_checks", []),
        "warning_checks": data.get("warning_checks", []),
        "issue_checks": data.get("issue_checks", []),
        "contact": contact,
    }


def grade_for_score(score):
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def status_for_score(score):
    if score >= 70:
        return "Shortlist"
    if score >= 45:
        return "Review"
    return "Reject"


# ============================================================
# JD Quality Analysis
# ============================================================

_JD_ANALYSIS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "overall_score": {"type": "integer"},
        "clarity_score": {"type": "integer"},
        "completeness_score": {"type": "integer"},
        "keyword_richness_score": {"type": "integer"},
        "role_definition_score": {"type": "integer"},
        "attractiveness_score": {"type": "integer"},
        "seniority_alignment_score": {"type": "integer"},
        "missing_sections": {"type": "array", "items": {"type": "string"}},
        "missing_keywords": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {"type": "string"}},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "suggestions": {"type": "array", "items": {"type": "string"}},
        "seniority_level": {"type": "string"},
        "detected_role": {"type": "string"},
    },
    "required": [
        "overall_score", "clarity_score", "completeness_score",
        "keyword_richness_score", "role_definition_score",
        "attractiveness_score", "seniority_alignment_score",
        "missing_sections", "missing_keywords", "issues",
        "strengths", "suggestions", "seniority_level", "detected_role",
    ],
}

_JD_SECTION_HINTS = ["responsibilities", "requirements", "qualifications",
                     "about", "benefits", "skills", "compensation", "culture"]


def analyze_jd_quality(jd_text):
    """Rule-based JD quality analysis — used as fallback when Claude is unavailable."""
    lower = jd_text.lower()
    word_count = len(tokenize(jd_text))

    section_hits = sum(1 for s in _JD_SECTION_HINTS if s in lower)
    completeness = round((section_hits / len(_JD_SECTION_HINTS)) * 100)

    skills = extract_skills(jd_text)
    keyword_richness = min(100, len(skills) * 9)

    if word_count < 80:    clarity = 25
    elif word_count < 200: clarity = 55
    elif word_count < 400: clarity = 75
    else:                  clarity = 90

    has_years = bool(YEARS_RE.search(jd_text))
    level_words = ["senior", "junior", "lead", "manager", "intern", "entry", "mid-level", "associate"]
    has_level = any(w in lower for w in level_words)
    role_definition = 40 + (30 if has_years else 0) + (30 if has_level else 0)
    attractiveness = 50
    seniority_alignment = 55

    overall = round(0.20 * clarity + 0.20 * completeness + 0.20 * keyword_richness
                    + 0.15 * role_definition + 0.15 * attractiveness + 0.10 * seniority_alignment)

    missing_sections = [s.title() for s in _JD_SECTION_HINTS if s not in lower]
    issues, strengths, suggestions = [], [], []

    if word_count < 150:
        issues.append("JD is too short — lacks sufficient detail for candidates to evaluate fit")
    if not has_years:
        issues.append("No experience requirement (years) mentioned")
    if not has_level:
        issues.append("Seniority level not specified (Junior / Mid / Senior)")
    if completeness < 50:
        issues.append(f"Missing key sections: {', '.join(missing_sections[:3])}")

    if word_count >= 300:
        strengths.append("Good length with sufficient detail")
    if len(skills) >= 5:
        strengths.append(f"Contains {len(skills)} specific skill keywords")
    if has_years:
        strengths.append("Experience requirement is clearly stated")

    if not has_years:
        suggestions.append("Add a specific years-of-experience requirement (e.g. '2+ years')")
    if not has_level:
        suggestions.append("Specify the seniority level (Junior / Mid-Level / Senior)")
    if completeness < 80:
        suggestions.append("Add missing sections: " + ", ".join(missing_sections[:3]))
    if len(skills) < 5:
        suggestions.append("Add more specific technical skill requirements")

    level_map = {"intern": "Internship", "junior": "Junior", "senior": "Senior",
                 "lead": "Lead", "manager": "Manager", "associate": "Associate"}
    seniority = next((v for k, v in level_map.items() if k in lower), "Mid-Level")
    detected_role = next((l.strip()[:80] for l in jd_text.splitlines() if l.strip()), "Unknown Role")

    return {
        "overall_score": overall,
        "clarity_score": clarity,
        "completeness_score": completeness,
        "keyword_richness_score": keyword_richness,
        "role_definition_score": role_definition,
        "attractiveness_score": attractiveness,
        "seniority_alignment_score": seniority_alignment,
        "missing_sections": missing_sections,
        "missing_keywords": [],
        "issues": issues,
        "strengths": strengths,
        "suggestions": suggestions,
        "seniority_level": seniority,
        "detected_role": detected_role,
    }


def analyze_jd_quality_with_claude(jd_text, description=""):
    """Claude-powered JD quality analysis."""
    try:
        import anthropic
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    client = _get_claude_client()
    desc_hint = f"\nHiring manager's context: {description}" if description else ""

    prompt = (
        "You are an expert HR consultant reviewing a Job Description for quality, "
        "effectiveness, and ATS-readiness.\n\n"
        "=== JOB DESCRIPTION ===\n"
        f"{jd_text[:20000]}\n"
        f"{desc_hint}\n\n"
        "Score each dimension 0-100. Be strict: 0-40 poor, 41-60 average, 61-80 good, 81+ excellent.\n"
        "Dimensions:\n"
        "- clarity_score: Are responsibilities and expectations crystal clear?\n"
        "- completeness_score: All sections present? (summary, responsibilities, requirements, qualifications, company info)\n"
        "- keyword_richness_score: Relevant technical/domain keywords that ATS and candidates search for?\n"
        "- role_definition_score: Seniority, team context, reporting structure, scope of work clear?\n"
        "- attractiveness_score: Would strong candidates want to apply? (growth, culture, comp hinted)\n"
        "- seniority_alignment_score: JD content matches implied/stated seniority level?\n"
        "- overall_score: Weighted overall (clarity 20%, completeness 20%, keywords 20%, "
        "role_def 15%, attractiveness 15%, seniority 10%)\n\n"
        "missing_keywords: important domain/technical keywords absent from this JD.\n"
        "missing_sections: section names that should exist but are absent.\n"
        "issues: specific problems (3-6 items).\n"
        "strengths: what the JD does well (2-4 items).\n"
        "suggestions: actionable improvements (3-6 items).\n"
        "seniority_level: one of Internship / Junior / Mid-Level / Senior / Lead / Manager.\n"
        "detected_role: the job title as a short string.\n\n"
        "Return ONLY the JSON object."
    )

    try:
        with _claude_semaphore:
            try:
                resp = _call_claude_with_retry(
                    client,
                    model="claude-opus-4-8",
                    max_tokens=2048,
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": prompt}],
                    output_config={"format": {"type": "json_schema", "schema": _JD_ANALYSIS_SCHEMA}},
                )
            except TypeError:
                resp = _call_claude_with_retry(
                    client,
                    model="claude-opus-4-8",
                    max_tokens=2048,
                    thinking={"type": "adaptive"},
                    messages=[{"role": "user", "content": prompt}],
                )
        result_text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        return json.loads(_strip_json(result_text))
    except Exception as e:
        _claude_debug(f"JD quality analysis failed: {e}")
        return None


def fix_jd_with_claude(jd_text, description, analysis):
    """Rewrite and improve the JD based on the analysis findings."""
    try:
        import anthropic
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    client = _get_claude_client()
    issues_text = "\n".join(f"- {i}" for i in analysis.get("issues", []))
    missing_kw = ", ".join(analysis.get("missing_keywords", []))
    missing_sec = ", ".join(analysis.get("missing_sections", []))
    suggestions_text = "\n".join(f"- {s}" for s in analysis.get("suggestions", []))
    desc_hint = f"\nHiring manager context: {description}" if description else ""

    prompt = (
        "You are an expert HR consultant rewriting a Job Description to make it more "
        "effective, ATS-friendly, and attractive to qualified candidates.\n\n"
        "=== ORIGINAL JD ===\n"
        f"{jd_text[:20000]}\n"
        f"{desc_hint}\n\n"
        "=== WHAT TO FIX ===\n"
        f"Issues:\n{issues_text or 'None critical'}\n\n"
        f"Missing keywords to weave in naturally: {missing_kw or 'None'}\n"
        f"Missing sections to add: {missing_sec or 'None'}\n\n"
        f"Suggestions:\n{suggestions_text or 'None'}\n\n"
        "STRICT RULES:\n"
        "1. Keep ALL existing accurate information — do not remove or contradict anything\n"
        "2. Add missing keywords naturally into responsibilities/requirements — no fabrication\n"
        "3. Add missing sections with realistic content inferred from the original\n"
        "4. Use [Company Name] / [Location] placeholders only where truly unknown\n"
        "5. Do NOT invent qualifications, years, or skills not implied by the original\n"
        "6. Use clear headings: Job Title · About the Role · Key Responsibilities · "
        "Requirements · Nice to Have · What We Offer · About [Company Name]\n"
        "7. Return ONLY the improved JD text — no commentary, no markdown fences."
    )

    try:
        with _claude_semaphore:
            resp = _call_claude_with_retry(
                client,
                model="claude-opus-4-8",
                max_tokens=4096,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": prompt}],
            )
        result_text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        return result_text.strip() if result_text else None
    except Exception as e:
        _claude_debug(f"JD fix failed: {e}")
        return None
