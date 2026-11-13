"""Reading resume files: column layouts, Word tables and text boxes, scanned files."""
import datetime
import io
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import engine  # noqa: E402
import experience as X  # noqa: E402
import parsing  # noqa: E402

T = datetime.date(2026, 11, 13)
FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def read(name):
    with open(os.path.join(FIX, name), "rb") as f:
        return parsing.extract_text_from_file(name, f)


def test_two_column_pdf_keeps_columns_apart():
    t = read("two_column.pdf")
    assert parsing.COLUMN_BREAK in t
    e = X.from_text(t, T)
    assert X.format_duration(e["work"]) == "9 yrs 5 mos"            # Jul 2017 -> Nov 2026
    kinds = {r["text"]: r["kind"] for r in e["roles"]}
    assert kinds["2013 - 2017"] == "excluded" and kinds["Jan 2020 - Mar 2020"] == "excluded"
    ci = parsing.extract_contact_info(t)
    assert ci["name"] == "Meera Krishnan" and ci["phone"] == "+91 99001 22334"


def test_right_aligned_dates_are_not_a_second_column():
    t = read("right_dates.pdf")
    assert parsing.COLUMN_BREAK not in t
    e = X.from_text(t, T)
    assert X.format_duration(e["work"]) == "10 yrs 5 mos" and e["internship"] == 6


def test_single_column_pdf_unchanged():
    t = read("one_column.pdf")
    assert parsing.COLUMN_BREAK not in t
    assert X.format_duration(X.from_text(t, T)["work"]) == "10 yrs 5 mos"


def test_docx_tables_stay_in_place_and_text_boxes_are_read():
    t = read("table_middle.docx")
    assert t.index("Flipkart") < t.index("EDUCATION")
    t = read("textbox.docx")
    assert "HR Business Partner, Swiggy" in t and t.index("Swiggy") < t.index("EDUCATION")
    assert X.format_duration(X.from_text(t, T)["work"]) == "8 yrs 6 mos"


def test_scanned_pdf_is_flagged_not_silently_scored():
    t = read("scanned.pdf")
    ok, why = parsing.read_quality(t)
    assert not ok and "scanned" in why


def test_read_quality_accepts_normal_text_and_rejects_glued_words():
    assert parsing.read_quality(read("one_column.pdf"))[0]
    glued = "SeniorDataAnalystAtFlipkartBangalore " * 10
    assert not parsing.read_quality(glued)[0]


class FakeClient:
    def __init__(self, payload):
        self.messages = self
        self.payload = payload
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        usage = SimpleNamespace(input_tokens=1500, output_tokens=400, cache_read_input_tokens=0,
                                cache_creation_input_tokens=0)
        return SimpleNamespace(stop_reason="end_turn", model="claude-sonnet-5-5", usage=usage,
                               content=[SimpleNamespace(type="text", text=json.dumps(self.payload))])


def test_claude_transcribes_a_scanned_pdf():
    with open(os.path.join(FIX, "scanned.pdf"), "rb") as f:
        data = f.read()
    fake = FakeClient({"readable": True, "text": "Arjun Mehta\nEXPERIENCE\nSales Manager, Byju's   Jun 2019 - Present"})
    text, meta = engine.transcribe_document(data, "scanned.pdf", client=fake)
    assert "Sales Manager" in text and meta["path"] == "ai" and meta["kind"] == "transcribe"
    block = fake.calls[0]["messages"][0]["content"][0]
    assert block["type"] == "document" and block["source"]["media_type"] == "application/pdf"
    text, meta = engine.transcribe_document(b"\x89PNG....", "cv.png", client=FakeClient({"readable": False, "text": ""}))
    assert text == ""


def test_read_resume_reports_unreadable_files(monkeypatch):
    os.environ.update(SECRET_KEY="x" * 40, TEAM_PASSWORD="hunter2-team-pass")
    for m in ("app", "auth"):
        sys.modules.pop(m, None)
    import app as app_module
    with open(os.path.join(FIX, "scanned.pdf"), "rb") as f:
        data = f.read()
    monkeypatch.setattr(engine, "llm_available", lambda: False)
    text, problem, notes, metas = app_module.read_resume("scanned.pdf", data)
    assert problem and "scanned" in problem and not metas
    # with Claude available, the scanned file is transcribed and gets a note for HR
    monkeypatch.setattr(engine, "llm_available", lambda: True)
    monkeypatch.setattr(engine, "transcribe_document", lambda d, n: (
        "Arjun Mehta\nEXPERIENCE\nSales Manager, Byju's   Jun 2019 - Present\nSales Executive, Airtel  "
        "Jan 2016 - May 2019\nEDUCATION\nMBA, Symbiosis 2014 - 2016", dict(engine.new_meta("transcribe"), path="ai")))
    text, problem, notes, metas = app_module.read_resume("scanned.pdf", data)
    assert problem is None and "Sales Manager" in text and "scanned" in notes[0] and metas[0]["path"] == "ai"
