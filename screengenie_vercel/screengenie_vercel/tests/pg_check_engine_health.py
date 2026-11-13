import os, sys
os.environ.update(DATABASE_URL="postgresql://postgres:pw@127.0.0.1:5499/postgres", SECRET_KEY="x"*40, TEAM_PASSWORD="hunter2-team-pass", PGSSLMODE="disable")
sys.path.insert(0, ".")
import pg8000.native, pg8000
import pg8000.dbapi
_orig = pg8000.dbapi.connect
pg8000.dbapi.connect = lambda *a, **k: _orig(*a, **dict(k, ssl_context=None))
import app as A, engine
c = A.app.test_client()
with c.session_transaction() as s: s["signed_in"] = True
r = c.get("/engine-health"); print("empty page:", r.status_code, b"Engine health" in r.data)
assert r.status_code == 200, r.data[:3000]
with A.app.app_context():
    db = A.get_db(); cur = db.cursor()
    ok = dict(engine.new_meta("resume"), path="ai", model="claude-sonnet-5-5", input_tokens=5000, output_tokens=1200, cache_read_tokens=3000, seconds=7.5, cost_usd=0.025, attempts=1)
    bad = dict(engine.new_meta("resume"), error="RateLimitError: slow down", seconds=1.0)
    A.log_engine_run(cur, 1, "a.pdf", ok); A.log_engine_run(cur, 1, "b.pdf", ok); A.log_engine_run(cur, 1, "c.pdf", bad)
    A.log_engine_run(cur, 1, "d.pdf", dict(engine.new_meta("requisition"), path="ai"))
    # a failing log insert must not poison the transaction
    A.log_engine_run(cur, 1, "e.pdf", dict(ok, input_tokens="not-a-number"))
    cur.execute("SELECT COUNT(*) FROM engine_runs"); print("rows after bad insert:", cur.fetchone()[0])
    db.commit()
r = c.get("/engine-health"); t = r.data.decode()
print("data page:", r.status_code, "67%" in t, "RateLimitError" in t, "₹" in t)
assert r.status_code == 200 and "RateLimitError" in t
