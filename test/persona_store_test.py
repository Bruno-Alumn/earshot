"""Stored personality reads: the first after PERSONA_FIRST lines, the next after
PERSONA_EVERY more, every read kept in personas.json; Claude when there is a key,
else the local model. Claude itself is mocked here (no network, no key).
  cd server && uv run python ../test/persona_store_test.py"""
import json, sys, tempfile, time
from pathlib import Path
sys.argv = sys.argv[:1]; sys.path.insert(0, ".")
import server as S
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

root = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root / "transcripts"; S.TRANSCRIPTS.mkdir()
A, B = "111111111111111111", "222222222222222222"
day = time.strftime("%Y-%m-%d")
t0 = int(time.time() * 1000) - 3_600_000
def write(rows):
    with (S.TRANSCRIPTS / f"{day}.jsonl").open("a", encoding="utf-8") as f:
        for t, uid, who, text in rows:
            f.write(json.dumps({"id": f"{day}-{t}", "t": t, "end": t + 1500, "user": uid, "speaker": who, "text": text,
                                "ch": "v2", "chName": "Voice 2", "guild": "g", "guildName": "WARDOGS"}) + "\n")
    S._people_cache = None
    S.day_data.cache_clear() if hasattr(S.day_data, "cache_clear") else None
rows, t = [], t0
for i in range(45):
    rows.append((t, B, "Gator", f"where should we push number {i}")); t += 2000
    rows.append((t, A, "John", f"go left side, I've got the flank {i}")); t += 2000
for i in range(10):
    rows.append((t, B, "Gator", "ok")); t += 2000
write(rows)

class Hub: people_dir = {}
hub = Hub()

# excerpt: their lines marked >>, others' lines just before as context
text, used, words = S.person_excerpt(A, 20000, context=True)
ls = text.splitlines()
check(ls[0].startswith("--- call on") and "Voice 2" in ls[0], "excerpt starts with the call header")
check(used == 45 and sum(1 for x in ls if x.startswith(">> ")) == 45, f"all 45 of John's lines ({used})")
check(any(x.startswith("  · Gator: where should we push") for x in ls), "context lines from others, named")
check(ls[1].startswith("  · Gator") and ls[2].startswith(">> "), "context comes just before their line")
small, used2, _ = S.person_excerpt(A, 40, context=False)
check(used2 < 45 and small.splitlines()[-1].endswith("44"), f"a small budget keeps the newest lines ({used2})")
check(not any(x.startswith("  · ") for x in small.splitlines()), "no context lines without context")

# Claude's JSON -> the same report shape as before
sample = {"summary": "John (Gator) comes across as a focused shot-caller.", "self_focus": {"score": 1, "reason": "mentions own flank"},
          "antagonism": {"score": 0, "reason": "none"}, "manipulation": {"score": 0, "reason": "none"},
          "impulsiveness": {"score": 1, "reason": "quick calls"}, "warmth": {"score": 2, "reason": "covers teammates"},
          "openness": {"score": 3, "reason": "x"}, "conscientiousness": {"score": 4, "reason": "plans"}, "extraversion": {"score": 4, "reason": "talks a lot"},
          "agreeableness": {"score": 4, "reason": "cooperative"}, "stability": {"score": 4, "reason": "calm"}, "quotes": ["go left side"]}
built = S.persona_from_json(sample)
d = built["data"]
check(d["summary"].startswith("John comes across"), "a second name in brackets is stripped from the summary")
check(len(d["signals"]) == 5 and len(d["big5"]) == 5 and d["signals"][-1]["key"] == "warmth", "five signals (warmth last) and Big Five")
check(d["traits"] == ["warm", "outgoing", "easygoing"], f"traits from the scores, at most three ({d['traits']})")
sch = S._persona_schema()
check(set(sch["required"]) == set(sch["properties"]) and sch["additionalProperties"] is False, "schema: every field required, nothing extra")

# the book
calls = []
S.claude_persona_real = S.claude_persona
S.claude_persona = lambda name, excerpt, key: (calls.append((name, excerpt.count(">> "))) or (built, "claude-opus-5-5"))
class Reader:
    def read_lines(self, name, said): calls.append(("local", len(said))); return built
book = S.PersonaBook(hub, Reader())
S.claude_key = lambda: ""
check(S.persona_engine() == "local", "no key: the local model writes reads")
S.claude_key = lambda: "sk-ant-test"
check(S.persona_engine() == "claude", "with a key and Claude picked: Claude")
S.save_settings(personaModel="local")
check(S.persona_engine() == "local", "LFM3 picked: local even with a key")
S.save_settings(personaModel="claude")

check(book._due() == B, "the person heard most recently goes first (Gator, 55 lines)")
book.status[B] = {"state": "reading"}
check(book._due() == A, "then John (45 lines)")
book.status.pop(B)
v = book.view(B)
check(v["lines"] == 55, f"Gator's line count ({v['lines']})")
book._read(A)
check(calls and calls[-1] == ("John", 45), f"Claude got John's name and 45 lines ({calls[-1:]})")
v = book.view(A)
check(len(v["reads"]) == 1 and v["reads"][0]["model"] == "Claude Opus 5.5" and v["reads"][0]["lines"] == 45, "the read is stored with its model and line count")
check(v["next"] == 45 + S.PERSONA_EVERY, f"next read at {45 + S.PERSONA_EVERY} lines ({v['next']})")
due = book._due()
check(due == B, f"after John, Gator is still due ({due})")
book._read(B)
check(book._due() is None, "nobody due once both have a read")
saved = json.loads((root / "personas.json").read_text())
check(set(saved) == {A, B}, "both reads saved in personas.json")
book2 = S.PersonaBook(hub, Reader())
check(len(book2.view(A)["reads"]) == 1, "reads survive a restart")

# 400 more lines -> due again; 399 -> not
more, t2 = [], t
for i in range(399):
    more.append((t2, A, "John", f"line {i}")); t2 += 1000
write(more)
check(book._due() is None, "399 new lines: not due yet")
write([(t2, A, "John", "one more")])
check(book._due() == A, "400 new lines: due again")
S.save_settings(personaModel="local")
book._read(A)
v = book.view(A)
check(len(v["reads"]) == 2 and v["reads"][-1]["model"] == S.LLM_NAME and v["reads"][0]["model"] == "Claude Opus 5.5", "second read kept next to the first (history)")
check(calls[-1][0] == "local" and calls[-1][1] > 0, "local model got their lines only")
S.save_settings(personaModel="claude")

# asking: queues only when due, or when forced
r = book.ask(A)
check(r["state"] is None and not book.queue, "opening a read that's up to date queues nothing")
r = book.ask(A, force=True)
check(r["state"] == "queued" and book.queue == [A], "Read again queues it")
check(book.ask("333333333333333333").get("error"), "unknown person: an error")

# a failure is kept and shown, and not retried for a while
book.queue.clear(); book.status.clear()
def boom(*a): raise S.PersonaError("Couldn't reach Anthropic. Earshot tries again later.", retry=True)
S.claude_persona = boom
write([(t2 + 5000 + i * 1000, A, "John", "x") for i in range(400)])
try:
    book._read(A)
except S.PersonaError as e:
    book._fail(A, str(e), e.retry)
check(book.view(A)["error"].startswith("Couldn't reach"), "the error shows in the view")
check(book._due() is None, "a failed read waits before trying again")
check(book.ask(A)["state"] is None and not book.queue, "opening it while it waits doesn't retry (no paid call every 30 s)")
book.errors[A]["at"] -= 1801
check(book._due() == A, "network error: tried again after 30 min")
book._fail(A, "again", True); book.errors[A]["at"] -= 1801
check(book._due() is None, "second failure in a row: waits an hour")
def refuse(*a): raise S.PersonaError("Claude declined to read this person.")
S.claude_persona = refuse
book.errors.pop(A)
try:
    book._read(A)
except S.PersonaError as e:
    book._fail(A, str(e), e.retry)
book.errors[A]["at"] -= 86400
check(book._due() is None, "a refusal isn't retried on a timer")
write([(t2 + 900000 + i * 1000, A, "John", "y") for i in range(100)])
check(book._due() == A, "...only after 100 more lines")
check(book.ask(A, force=True)["state"] == "queued", "Read again always retries")
book.queue.clear(); book.status.clear(); book.errors.clear()

# server-side fallback: parse the answer after the last fallback block
class Blk:
    def __init__(self, type, text=""): self.type, self.text = type, text
class Resp:
    stop_reason = "end_turn"; model = "claude-opus-4-8"
    content = [Blk("text", '{"summary": "half'), Blk("fallback"), Blk("text", json.dumps(sample))]
class Stream:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def get_final_message(self): return Resp()
import anthropic
class FakeClient:
    def __init__(self, **kw): self.beta = type("B", (), {"messages": type("M", (), {"stream": staticmethod(lambda **kw: Stream())})()})()
real_cls, anthropic.Anthropic = anthropic.Anthropic, FakeClient
import importlib
cp = S.__dict__["claude_persona_real"] if "claude_persona_real" in S.__dict__ else None

got, model = S.claude_persona_real("John", ">> hi", "k")
check(got["data"]["summary"].startswith("John comes across") and model == "claude-opus-4-8", "fallback answer parsed (not the declined partial)")
Resp.content = [Blk("text", '{"summary": "John comes'), Blk("fallback"), Blk("text", json.dumps(sample)[len('{"summary": "John comes'):])]
got, _ = S.claude_persona_real("John", ">> hi", "k")
check(got["data"]["big5"], "fallback that continues the partial is joined and parsed")
anthropic.Anthropic = real_cls

# deleting lines drops reads that may have used them
before_reads = len(book.view(A)["reads"])
book.forget({A: 0})
check(book.view(A)["reads"] == [] and A not in json.loads((root / "personas.json").read_text()), f"deleting John's lines forgets his reads ({before_reads} -> 0)")
book.forget({B: 10**15})
check(len(book.view(B)["reads"]) == 1, "reads made before the deleted lines stay")

# two calls at once: the excerpt follows the line's own channel
d2 = "2026-01-05"; base = int(time.mktime(time.strptime(d2, "%Y-%m-%d"))) * 1000 + 3600_000
with (S.TRANSCRIPTS / f"{d2}.jsonl").open("w") as f:
    for k, (dt_, ch, uid, who, text) in enumerate([(0, "v1", B, "Gator", "other channel talk"), (500, "v2", "444444444444444444", "Ann", "same channel talk"), (1000, "v2", A, "John", "my line")]):
        f.write(json.dumps({"id": f"x{k}", "t": base + dt_, "end": base + dt_ + 300, "user": uid, "speaker": who, "text": text, "ch": ch, "chName": {"v1": "Voice 1", "v2": "Voice 2"}[ch], "guildName": "W"}) + "\n")
S._day_cache.clear()
ex = [x for x in S.person_excerpt(A, 10**6, context=True)[0].split("--- call on " + d2)[1].split("\n---")[0].splitlines()]
check("Voice 2" in ex[0] and "  · Ann: same channel talk" in ex and not any("Gator" in x for x in ex), f"call header and context from the same channel ({ex[:3]})")

# the key never goes into settings.json
S.save_secret("anthropicKey", "sk-ant-xyz")
check("sk-ant" not in (root / "settings.json").read_text() and json.loads((root / "secrets.json").read_text())["anthropicKey"] == "sk-ant-xyz", "key kept in secrets.json, not settings")
check(oct((root / "secrets.json").stat().st_mode)[-3:] == "600", "secrets.json readable by this user only")
S.save_secret("anthropicKey", None)
check("anthropicKey" not in json.loads((root / "secrets.json").read_text()), "key removed")

print("\n" + ("All passed." if not fails else f"{len(fails)} failed"))
sys.exit(1 if fails else 0)
