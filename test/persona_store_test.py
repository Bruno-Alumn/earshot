"""Stored personality reads: the first after PERSONA_FIRST lines, the next after
PERSONA_EVERY more, every read kept in personas.json; Claude when there is a key,
else the local model. Claude itself is mocked here (no network, no key).
  cd server && uv run python ../test/persona_store_test.py"""
import json, re, sys, tempfile, time
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
sch = S._codebook_schema()
cs = sch["properties"]["codes"]
check(set(cs["required"]) == set(S.CODEBOOK) and cs["additionalProperties"] is False and sch["additionalProperties"] is False, "codebook schema: every code required, nothing extra")

# fixed scoring rules
sc = S.score_codes({"PUTDOWN": [1, 2, 3, 4, 5, 6, 999, -1, 2], "BANTER": [6], "HELP": list(range(1, 13)), "PLAN": [1, 2], "CARELESS": [3]}, 100)
check(sc["counts"]["PUTDOWN"] == 5, "line numbers outside the excerpt and repeats are dropped; banter wins over put-down")
check(sc["signals"]["antagonism"][0] == 3 and sc["signals"]["warmth"][0] == 3, f"signal rates map to fixed cut-offs ({sc['signals']['antagonism'][0]}, {sc['signals']['warmth'][0]})")
check(all(v[0] is None for v in S.score_codes({"HELP": [1, 2, 3]}, 60)["signals"].values()), "under 100 counted turns, signals are not rated (no lean toward 'none')")
big_sc = S.score_codes({"PUTDOWN": list(range(1, 61))}, 1000)
check(big_sc["signals"]["antagonism"][0] == 3, "with lots of lines the lean toward neutral barely matters")
check(S.score_codes({"PUTDOWN": [9]}, 10, denom=6)["counts"]["PUTDOWN"] == 1, "a tag on a late line still counts when fewer lines count toward the rate (bug fix)")
check(sc["big5"]["C"][0] == 3 and sc["big5"]["O"][0] is None, "Big Five: net per 100 lines; under 3 lines of evidence is not rated")
sc = S.score_codes({"PUTDOWN": [1, 2]}, 40 + 60)
check(sc["signals"]["antagonism"][0] == 2 and S.score_codes({"PUTDOWN": [1, 2, 3, 4, 5, 6]}, 100)["signals"]["antagonism"][0] == 3
      and S.score_codes({"PUTDOWN": [1, 2]}, 100, denom=100)["signals"]["antagonism"][0] == 2, "agreed lines needed: 1 for a score of 1, 2 for 2, 3 for 3")
sc = S.score_codes({"COOPERATE": [1, 2, 3, 4, 5]}, 50)
check(sc["big5"]["A"][0] == 4, "a 5 needs at least 8 lines of evidence")
check(S.score_codes({"INITIATE": [1, 2, 3]}, 300, talk_ratio=2.0)["big5"]["E"][0] == S.score_codes({"INITIATE": [1, 2, 3]}, 300, talk_ratio=0.3)["big5"]["E"][0],
      "how much someone talks no longer changes Extraversion (it's shown, not scored)")
check(S.score_codes({}, 200, talk_ratio=3.0)["big5"]["E"][0] is None, "talk share alone doesn't rate Extraversion (no tagged evidence)")
check(S.score_codes({"INITIATE": [1, 2, 3]}, 30, talk_ratio=3.0)["big5"]["E"][0] == 4, "the talk step can't push past 4 on under 8 lines")
check(S.score_codes({"WITHDRAWN": [1, 2, 3, 4, 5, 6]}, 100)["big5"]["E"][0] == 2, "Extraversion can go below average (WITHDRAWN)")
sc = S.score_codes({"OUTBURST": [1, 2], "RASH": [1, 2], "UPSET": [1, 2]}, 100)
check(sc["signals"]["impulsiveness"][1].startswith("2 of") and sc["big5"]["S"][0] is None, "a line with two codes for one trait counts once (evidence is lines, not tags)")
sc = S.score_codes({"BANTER": [1, 2, 3, 4], "DISMISS": [1, 2], "BLAME": [3], "ARGUE": [4]}, 100)
check(sc["counts"]["DISMISS"] == 0 and sc["counts"]["BLAME"] == 0 and sc["counts"]["ARGUE"] == 0, "banter cancels dismiss, blame and argue on the same line too")
check(S.score_codes({"HELP": [1, 2, 3, 4]}, 100)["big5"]["A"][0] is None, "routine help counts toward warmth only, not Agreeableness")
check(S.score_codes({}, 50)["confidence"] == "low" and S.score_codes({}, 500)["confidence"] == "high", "confidence from how many lines were read")
nt, nn = S.number_excerpt("--- call ---\n  · Ann: hi\n>> one\n>> two")
check(nn == 2 and ">> [2] two" in nt, "lines numbered in order")
big = "--- call on x ---\n" + "\n".join(f"  · Ann: q{i}\n>> [{i}] a{i} said this" for i in range(1, 301))
ch = S.excerpt_chunks(big)
check([(a, b) for _, a, b in ch] == [(1, 120), (121, 240), (241, 300)], f"tagged in pieces of 120 lines ({[(a, b) for _, a, b in ch]})")
check(all(t.startswith("--- call on x") for t, _, _ in ch) and "  · Ann: q121" in ch[1][0], "each piece keeps its call header and the context before its first line")
check(ch[0][0].rstrip().endswith("  · Ann: q121"), "...and the context right after a piece's last line stays with that piece too")
two = "--- call on a ---\n>> [1] x\n  · Bo: after\n--- call on b ---\n  · Cy: before\n>> [2] y"
ch2 = S.excerpt_chunks(two, size=1)
check(len(ch2) == 2 and "Bo: after" in ch2[0][0] and "Bo: after" not in ch2[1][0] and ch2[1][0].startswith("--- call on b"), "a new call starts a clean piece")

# the book
calls = []
S.claude_persona_real = S.claude_persona
S.claude_persona = lambda name, excerpt, key, *a: (calls.append((name, excerpt.count(">> "))) or (built, "claude-sonnet-5-5"))
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
check(len(v["reads"]) == 1 and v["reads"][0]["model"] == "Claude Sonnet 5.5" and v["reads"][0]["lines"] == 45, "the read is stored with its model and line count")
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
check(len(v["reads"]) == 2 and v["reads"][-1]["model"] == S.LLM_NAME and v["reads"][0]["model"] == "Claude Sonnet 5.5", "second read kept next to the first (history)")
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

got, model = S.claude_json("sys", {}, "user", "k")
check(got["summary"].startswith("John (Gator)") and model == "claude-opus-4-8", "fallback answer parsed (not the declined partial)")
Resp.content = [Blk("text", '{"summary": "John comes'), Blk("fallback"), Blk("text", json.dumps(sample)[len('{"summary": "John comes'):])]
got, _ = S.claude_json("sys", {}, "user", "k")
check(got.get("openness"), "fallback that continues the partial is joined and parsed")

# a whole codebook read through the mocked client: tags per piece, then a summary
seen = []
def fake_json(system, schema, user, key):
    seen.append(user)
    if "summary" in schema.get("properties", {}):
        return {"summary": "P comes across as a helpful planner."}, "claude-sonnet-5-5"
    a, b = map(int, re.search(r"lines (\d+) to (\d+)", user).groups())
    odd = sum(1 for u in seen if "Part of" in u) % 2      # every other coder also tags a PLAN line the other doesn't
    return {"codes": {c: ([x for x in range(a, b + 1) if x % 10 == 0] if c == "HELP" else [b + 5] if c == "PUTDOWN" else [a] if c == "PLAN" and odd else []) for c in S.CODEBOOK}}, "claude-sonnet-5-5"
S.claude_json, real_json = fake_json, S.claude_json
built2, _ = S.claude_persona_real("John", big, "k", 300, 1.0)
d2 = built2["data"]
check(len(seen) == 3 * S.CODERS + 1 and "Tagged lines" in seen[-1], f"each of 3 pieces tagged {S.CODERS} times, then 1 summary call ({len(seen)})")
check(d2["counts"]["PLAN"] == 0 and d2["agreement"] == round(30 / 33, 2), f"only tags both coders gave count; agreement recorded ({d2['agreement']})")
check(d2["counts"]["HELP"] == 30 and d2["counts"]["PUTDOWN"] == 0, "codes merged across pieces; numbers outside a piece are dropped")
check(d2["method"] == S.READ_METHOD and d2["n"] == 300 and d2["summary"].startswith("John comes across"), "read records its method, line count and summary (P turned back into the name)")
check(not any("John" in u for u in seen), "the coders and the summary never see the person's name")
# lines that can't show anything don't dilute the rates
short = "--- call ---\n" + "\n".join(f">> [{i}] {'yeah' if i % 2 else 'that was a good push'}" for i in range(1, 101))
seen.clear()
def fake2(system, schema, user, key):
    seen.append(user)
    if "summary" in schema.get("properties", {}):
        return {"summary": "P is fine."}, "m"
    return {"codes": {c: ([2, 4, 6] if c == "HELP" else [1, 3] if c == "ENCOURAGE" else [8] if c == "GARBLED" else []) for c in S.CODEBOOK}}, "m"
S.claude_json = fake2
d3 = S.claude_persona_real("John", short, "k", 100)[0]["data"]
check(d3["n"] == 51 and d3["short"] == 48 and d3["garbled"] == 1, f"one-word lines and garbled lines leave the count; tagged short lines stay ({d3['n']}, {d3['short']}, {d3['garbled']})")
S.claude_json = real_json
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

# stored reads tagged the current way but scored by older rules are scored again on load
old = {"at": 1, "lines": 300, "used": 120, "method": "codebook-4", "name": "Old", "text": "",
       "tagged": {c: ([1, 2, 3, 4, 5, 6] if c == "HELP" else []) for c in S.CODEBOOK},
       "data": {"summary": "Old is helpful.", "n": 50, "talkRatio": 1.2, "signals": [], "big5": [], "agreement": 0.7}}
(root / "personas.json").write_text(json.dumps({"999999999999999999": {"reads": [old]}}))
book3 = S.PersonaBook(hub, Reader())
r3 = book3.data["999999999999999999"]["reads"][0]
check(r3["method"] == S.READ_METHOD and r3["data"]["rescoredFrom"] in ("codebook-4", "codebook-5") and r3["data"]["summary"] == "Old is helpful." and r3["data"]["agreement"] == 0.7,
      "an older-rules read is scored again, keeping its summary and agreement")
check(all(x["score"] is None for x in r3["data"]["signals"]), "...and with 50 counted turns its signals are now not rated")

# a gendered summary is dropped (the scores stay)
def fake3(system, schema, user, key):
    if "summary" in schema.get("properties", {}):
        return {"summary": "P is a loud player and he jokes a lot."}, "m"
    return {"codes": {c: [] for c in S.CODEBOOK}}, "m"
S.claude_json = fake3
check(S.claude_persona_real("John", short, "k", 100)[0]["data"]["summary"] == "", "a summary that guesses he/she is dropped")
S.claude_json = real_json

# lines said back to back become one turn
d5 = "2026-01-07"; base5 = int(time.mktime(time.strptime(d5, "%Y-%m-%d"))) * 1000 + 3600_000
with (S.TRANSCRIPTS / f"{d5}.jsonl").open("w") as f:
    rows5 = [(0, A, "John", "so"), (900, A, "John", "basically we"), (1800, A, "John", "should push left"), (3000, B, "Gator", "ok"), (3500, A, "John", "cool"), (9000, A, "John", "later thought here")]
    for k, (dt_, uid, who, text) in enumerate(rows5):
        f.write(json.dumps({"id": f"z{k}", "t": base5 + dt_, "end": base5 + dt_ + 500, "user": uid, "speaker": who, "text": text, "ch": "v", "chName": "C", "guildName": "G"}) + "\n")
S._day_cache.clear()
part5 = S.person_excerpt(A, 10**6, context=True)[0].split("--- call on " + d5)[1].split("\n---")[0]
mine5 = [x for x in part5.splitlines() if x.startswith(">> ")]
check(mine5 == [">> so basically we should push left", ">> cool", ">> later thought here"], f"back-to-back lines join into one turn; someone speaking between or a pause splits them ({mine5})")

# names are hidden from the coders: the person is P, others letters, in labels and in the text
class Hub2: people_dir = {A: {"username": "yangsauce", "names": ["John"], "nicks": []}, B: {"username": "gatorvg", "names": ["Gator"], "nicks": []}}
S.PERSONA_BOOK = type("PB", (), {"hub": Hub2()})()
d4 = "2026-01-06"; base4 = int(time.mktime(time.strptime(d4, "%Y-%m-%d"))) * 1000 + 3600_000
with (S.TRANSCRIPTS / f"{d4}.jsonl").open("w") as f:
    for k, (dt_, uid, who, text) in enumerate([(0, B, "Gator", "John you there"), (500, A, "John", "yeah Gator I got you, @gatorvg"), (900, B, "Gator", "thanks john")]):
        f.write(json.dumps({"id": f"y{k}", "t": base4 + dt_, "end": base4 + dt_ + 300, "user": uid, "speaker": who, "text": text, "ch": "v", "chName": "Secret Channel", "guildName": "Secret Server"}) + "\n")
S._day_cache.clear()
an = S.person_excerpt(A, 10**6, context=True, anonymous=True)[0]
part = an.split("--- call on " + d4)[1].split("\n---")[0]
check("John" not in part and "Gator" not in part and "gatorvg" not in part and "Secret" not in part, f"no names, channel or server in the anonymous excerpt ({part!r})")
check("  · A: P you there" in part and ">> yeah A I got you, A" in part and "  · A: thanks P" in part, "the person is P and the other speaker a letter, in labels and in the text")
S.PERSONA_BOOK = None

# the key never goes into settings.json
S.save_secret("anthropicKey", "sk-ant-xyz")
check("sk-ant" not in (root / "settings.json").read_text() and json.loads((root / "secrets.json").read_text())["anthropicKey"] == "sk-ant-xyz", "key kept in secrets.json, not settings")
check(oct((root / "secrets.json").stat().st_mode)[-3:] == "600", "secrets.json readable by this user only")
S.save_secret("anthropicKey", None)
check("anthropicKey" not in json.loads((root / "secrets.json").read_text()), "key removed")

print("\n" + ("All passed." if not fails else f"{len(fails)} failed"))
sys.exit(1 if fails else 0)
