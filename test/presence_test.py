"""Voice presence: stints (who was in which voice channel, from when to when) built
from the main account's lobby snapshots, "in voice with" totals, and the API views.
No live data: everything runs on temp folders.
  cd server && .venv/bin/python ../test/presence_test.py"""
import json, sys, tempfile, time
from pathlib import Path
sys.argv = sys.argv[:1]; sys.path.insert(0, ".")
import server as S
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok: fails.append(what)

root = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root / "transcripts"; S.TRANSCRIPTS.mkdir()
G, H = "900000000000000001", "900000000000000002"            # tracked server, other server
C1, C2, C3 = "800000000000000001", "800000000000000002", "800000000000000003"
A, B, D, E, O = ("100000000000000001", "100000000000000002", "100000000000000004",
                 "100000000000000005", "100000000000000009")   # O is one of your own accounts
T = int(time.time() * 1000) - 3_600_000
SCOPE = {"mode": "recorded", "ids": {G}, "names": set()}

def person(uid, since=None, **f):
    return {"id": uid, "since": since, "mute": False, "deaf": False, "stream": False, "video": False, **f}

def lobby(at, chans, other=None, ids=(G, H), capped=False):
    gs = [{"id": G, "name": "WARDOGS", "channels": [{"id": c, "name": "Voice " + c[-1], "limit": 0, "people": ps} for c, ps in chans.items() if ps]}]
    gs = [g for g in gs if g["channels"]]
    if other:
        gs.append({"id": H, "name": "Other", "channels": [{"id": C3, "name": "Lounge", "limit": 5, "people": other}]})
    return {"at": at, "ms": 1.0, "guildIds": list(ids), "capped": capped, "guilds": gs, "accounts": {}}

P = S.Presence()
SRC = {"account": "main", "role": "main", "label": "me", "fallback": False}
obs = lambda lob, at, scope=SCOPE, own=frozenset(), rec=frozenset(), tok=1: P.observe(lob, at, SRC, tok, scope, set(own), set(rec))

# --- join: Discord's join time is used when believable
obs(lobby(T, {C1: [person(A, since=T - 600_000), person(B)]}, other=[person(D, since=T - 50_000)]), T, rec={C1})
check(set(P.open) == {A, B}, "two people open in the tracked server; the other server is out of scope")
check(P.open[A]["s"] == T - 600_000 and P.open[A]["from"] == "discord", "start = Discord's join time (first snapshot)")
check(P.open[B]["s"] == T and P.open[B]["from"] == "seen", "no join time: start = first seen")
check(P.open[A]["rec"] is True, "stint in a channel being recorded is marked recorded")
far = int(T - 30 * 86_400_000)
P2 = S.Presence.__new__(S.Presence)        # (just the start rule, with no continuity and an absurd join time)
P2.guild_at = {}
check(S.Presence._start(P2, far, G, T, None)[0] == T, "a join time more than 7 days back isn't believed")

# --- move: one stint ends, another starts
t1 = T + 15_000
obs(lobby(t1, {C1: [person(A, since=T - 600_000)], C2: [person(B, since=T - 999_999)]}), t1)
mid = (T + t1) // 2
b_old = P.by_user[B][-1]
check(b_old["c"] == C1 and not b_old["cut"] and b_old["e"] == mid, f"move: old stint ends between snapshots ({b_old['e'] - T})")
check(P.open[B]["c"] == C2 and P.open[B]["s"] == mid, "move: new stint starts there (a join time from before the last snapshot isn't used)")

# --- leave: closed between the last two snapshots, not cut
t2 = t1 + 15_000
obs(lobby(t2, {C2: [person(B)]}), t2)
a_st = P.by_user[A][-1]
check(A not in P.open and not a_st["cut"] and a_st["e"] == (t1 + t2) // 2, "leave: closed between snapshots, not cut")

# --- stale report: cut at last seen, then the same stint goes on if they're still there soon after
P.check(t2 + 46_000)
check(not P.open and P.by_user[B][-1]["cut"] and P.by_user[B][-1]["e"] == t2, "stale report: open stints closed at last seen, cut")
bid = P.by_user[B][-1]["id"]
t3 = t2 + 100_000
obs(lobby(t3, {C2: [person(B)]}), t3)
check(P.open.get(B, {}).get("id") == bid and not P.open[B]["cut"], "back within 2 min in the same channel: the cut stint is resumed")

# --- disconnect, then back an hour later with the same Discord join time: resumed across the gap
obs(lobby(t3 + 15_000, {C2: [person(B)], C1: [person(D, since=t3)]}), t3 + 15_000)
d_id = P.open[D]["id"]
P.gone(1)
check(not P.open and P.by_user[D][-1]["cut"], "the source window disconnects: everything cut")
t4 = t3 + 3_600_000
obs(lobby(t4, {C1: [person(D, since=t3)], C2: [person(E, since=t3 - 10_000)]}), t4)
check(P.open[D]["id"] == d_id and P.open[D]["s"] == t3, "same Discord join time after an hour: one stint, no overlap")

# --- clamp: a new stint never starts before that person's previous one ended
P.cut_all()
e_end = P.by_user[E][-1]["e"]
t5 = t4 + 900_000
obs(lobby(t5, {C1: [person(E, since=t3 - 10_000)]}), t5)
check(P.open[E]["s"] >= e_end and P.open[E]["c"] == C1, "new stint (other channel) starts no earlier than the last one ended")

# --- scope: all / back to recorded / a server we can't see any more / off
P.cut_all()
t6 = t5 + 600_000
ALL = {"mode": "all", "ids": set(), "names": set()}
obs(lobby(t6, {C1: [person(A)]}, other=[person(D)]), t6, scope=ALL)
check(P.open[D]["g"] == H and P.view and len(P.view["guilds"]) == 2, "mode all: the other server is tracked too")
obs(lobby(t6 + 15_000, {C1: [person(A)]}, other=[person(D)]), t6 + 15_000)
check(D not in P.open and P.by_user[D][-1]["cut"], "back to recorded: stints in the other server are cut, not left")
obs(lobby(t6 + 30_000, {}), t6 + 30_000)
check(A not in P.open and not P.by_user[A][-1]["cut"], "server still listed, nobody in voice: a real leave")
obs(lobby(t6 + 45_000, {C1: [person(A)]}), t6 + 45_000)
obs(lobby(t6 + 60_000, {}, ids=(H,)), t6 + 60_000)
check(A not in P.open and P.by_user[A][-1]["cut"], "the account left that server: cut, not left")
obs(lobby(t6 + 75_000, {C1: [person(A)]}), t6 + 75_000)
obs(lobby(t6 + 90_000, {C1: [person(A)]}), t6 + 90_000, scope={"mode": "off"})
check(not P.open and P.view is None and P.by_user[A][-1]["cut"], "mode off: nothing open, no view")
recs = S.presence_scope(type("Hb", (), {"ext_reports": {}})(), {"watchGuilds": [H]}, "recorded", time.time())
check(H in recs["ids"], "recorded scope includes servers with auto-join channels")
day = time.strftime("%Y-%m-%d", time.localtime(T / 1000))
with (S.TRANSCRIPTS / f"{day}.jsonl").open("w") as f:
    f.write(json.dumps({"id": "x", "t": T, "end": T + 2000, "user": A, "speaker": "Ann", "text": "hello there",
                        "ch": C1, "chName": "Voice 1", "guild": G, "guildName": "WARDOGS"}) + "\n")
S._rec_guilds = (0.0, set(), set())
check(G in S.recorded_guilds()[0], "recorded scope: servers from recorded calls")

# --- overlap math, own accounts left out
P.closed, P.by_user, P.open = [], {}, {}
def st(uid, c, s, e, own=False):
    r = {"id": uid + str(s), "u": uid, "g": G, "gn": "WARDOGS", "c": c, "cn": "Voice", "s": s, "e": e, "cut": False, "own": own, "rec": False}
    P.closed.append(r); P.by_user.setdefault(uid, []).append(r)
st(A, C1, 0, 100_000); st(B, C1, 50_000, 150_000); st(D, C1, 60_000, 70_000); st(O, C1, 0, 200_000, own=True)
st(A, C2, 200_000, 300_000); st(B, C2, 250_000, 260_000); st(E, C1, 200_000, 300_000)
P.recompute(400_000)
check(P.pairs[A][B] == 60 and P.pairs[B][A] == 60, f"A and B: 50 s in one channel + 10 s in another ({P.pairs[A].get(B)})")
check(P.pairs[A][D] == 10 and P.pairs[B][D] == 10, "a short stint overlaps both")
check(O not in P.pairs and all(O not in d for d in P.pairs.values()), "your own accounts are left out of in-voice-with")
check(E not in P.pairs.get(A, {}), "same channel at different times: no overlap")
check(P.totals[A] == [200, 2], "A: 200 s over 2 stints")

# --- storage: one line per stint after a reload, open stints closed as cut after a restart, pruning
root2 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root2 / "transcripts"; S.TRANSCRIPTS.mkdir()
R = S.Presence()
obs2 = lambda lob, at: R.observe(lob, at, SRC, 1, SCOPE, set(), set())
obs2(lobby(T, {C1: [person(A, since=T - 5000)]}), T)
R.check(T + 50_000)                                            # cut (one line)
obs2(lobby(T + 60_000, {C1: [person(A, since=T - 5000)]}), T + 60_000)   # resumed
obs2(lobby(T + 75_000, {}), T + 75_000)                        # left (a second line, same id)
check(len((root2 / "presence.jsonl").read_text().splitlines()) == 2, "a resumed stint is written again (same id)")
R2 = S.Presence()
check(len(R2.closed) == 1 and not R2.closed[0]["cut"] and len((root2 / "presence.jsonl").read_text().splitlines()) == 1,
      "reload: one stint, the file compacted")
obs3 = lambda lob, at: R2.observe(lob, at, SRC, 1, SCOPE, set(), set())
obs3(lobby(T + 200_000, {C2: [person(B)]}), T + 200_000)
R2._write_open()
R3 = S.Presence()
check(not R3.open and R3.by_user[B][-1]["cut"] and R3.by_user[B][-1]["e"] == T + 200_000, "after a restart: open stints closed where last seen, cut")
old = {"id": "old1", "u": D, "g": G, "c": C1, "s": T - 200 * 86_400_000, "e": T - 199 * 86_400_000, "cut": False}
with (root2 / "presence.jsonl").open("a") as f:
    f.write(json.dumps(old) + "\n")
R4 = S.Presence()
check(not any(r["id"] == "old1" for r in R4.closed), "stints older than 180 days are pruned on load")
check("old1" not in (root2 / "presence.jsonl").read_text(), "and removed from the file")
check(R4.prune(T) == 0, "prune: nothing more to drop")

# --- the server side: source choice, names, API shapes
root3 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root3 / "transcripts"; S.TRANSCRIPTS.mkdir()
hub = S.Hub("test")
hub.presence = S.Presence()
now = time.time(); nms = int(now * 1000)
main_ws, lis_ws = object(), object()
hub.conns = {main_ws: {"account": "main", "role": "main", "userId": O, "label": "me", "since": now - 100, "capturing": True},
             lis_ws: {"account": "l1", "role": "listener", "userId": E, "label": "listener", "since": now - 50, "capturing": True}}
lob = lobby(nms, {C1: [person(A, since=nms - 120_000, mute=True), person(O)]})
lob["accounts"] = {A: {"u": "ann", "g": "Ann", "n": "Annie"}}
lob_l = lobby(nms, {C1: [person(B)]})
hub.ext_reports = {main_ws: ({"role": "main", "lobby": lob, "me": {"id": O}, "channel": {"id": C1, "guild": G}, "pcs": 1, "tracks": 1}, now),
                   lis_ws: ({"role": "listener", "lobby": lob_l, "me": {"id": E}}, now)}
check(S.presence_source(hub, now) is main_ws, "source: the main account's report")
S.presence_feed(hub, lis_ws, now)
check(not hub.presence.open, "a listener's lobby is ignored while the main one reports")
S.presence_feed(hub, main_ws, now)
check(set(hub.presence.open) == {A, O} and hub.presence.open[O]["own"], "main lobby used; your own account marked own")
check(hub.presence.open[A]["rec"], "channel your account is recording: stint marked recorded")
check(hub.people_dir[A]["username"] == "ann" and hub.people_dir[A].get("seen") == "voice" and A not in hub.people_dirty,
      "names from the lobby are kept, flagged voice-only, no people file")
check(S.write_people_files(hub, [A]) == 0, "no people/<name>.md for someone only seen in voice")
hub.learn_account(A, {"u": "ann", "g": "Ann", "n": "Annie"})
check("seen" not in hub.people_dir[A] and A in hub.people_dirty, "heard in a call later: an ordinary person record")
v = S.voice_view(hub, now)
check(set(v) >= {"watch", "source", "updatedAt", "stale", "guilds"} and v["watch"] == "recorded" and not v["stale"], "/api/voice keys")
ch = v["guilds"][0]["channels"][0]
pa = next(p for p in ch["people"] if p["id"] == A)
check(ch["recording"] and ch["ours"][0]["account"] == "main" and ch["ours"][0]["recording"], "channel shows your account recording it")
check(pa["name"] == "Annie" and pa["mute"] and pa["since"] == nms - 120_000 and not pa["own"], f"person: name by nameStyle (nick), flags, since ({pa['name']})")
check(v["source"]["role"] == "main" and v["source"]["fallback"] is False, "source named, not a fallback")
del hub.ext_reports[main_ws]; del hub.conns[main_ws]; hub.presence.gone(id(main_ws))
S.presence_feed(hub, lis_ws, now + 1)
v2 = S.voice_view(hub, now + 1)
check(v2["source"]["role"] == "listener" and v2["source"]["fallback"] is True and B in hub.presence.open, "main gone: falls back to a listener, and says so")
check(S.voice_view(hub, now + 60)["stale"], "no fresh report: stale")
hub.presence.recompute(nms + 5000)
vp = S.voice_person(hub, A, now + 5)
check(set(vp) >= {"id", "name", "totalSec", "sessions", "lastSeen", "with", "recent"}, "/api/voice/person keys")
check(vp["sessions"] == 1 and vp["totalSec"] >= 120 and vp["recent"][0]["recorded"] and vp["recent"][0]["cut"], f"person: stints, time, recorded, cut ({vp['totalSec']})")
check(vp["with"] == [], "own account isn't listed as someone they were in voice with")
check(S.voice_person(hub, "100000000000000077", now)["sessions"] == 0, "someone never seen: empty")
json.dumps(v); json.dumps(vp)

# --- names only for tracked servers; the setting; kept out of personality reads
root4 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root4 / "transcripts"; S.TRANSCRIPTS.mkdir()
hub2 = S.Hub("test"); hub2.presence = S.Presence()
ws2 = object(); now2 = time.time(); n2 = int(now2 * 1000)
hub2.conns = {ws2: {"account": "main", "role": "main", "userId": O, "label": "me", "since": now2, "capturing": True}}
lob2 = lobby(n2, {C1: [person(A)]}, other=[person(D)])
lob2["accounts"] = {A: {"u": "ann", "g": "Ann", "n": None}, D: {"u": "dee", "g": "Dee", "n": None}}
S._rec_guilds = (time.time(), {G}, set())
hub2.ext_reports = {ws2: ({"role": "main", "lobby": lob2}, now2)}
S.presence_feed(hub2, ws2, now2)
check(A in hub2.people_dir and D not in hub2.people_dir, "names learned only for people in tracked servers")
check(S.name_finder(hub2)[1].get("ann") is None, "voice-only names don't feed the personality name scrubber")
check("error" in S.set_setting("voiceWatch", "bogus"), "voiceWatch: a bad value is refused")
check(S.set_setting("voiceWatch", "off").get("ok") and S.current_settings()["voiceWatch"] == "off", "voiceWatch: off saved")
lob3 = {**lob2, "at": n2 + 15_000}
hub2.ext_reports = {ws2: ({"role": "main", "lobby": lob3}, now2 + 15)}
S.presence_feed(hub2, ws2, now2 + 15)
v3 = S.voice_view(hub2, now2 + 15)
check(not hub2.presence.open and v3["watch"] == "off" and v3["guilds"] == [], "voiceWatch off through the setting: nothing tracked")
S.set_setting("voiceWatch", "recorded")

# --- a long gap between snapshots that the stale check didn't see: cut, never stretched
root5 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root5 / "transcripts"; S.TRANSCRIPTS.mkdir()
G2 = S.Presence()
G2.observe(lobby(T, {C1: [person(A), person(B)]}), T, SRC, 1, SCOPE, set(), set())
G2.observe(lobby(T + 600_000, {C1: [person(A)]}), T + 600_000, SRC, 1, SCOPE, set(), set())
b = G2.by_user[B][-1]
check(b["cut"] and b["e"] == T, "10 min gap: someone gone is cut at last seen, not left in the middle of the gap")
check(G2.open[A]["s"] >= T and G2.by_user[A][-1]["cut"] and G2.by_user[A][-1]["e"] == T, "still there: old stint cut, a new one after the gap")

# --- review: killed within a minute of a real leave, the saved open copy doesn't replace it
root6 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root6 / "transcripts"; S.TRANSCRIPTS.mkdir()
K = S.Presence()
K.observe(lobby(T, {C1: [person(A, since=T - 5000)]}), T, SRC, 1, SCOPE, set(), set())
K._write_open()                                                  # the minute save, A still open
K.observe(lobby(T + 15_000, {}), T + 15_000, SRC, 1, SCOPE, set(), set())   # A leaves; then the server is killed
K2 = S.Presence()
k = K2.by_user[A]
check(len(k) == 1 and not k[0]["cut"] and k[0]["e"] == T + 7_500, f"kill after a real leave: the leave is kept, not an older cut copy ({k})")

# --- review: one empty snapshot (Discord reconnecting) doesn't split a stint with the same join time
root7 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root7 / "transcripts"; S.TRANSCRIPTS.mkdir()
F = S.Presence()
fo = lambda lob, at, tok=1: F.observe(lob, at, SRC, tok, SCOPE, set(), set())
fo(lobby(T, {C1: [person(A, since=T - 60_000), person(B)]}), T)
a_id = F.open[A]["id"]
fo(lobby(T + 15_000, {}), T + 15_000)
fo(lobby(T + 30_000, {C1: [person(A, since=T - 60_000), person(B)]}), T + 30_000)
check(F.open[A]["id"] == a_id and len(F.by_user.get(A, [])) == 0, "empty snapshot, same join time: the same stint goes on")
check(F.open[B]["id"] != a_id and len(F.by_user[B]) == 1, "no join time to match: a leave and a new stint (as before)")
fo(lobby(T + 45_000, {}), T + 45_000)
fo(lobby(T + 60_000, {C1: [person(A, since=T + 50_000)]}), T + 60_000)
check(F.open[A]["id"] != a_id and F.open[A]["s"] == T + 50_000, "left and rejoined (new join time): a new stint")
# time never runs back: a snapshot older than the last one
fo(lobby(T + 55_000, {C1: [person(A, since=T + 50_000)]}), T + 55_000, tok=2)
check(all(r["e"] >= r["s"] for r in F.closed) and F.updated == T + 60_000, "an older snapshot from another window doesn't move time back")

# --- review: a server whose list was cut short (cap): missing people are cut, not left
root8 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root8 / "transcripts"; S.TRANSCRIPTS.mkdir()
Cp = S.Presence()
Cp.observe(lobby(T, {C1: [person(A), person(B)]}), T, SRC, 1, SCOPE, set(), set())
lc = lobby(T + 15_000, {C1: [person(A)]}); lc["guilds"][0]["capped"] = True; lc["capped"] = True
Cp.observe(lc, T + 15_000, SRC, 1, SCOPE, set(), set())
check(Cp.by_user[B][-1]["cut"] and A in Cp.open, "capped server: someone missing from the list is cut, not counted as leaving")

# --- review: own accounts stay out of "in voice with" even when not connected at the time
root9 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root9 / "transcripts"; S.TRANSCRIPTS.mkdir()
W = S.Presence()
W.observe(lobby(T, {C1: [person(A), person(O)]}), T, SRC, 1, SCOPE, set(), set())     # O not known as ours yet
W.observe(lobby(T + 15_000, {C1: [person(A), person(O)]}), T + 15_000, SRC, 1, SCOPE, {O}, set())
W.observe(lobby(T + 30_000, {C1: [person(A), person(B)]}), T + 30_000, SRC, 1, SCOPE, set(), set())
W.observe(lobby(T + 45_000, {C1: [person(A), person(O), person(B)]}), T + 45_000, SRC, 1, SCOPE, set(), set())
W.cut_all(); W.recompute(T + 100_000)
check(O not in W.pairs and O not in W.pairs.get(A, {}) and W.pairs[A].get(B), f"own account learnt later is still left out of pairs ({W.pairs})")
W2 = S.Presence()
check(O in W2.own_seen, "your account ids are kept across a restart")

# --- review: source stays put between two listeners; an old lobby isn't used
root10 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root10 / "transcripts"; S.TRANSCRIPTS.mkdir()
hub3 = S.Hub("test"); hub3.presence = S.Presence()
l1, l2 = object(), object()
hub3.conns = {l1: {"account": "l1", "role": "listener", "capturing": True, "since": time.time()}, l2: {"account": "l2", "role": "listener", "capturing": True, "since": time.time()}}
t0 = time.time()
lines_before = None
for i in range(8):
    now_i = t0 + i * 5; ms_i = int(now_i * 1000)
    who = l1 if i % 2 == 0 else l2                                # they take turns reporting
    other = l2 if who is l1 else l1
    rep = lambda: {"role": "listener", "lobby": lobby(ms_i, {C1: [person(A, since=ms_i - 1000 * i - 60_000)]})}
    hub3.ext_reports[who] = (rep(), now_i)
    hub3.ext_reports.setdefault(other, (rep(), now_i))
    S.presence_feed(hub3, who, now_i)
pfile = root10 / "presence.jsonl"
check(not pfile.exists() or not pfile.read_text().strip(), "two listeners taking turns: no cuts, nothing written")
check(len(hub3.presence.open) == 1, "one open stint")
hub4 = S.Hub("test"); hub4.presence = S.Presence()
w4 = object(); hub4.conns = {w4: {"account": "main", "role": "main", "capturing": True, "since": time.time()}}
n4 = time.time()
hub4.ext_reports = {w4: ({"role": "main", "lobby": lobby(int(n4 * 1000) - 120_000, {C1: [person(A)]})}, n4)}
S.presence_feed(hub4, w4, n4)
check(not hub4.presence.open and S.presence_source(hub4, n4) is None, "a lobby two minutes old isn't used (never stamped as now)")

# --- review: someone flagged voice-only who has spoken counts as an ordinary person
root11 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root11 / "transcripts"; S.TRANSCRIPTS.mkdir()
hub5 = S.Hub("test")
hub5.people_dir = {A: {"username": "ann", "usernames": [], "names": ["Ann"], "nicks": [], "seen": "voice"},
                   B: {"username": "bob", "usernames": [], "names": ["Bobby"], "nicks": [], "seen": "voice"}}
with (S.TRANSCRIPTS / f"{day}.jsonl").open("w") as f:
    f.write(json.dumps({"id": "y", "t": T, "end": T + 2000, "user": A, "speaker": "Ann", "text": "hi all", "ch": C1}) + "\n")
check(S.voice_only_ids(hub5) == {B}, "voice-only = flagged and no lines")
check(S.write_people_files(hub5, [A, B]) == 1, "people file for the one who spoke, none for the one only seen")
check(S.name_finder(hub5)[1].get("bobby") is None and S.name_finder(hub5)[1].get("ann") == A, "name scrubber: spoke yes, only seen no")

# --- review: pair totals at 180-day scale stay off requests and finish in reasonable time
root12 = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = root12 / "transcripts"; S.TRANSCRIPTS.mkdir()
Big = S.Presence()
import random
random.seed(1)
users = [str(200000000000000000 + i) for i in range(300)]
chans = [str(700000000000000000 + i) for i in range(20)]
recs = []
for i in range(150_000):
    s = T - random.randint(0, 179 * 86_400_000); recs.append({"id": f"b{i}", "u": random.choice(users), "g": G, "c": random.choice(chans),
                                                              "s": s, "e": s + random.randint(60_000, 3 * 3_600_000), "cut": False})
Big._set_closed(recs)
t_ = time.time(); Big.recompute(T); took = time.time() - t_
check(took < 20, f"150k stints: pair totals in {took:.1f} s (background thread)")
t_ = time.time(); Big.fresh_pairs(T); check(time.time() - t_ < 0.01, "a request never recomputes the totals")

print("\nFAILED: " + "; ".join(fails) if fails else "\nall presence checks passed")
sys.exit(1 if fails else 0)
