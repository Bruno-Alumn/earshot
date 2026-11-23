"""Unit test for the sync merge rules (server/sync.py).
    python3 test/sync_test.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "server"))
import sync  # noqa: E402

failed = False


def ok(name, cond):
    global failed
    print(("  ok  " if cond else "  FAIL ") + name)
    failed |= not cond


# keys
a = {"id": "abc", "t": 1, "end": 2, "user": "1", "text": "hi"}
legacy = {"t": 1, "end": 2, "user": "1", "text": "hi"}
tomb = {"tomb": True, "t": 5, "replaces": ["b", "a"]}
ok("id'd line keyed by id", sync.rec_key(a) == "abc")
ok("legacy line keyed by content, stable", sync.rec_key(legacy) == sync.rec_key(dict(legacy)) and sync.rec_key(legacy).startswith("k"))
ok("tomb key ignores order of replaces", sync.rec_key(tomb) == sync.rec_key({"tomb": True, "t": 5, "replaces": ["a", "b"]}))

# the same speech heard by both machines
x = {"id": "1", "t": 10000, "user": "u", "text": "Get in the car!"}
ok("same words 1 s apart: same speech", sync.same_speech(x, {"id": "2", "t": 11000, "user": "u", "text": "get in the car"}))
ok("3 s apart: different", not sync.same_speech(x, {"id": "2", "t": 13001, "user": "u", "text": "get in the car"}))
ok("another person: different", not sync.same_speech(x, {"id": "2", "t": 10000, "user": "v", "text": "get in the car"}))
ok("one is a longer cut of the other", sync.same_speech({"t": 0, "user": "u", "text": "we should push the left side"}, {"t": 500, "user": "u", "text": "we should push the left side now"}))
ok("short different words: different", not sync.same_speech({"t": 0, "user": "u", "text": "yes"}, {"t": 0, "user": "u", "text": "yes sir"}))

# deleted stretches
ok("in a deleted stretch", sync.in_ranges({"t": 50, "ch": "c"}, [(0, 100, "c")]))
ok("other channel, same time: kept", not sync.in_ranges({"t": 50, "ch": "d"}, [(0, 100, "c")]))
ok("no channel on the line: deleted", sync.in_ranges({"t": 50}, [(0, 100, "c")]))

# people
L = {"111111111111111": {"username": "old", "usernames": ["old"], "names": ["A"], "nicks": [], "first": 5}}
R = {"111111111111111": {"username": "new", "usernames": ["old", "new"], "names": ["B"], "nicks": ["n"], "first": 3},
     "222222222222222": {"username": "z", "usernames": ["z"], "names": [], "nicks": [], "first": 9, "seen": "voice"}}
P, ch = sync.merge_people(L, R)
p = P["111111111111111"]
ok("people: rename seen on the other machine wins", p["username"] == "new" and p["usernames"] == ["old", "new"])
ok("people: names unioned, local first", p["names"] == ["A", "B"] and p["nicks"] == ["n"])
ok("people: earliest first seen", p["first"] == 3)
ok("people: new person added with flags", P["222222222222222"].get("seen") == "voice" and ch == {"111111111111111", "222222222222222"})
ok("people: idempotent", sync.merge_people(P, R)[1] == set())

# names
N, n = sync.merge_names({"1": "Local"}, {"1": "Remote", "2": "New", "3": "user-1234"})
ok("names: local kept, new added, placeholders skipped", N == {"1": "Local", "2": "New"} and n == 1)

# Q&A
Q, n = sync.merge_qa({"q1": {"real": True, "answer": "gone"}}, {"q1": {"real": True, "answer": "a1"}, "q2": {"real": False, "answer": None}, "q3": {"real": True, "answer": None}},
                     alive={"q1", "q2", "a1"}, gone=set())
ok("qa: answer that still exists wins", Q["q1"]["answer"] == "a1")
ok("qa: question line missing here: skipped", "q3" not in Q and Q["q2"]["real"] is False)

# archive, three-way
ok("archive: unarchived on one side stays unarchived", sync.merge_set3({"a", "b"}, {"a"}, {"a", "b"}) == {"a"})
ok("archive: added on either side kept", sync.merge_set3({"a"}, {"a", "x"}, {"a", "y"}) == {"a", "x", "y"})
ok("archive: first sync is a union", sync.merge_set3(set(), {"a"}, {"b"}) == {"a", "b"})

# merged calls
G = sync.merge_groups3([["c1", "c2"]], [["c1", "c2"]], [])
ok("merges: broken up on the other side: broken up", G == [])
G = sync.merge_groups3([], [["c1", "c2"]], [["c2", "c3"]])
ok("merges: overlapping new groups join", G == [["c1", "c2", "c3"]])

# personality reads
per, n = sync.merge_personas({"u": {"reads": [{"at": 1}, {"at": 5}]}}, {"u": {"reads": [{"at": 3}, {"at": 5}]}, "v": {"reads": [{"at": 2}]}}, forgets=[])
ok("reads: union by time, in order", [r["at"] for r in per["u"]["reads"]] == [1, 3, 5] and "v" in per and n == 2)
per, n = sync.merge_personas({"u": {"reads": [{"at": 1}, {"at": 5}]}}, {"u": {"reads": [{"at": 6}]}}, forgets=[["u", 4, 7]])
ok("reads: a deletion's forgotten reads don't come back", [r["at"] for r in per["u"]["reads"]] == [1])

# test scores
T, C = sync.merge_tests({}, {"tests": {"u": {"O": 1, "at": 10}}, "cleared": {}}, {"tests": {}, "cleared": {"u": 20}})
ok("tests: a later clear wins", T == {} and C == {"u": 20})
T, C = sync.merge_tests({}, {"tests": {"u": {"O": 1, "at": 30}}, "cleared": {}}, {"tests": {}, "cleared": {"u": 20}})
ok("tests: a later entry wins over a clear", T["u"]["at"] == 30 and C == {})

# voice history
U = sync.presence_union([{"u": "a", "c": "c", "s": 0, "e": 100}], [{"u": "a", "c": "c", "s": 50, "e": 150, "own": True}, {"u": "a", "c": "d", "s": 0, "e": 10}])
ok("presence: overlapping stints become one span", any(r["c"] == "c" and r["s"] == 0 and r["e"] == 150 and r.get("own") for r in U) and len(U) == 2)

ok("hash ignores key order", sync.canon_hash({"a": 1, "b": 2}) == sync.canon_hash({"b": 2, "a": 1}))

print("\nFAILED" if failed else "\nPASS")
sys.exit(1 if failed else 0)
