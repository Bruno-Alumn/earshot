"""Merge rules for syncing two Earshot installs (say a Mac and a PC).

Each install transcribes its own calls with its own Discord accounts. Every few
minutes the desktop apps swap what the other is missing, and the server merges
it with the rules here. Only data is synced (transcripts, people, names,
questions and answers, merged and archived calls, personality reads, test scores,
voice history); accounts, sign-ins, keys and settings stay on each machine.

Everything here is pure (no files, no server state) so test/sync_test.py can
check every rule. server.py does the file work and keeps its caches right.

Rules, in short:
  - transcript lines: a union. A line is known by its id (older lines without one
    by their content). A line the other machine heard from the same call (the same
    person within 2 seconds, the same words) is kept once.
  - deletions: logged (sync/deleted.jsonl) and sent along, so a deleted call
    doesn't come back from the other machine.
  - people, names, Q&A, reads: unions, per person or per line.
  - merged calls, archive, test scores: three-way, against what both had after
    the last sync, so undoing something on one machine undoes it on both.
"""
from __future__ import annotations

import hashlib
import json
import re

DUP_MS = 2000                 # the same person, this close in time, the same words: one line heard twice


def content_key(r: dict) -> str:
    h = hashlib.sha1(f"{r.get('t')}|{r.get('end')}|{r.get('user')}|{r.get('text')}".encode("utf-8")).hexdigest()
    return "k" + h[:12]


def tomb_key(r: dict) -> str:
    h = hashlib.sha1((",".join(sorted(map(str, r.get("replaces") or []))) + "|" + str(r.get("t"))).encode()).hexdigest()
    return "x" + h[:12]


def rec_key(r: dict) -> str:
    """What identifies a transcript record on both machines."""
    if r.get("tomb"):
        return tomb_key(r)
    return str(r["id"]) if r.get("id") else content_key(r)


_norm_re = re.compile(r"[^\w]+", re.UNICODE)


def norm_text(s: str) -> str:
    return _norm_re.sub(" ", (s or "").lower()).strip()


def same_speech(a: dict, b: dict) -> bool:
    """Two installs in the same channel transcribe the same words with different ids."""
    if a.get("tomb") or b.get("tomb"):
        return False
    if not a.get("user") or a.get("user") != b.get("user"):
        return False
    if abs(int(a.get("t", 0)) - int(b.get("t", 0))) > DUP_MS:
        return False
    x, y = norm_text(a.get("text", "")), norm_text(b.get("text", ""))
    if not x or not y:
        return False
    return x == y or (min(len(x), len(y)) >= 12 and (x in y or y in x))


def day_hash(keys) -> str:
    return hashlib.sha1("\n".join(sorted(set(keys))).encode()).hexdigest()


def canon_hash(obj) -> str:
    """A hash that ignores key order, so two machines holding the same data agree."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def in_ranges(r: dict, ranges: list, slack: int = 0) -> bool:
    """Is a line inside a deleted stretch [start, end, channel] (as delete_group cut it)?"""
    t, ch = r.get("t"), r.get("ch")
    if not isinstance(t, (int, float)):
        return False
    for s, e, c in ranges:
        if s - slack <= t <= e + slack and (not c or not ch or c == ch):
            return True
    return False


# ------------------------------------------------------------------ people.json
def merge_people(local: dict, remote: dict) -> tuple[dict, set]:
    """Per person: every name either machine saw. Returns (merged, changed uids)."""
    out = {u: dict(p, usernames=list(p.get("usernames") or []), names=list(p.get("names") or []),
                   nicks=list(p.get("nicks") or [])) for u, p in local.items() if isinstance(p, dict)}
    changed = set()
    for uid, rp in (remote or {}).items():
        if not isinstance(rp, dict) or not re.fullmatch(r"\d{15,22}", str(uid)):
            continue
        lp = out.get(uid)
        if lp is None:
            out[uid] = {"username": rp.get("username"), "usernames": list(rp.get("usernames") or []),
                        "names": list(rp.get("names") or []), "nicks": list(rp.get("nicks") or []),
                        "first": rp.get("first"), **({"seen": rp["seen"]} if rp.get("seen") else {})}
            changed.add(uid)
            continue
        before = json.dumps(lp, sort_keys=True)
        for k in ("usernames", "names", "nicks"):
            for v in rp.get(k) or []:
                if v and v not in lp[k]:
                    lp[k].append(v)
        lu, ru = lp.get("username"), rp.get("username")
        if ru and lu != ru:
            if not lu or (lu in (rp.get("usernames") or []) and ru not in (local.get(uid, {}).get("usernames") or [])):
                lp["username"] = ru                      # the other machine saw a later rename
            if ru not in lp["usernames"]:
                lp["usernames"].append(ru)
        firsts = [x for x in (lp.get("first"), rp.get("first")) if isinstance(x, (int, float))]
        lp["first"] = min(firsts) if firsts else None
        if lp.get("seen") == "voice" and rp.get("seen") != "voice":
            lp.pop("seen", None)                         # heard talking on the other machine
        if json.dumps(lp, sort_keys=True) != before:
            changed.add(uid)
    return out, changed


# ------------------------------------------------------------------ names.json
def merge_names(local: dict, remote: dict) -> tuple[dict, int]:
    """New ids from the other machine; a name this machine already has stays."""
    out, n = dict(local), 0
    for uid, name in (remote or {}).items():
        if uid not in out and isinstance(name, str) and name and not re.fullmatch(r"user-\d{4}", name):
            out[uid] = name; n += 1
    return out, n


# ------------------------------------------------------------------ <day>.qa.json
def merge_qa(local: dict, remote: dict, alive: set, gone: set) -> tuple[dict, int]:
    """Per question line: keep the judgement whose answer still exists; this machine's on a tie."""
    out, n = dict(local), 0
    for qid, rv in (remote or {}).items():
        if qid in gone or qid not in alive or not isinstance(rv, dict):
            continue
        lv = out.get(qid)
        if lv is None or (lv.get("answer") and lv["answer"] not in alive and (not rv.get("answer") or rv["answer"] in alive)):
            if rv.get("answer") and rv["answer"] not in alive:
                continue
            out[qid] = {"real": bool(rv.get("real")), "answer": rv.get("answer")}; n += 1
    for qid in [q for q in out if q in gone]:
        out.pop(qid); n += 1
    return out, n


# ------------------------------------------------------------------ three-way sets (archive.json)
def merge_set3(base, local, remote) -> set:
    """Kept on both, or added on one since the last sync. Removed on one since: removed."""
    base, local, remote = set(base or ()), set(local or ()), set(remote or ())
    return (local & remote) | (local - base) | (remote - base)


# ------------------------------------------------------------------ merges.json groups
def merge_groups3(base: list, local: list, remote: list) -> list:
    """Merged calls: groups of call ids. A group either side made since the last sync
    is kept; one either side broke up since is dropped; overlapping groups join."""
    fz = lambda gs: {frozenset(g) for g in gs or [] if len(set(g)) >= 2}
    b, l, r = fz(base), fz(local), fz(remote)
    keep = (l & r) | (l - b) | (r - b)                   # a group broken up on one side since is in neither
    # join groups that share a call (connected components)
    groups: list[set] = []
    for g in keep:
        g = set(g)
        hit = [x for x in groups if x & g]
        for x in hit:
            g |= x; groups.remove(x)
        groups.append(g)
    return sorted((sorted(g) for g in groups if len(g) >= 2), key=lambda g: g[0])


# ------------------------------------------------------------------ personas.json
def merge_personas(local: dict, remote: dict, forgets: list) -> tuple[dict, int]:
    """Reads per person, from both machines, oldest first. Reads that a deletion
    since took away (forgets: [uid, from ms, at ms]) are not brought back."""
    out = {u: {**b, "reads": list(b.get("reads") or [])} for u, b in local.items() if isinstance(b, dict)}
    n = 0
    dead = {}
    for f in forgets or []:
        try:
            uid, frm, at = str(f[0]), int(f[1]), int(f[2])
        except (TypeError, ValueError, IndexError):
            continue
        dead.setdefault(uid, []).append((frm, at))
    def killed(uid, at):
        return any(frm <= at <= when for frm, when in dead.get(uid, []))
    for uid, rb in (remote or {}).items():
        if not isinstance(rb, dict):
            continue
        have = out.setdefault(uid, {"reads": []})
        ats = {x.get("at") for x in have["reads"]}
        for rd in rb.get("reads") or []:
            if not isinstance(rd, dict) or not isinstance(rd.get("at"), (int, float)) or rd["at"] in ats:
                continue
            if killed(uid, rd["at"]):
                continue
            have["reads"].append(rd); ats.add(rd["at"]); n += 1
        have["reads"].sort(key=lambda x: x.get("at", 0))
    for uid in list(out):
        before = len(out[uid]["reads"])
        out[uid]["reads"] = [x for x in out[uid]["reads"] if not killed(uid, x.get("at", 0))]
        n += before - len(out[uid]["reads"])
        if not out[uid]["reads"]:
            out.pop(uid)
    return out, n


# ------------------------------------------------------------------ study.json tests
def merge_tests(base: dict, local: dict, remote: dict) -> tuple[dict, dict]:
    """Entered test scores: the newest entry or clear wins.
    Each side is {"tests": {uid: {..., at}}, "cleared": {uid: at}}."""
    tests, cleared = {}, {}
    for side in (local, remote):
        for uid, at in (side.get("cleared") or {}).items():
            if isinstance(at, (int, float)) and at > cleared.get(uid, 0):
                cleared[uid] = at
        for uid, t in (side.get("tests") or {}).items():
            if isinstance(t, dict) and t.get("at", 0) >= tests.get(uid, {}).get("at", -1):
                tests[uid] = t
    for uid in list(tests):
        if cleared.get(uid, 0) >= tests[uid].get("at", 0):
            tests.pop(uid)
    return tests, {u: at for u, at in cleared.items() if u not in tests}


# ------------------------------------------------------------------ presence (voice history)
def presence_union(local: list, peer: list) -> list:
    """Both machines' voice stints, a stint both saw counted once: overlapping
    stints of the same person in the same channel become one span."""
    by: dict = {}
    for r in list(local) + list(peer):
        by.setdefault((r["u"], r["c"]), []).append(r)
    out = []
    for lst in by.values():
        lst.sort(key=lambda r: r["s"])
        cur = None
        for r in lst:
            if cur and r["s"] <= cur["e"]:
                if r["e"] > cur["e"]:
                    cur = {**cur, "e": r["e"], "cut": r.get("cut", False)}
                cur["own"] = cur.get("own") or r.get("own")
                cur["rec"] = cur.get("rec") or r.get("rec")
            else:
                if cur:
                    out.append(cur)
                cur = dict(r)
        if cur:
            out.append(cur)
    out.sort(key=lambda r: (r["e"], r["s"]))
    return out
