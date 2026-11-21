"""Own-voice detection test: a video (another speaker) playing near one person's
mic must be flagged "background"; the person's own speech must not be.

Builds one user's utterance stream from one speaker's clips with another
speaker's clips mixed in as separate utterances, in three ways:
  person-first   the person talks first, the video joins later
  video-first    the video is the first thing we hear (enrolment must not take it)
  room-video     the video through room echo at a third of the level, with noise
and reports false-flag rate (person flagged) and miss rate (video not flagged),
plus the embedding time per utterance. Then checks the server hooks: a flagged
line is written with "bg": true, and the 2B rewrite keeps the flag.

Data: AMI meeting audio (a parquet with audio{bytes} and AMI ids, which name the
speaker) when given with --ami; otherwise the fake call voices in test/site
(fewer, shorter clips: a smoke test).

Run from server/:  .venv/bin/python ../test/voiceprint_test.py [--ami path/to/0.parquet]
"""

import argparse
import collections
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "server"))
import voiceprint as V  # noqa: E402

SR = 16000
rng = np.random.default_rng(7)


def pieces(clips, n, lo=1.0, hi=7.0):
    """n utterances of random length cut from a speaker's clips."""
    out = []
    while len(out) < n:
        a = clips[rng.integers(len(clips))]
        L = int(rng.uniform(lo, hi) * SR)
        if len(a) <= L:
            out.append(a.copy()); continue
        # From the start of the clip, where the speaker starts talking (as our
        # segmenter cuts utterances), or from a random point, half the time each.
        s = 0 if rng.random() < 0.5 else rng.integers(0, len(a) - L)
        out.append(a[s:s + L].copy())
    return out


def room(a):
    """As heard from loudspeakers across a room: reverberant (a synthetic 0.4 s
    decay), a third of the level, with a little noise."""
    n = int(0.4 * SR)
    rir = rng.standard_normal(n) * np.exp(-6.9 * np.arange(n) / n)    # -60 dB over 0.4 s
    rir[0] = 4.0                                                         # the direct sound
    wet = np.convolve(a, rir)[: len(a)]
    wet *= 0.33 * np.abs(a).max() / (np.abs(wet).max() + 1e-9)
    return (wet + rng.standard_normal(len(a)) * 0.003).astype(np.float32)


def run(person, video, scenario, n_person=40, n_video=15):
    """Feeds one user's stream through Voiceprints; returns counts."""
    vp = V.Voiceprints(EMB)
    you, vid = pieces(person, n_person), pieces(video, n_video)
    if scenario == "room-video":
        vid = [room(v) for v in vid]
    if scenario == "video-first":
        # Four video utterances before the person says anything, the rest mixed in.
        order = [("v", x) for x in vid[:4]]
        rest = [("p", x) for x in you] + [("v", x) for x in vid[4:]]
    else:
        # Five of the person first, then the video mixed in at random.
        order = [("p", x) for x in you[:5]]
        rest = [("p", x) for x in you[5:]] + [("v", x) for x in vid]
    rest = [rest[i] for i in rng.permutation(len(rest))]
    c = collections.Counter()
    for who, a in order + rest:
        enrolled = vp.people.get("u", V.Person()).print is not None
        bg = vp.check("u", a)
        c[who, bg, enrolled] += 1
    return c, vp


def video_later(person, video, n_video, n_before=25):
    """The person talks (n_before x 3 s), a video plays (n_video x 4 s), the person
    talks again (15 x 3 s). Returns (video lines flagged, later person lines flagged)."""
    vp = V.Voiceprints(EMB)
    before, after = pieces(person, n_before, 3.0, 3.0), pieces(person, 15, 3.0, 3.0)
    vid = pieces(video, n_video, 4.0, 4.0)
    for a in before:
        vp.check("u", a)
    v_flag = sum(vp.check("u", a) for a in vid)
    p_flag = sum(vp.check("u", a) for a in after)
    return v_flag, p_flag


def lock_in():
    """Synthetic embeddings (same voice cosine about 0.72, different about 0): a
    TV with one main voice (40 x 3 s) and 7 others (6 x 3 s each) fills every
    voice slot before the person talks (200 x 2.5 s). The person's new voice must
    not be evicted on arrival (it would never grow and stay flagged forever)."""
    r = np.random.default_rng(3)
    d = 192
    base = [x / np.linalg.norm(x) for x in r.standard_normal((9, d))]
    def utt(i):
        u = base[i] + r.standard_normal(d) * 0.62 / np.sqrt(d)
        return u / np.linalg.norm(u)
    p = V.Person()
    for _ in range(40):
        p.judge(utt(0), 3.0)
    for i in range(1, 8):
        for _ in range(6):
            p.judge(utt(i), 3.0)
    flags = [p.judge(utt(8), 2.5)[0] for _ in range(200)]
    first_ok = next((i for i, f in enumerate(flags) if not f), None)
    still = sum(flags[100:])
    return sum(flags), first_ok, still, p.holder is not None and float(p.holder[0] @ base[8]) > 0.5


def rates(c):
    p_flag = sum(v for (w, bg, _), v in c.items() if w == "p" and bg)
    p_all = sum(v for (w, _, _), v in c.items() if w == "p")
    v_miss = sum(v for (w, bg, _), v in c.items() if w == "v" and not bg)
    v_all = sum(v for (w, _, _), v in c.items() if w == "v")
    v_miss_after = sum(v for (w, bg, e), v in c.items() if w == "v" and not bg and e)
    v_after = sum(v for (w, _, e), v in c.items() if w == "v" and e)
    return p_flag, p_all, v_miss, v_all, v_miss_after, v_after


def load_ami(path):
    import pyarrow.parquet as pq
    by, meeting = collections.defaultdict(list), {}
    for r in pq.read_table(path, columns=["audio", "id"]).to_pylist():
        _, m, _, who = r["id"].split("_")[:4]                   # AMI_<meeting>_<headset>_<speaker>_...
        by[who].append(sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")[0])
        meeting[who] = m[:6]
    return {k: v for k, v in by.items() if len(v) >= 12}, meeting


def load_site():
    w = lambda n: sf.read(HERE / "site" / f"{n}.wav", dtype="float32")[0]
    # b and b2 are the same voice; a is a different one.
    return {"site-b": [w("b"), w("b2")], "site-a": [w("a")]}


def check_server_hooks():
    """Writer.write stores bg; Writer.replace (the 2B rewrite) keeps it."""
    import server as S
    tmp = Path(tempfile.mkdtemp())
    S.TRANSCRIPTS = tmp
    class Hub:
        people, busy_t0, refine_q = {}, None, ()
        people_dirty = set()
        def publish(self, msg): pass
        def username_of(self, uid): return None
    hub = Hub()
    w = S.Writer(hub)
    t = time.time() * 1000
    hub.busy_t0 = t - 1000          # earlier speech still being transcribed: the .md waits
    a = w.write(t, 2.0, "u1", "Ann", "Something from a video.", None, bg=True)
    b = w.write(t + 3000, 2.0, "u1", "Ann", "Something Ann said.", None)
    w.replace([a], [(a["t"], a["end"], "Something from a video, rewritten.")])
    hub.busy_t0 = None
    w.flush_md(force=True)
    day = S.load_day(time.strftime("%Y-%m-%d", time.localtime(t / 1000)))
    got = {r["text"]: bool(r.get("bg")) for r in day}
    assert got == {"Something from a video, rewritten.": True, "Something Ann said.": False}, got
    # The readable .md marks it too (the rewrite replaced it before it was written).
    md = next(tmp.glob("*.md")).read_text(encoding="utf-8")
    assert f"Ann:** {S.BG_MARK}Something from a video, rewritten." in md and "Ann:** Something Ann said." in md, md
    assert "Something from a video.\n" not in md, md
    # Call stats: the video's words aren't Ann's; a call of only background has nobody in it.
    c = S.build_calls("d", day)[0]
    assert (c["lines"], c["total"], c["words"], c["preview"]) == (1, 2, 3, "Something Ann said."), c
    only = S.build_calls("d", [r for r in day if r.get("bg")])[0]
    assert only["people"] == [] and only["lines"] == 0 and only["total"] == 1 and only["preview"] == "", only
    print("server hooks: bg written, kept through the 2B rewrite, marked in the .md, left out of call stats: ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ami", help="AMI parquet (audio{bytes}, id)")
    ap.add_argument("--pairs", type=int, default=8, help="speaker pairs to try (AMI, up to 16)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    global EMB, rng
    rng = np.random.default_rng(args.seed)
    EMB = V.Embedder(); EMB.load(wait=True)
    if EMB.sess is None:
        sys.exit("model failed to load: " + str(EMB.error))

    if args.ami:
        spk, meeting = load_ami(args.ami)
        names = sorted(spk)
        # Each person with a "video" voice from another meeting (no shared crosstalk).
        pairs = []
        for i in range(0, len(names), max(1, len(names) // args.pairs)):
            v = next(names[(i + k) % len(names)] for k in range(5, 5 + len(names)) if meeting[names[(i + k) % len(names)]] != meeting[names[i]])
            pairs.append((names[i], v))
        pairs = pairs[:args.pairs]
        n_person, n_video = 40, 15
    else:
        spk = load_site()
        pairs = [("site-b", "site-a")]
        n_person, n_video = 20, 8
    print(f"{'person':>10} {'video':>10} {'scenario':>12} | false flags | missed video (after enrolment)")
    tot = collections.Counter()
    fails = []
    for p, v in pairs:
        for sc in ("person-first", "video-first", "room-video"):
            c, vp = run(spk[p], spk[v], sc, n_person, n_video)
            pf, pa, vm, va, vma, vaa = rates(c)
            tot[sc, "pf"] += pf; tot[sc, "pa"] += pa; tot[sc, "vm"] += vm; tot[sc, "va"] += va
            tot[sc, "vma"] += vma; tot[sc, "vaa"] += vaa
            person = vp.people.get("u")
            ok_enrol = person is not None and person.print is not None
            print(f"{p:>10} {v:>10} {sc:>12} | {pf:2d}/{pa:2d}      | {vm:2d}/{va:2d} ({vma}/{vaa})"
                  + ("" if ok_enrol else "  NOT ENROLLED"))
            tot["runs"] += 1; tot["unenrolled"] += not ok_enrol
    print()
    # Never enrolled = nothing flagged (safe, but the video is missed); allowed rarely.
    if tot["unenrolled"] * 5 > tot["runs"]:
        fails.append(f"{tot['unenrolled']} of {tot['runs']} streams never enrolled")
    for sc in ("person-first", "video-first", "room-video"):
        pf = tot[sc, "pf"] / max(1, tot[sc, "pa"]); vm = tot[sc, "vm"] / max(1, tot[sc, "va"])
        vma = tot[sc, "vma"] / max(1, tot[sc, "vaa"])
        print(f"{sc:>12}: false-flag rate {100*pf:4.1f}%   miss rate {100*vm:4.1f}% (after enrolment {100*vma:4.1f}%)")
        # Video first: until the person has talked more than the video, we can't
        # know which voice is theirs, so a few of their early lines may be flagged.
        # The video must actually be judged and flagged (0 of 0 would pass the rates).
        if tot[sc, "vaa"] == 0 or tot[sc, "vaa"] - tot[sc, "vma"] == 0:
            fails.append(f"{sc}: no video utterance was flagged after enrolment")
        limit = 0.10 if sc == "video-first" else 0.05
        if pf > limit:
            fails.append(f"{sc}: false-flag rate {100*pf:.1f}% > {100*limit:.0f}%")
        if vma > 0.2:
            fails.append(f"{sc}: miss rate after enrolment {100*vma:.1f}% > 20%")
    # A video playing for a while after the person has been enrolled: their own
    # later lines must not be flagged (the case that used to lose them).
    for p, v in pairs[:3]:
        for nb, n in ((25, 20), (25, 40), (25, 80), (12, 80)):
            vf, pf = video_later(spk[p], spk[v], n, nb)
            print(f"video later ({p}: {nb} x 3 s, then video {n} x 4 s): video lines flagged {vf}/{n}, person's later lines flagged {pf}/15")
            if pf > 1:
                fails.append(f"video later ({p}, {nb}, {n}): {pf} of the person's later lines flagged")
            if nb == 25 and n == 20 and vf == 0:
                fails.append("video later: no video line flagged")
    total, first_ok, still, owns = lock_in()
    print(f"lock-in (8 voices from a TV first): person flagged {total}/200, first unflagged at #{first_ok}, "
          f"{still} of the last 100 flagged, person holds the voiceprint at the end: {owns}")
    if still or not owns:
        fails.append("lock-in: the person never got their voiceprint back")
    ms = []
    for a in pieces(next(iter(spk.values())), 30, 2.0, 8.0):
        t = time.perf_counter(); EMB(a); ms.append(((time.perf_counter() - t) * 1000, len(a) / SR))
    per_s = np.median([m / s for m, s in ms])
    print(f"speed: median {np.median([m for m, _ in ms]):.0f} ms per utterance (2-8 s), {per_s:.1f} ms per second of audio")

    check_server_hooks()
    if fails:
        sys.exit("FAIL: " + "; ".join(fails))
    print("PASS")


if __name__ == "__main__":
    main()
