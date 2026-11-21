"""Local transcription server for Earshot.

The desktop app sends each call participant's audio as its own stream
(16 kHz mono int16, tagged with their Discord user id and name). Per person,
we cut the stream into utterances on silence, transcribe each utterance
locally with an MLX speech model, and append "[time] Name: text" to
transcripts/YYYY-MM-DD.md.

Run:  uv run server.py            (see --help for model choices)
"""

import argparse
import sys
import asyncio
import base64
import datetime as dt
import gc
import json
import os
import queue
import shutil
import signal
import subprocess
import re
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Response

SR = 16000
ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "transcripts"
DASHBOARD = Path(__file__).resolve().parent / "dashboard"

HALLUCINATIONS = {
    "thank you.", "thank you", "thanks for watching!", "thanks for watching.",
    "you", "bye.", "bye", ".", "thank you very much.", "subtitles by the amara.org community",
}


FILLER = re.compile(r"(?i)(^|[\s,])(?:um+|uh+|uhm|erm|hmm+|mm+)(?=[\s,.!?]|$)[,]?")


def tidy(text: str) -> str:
    """Readability pass: drop um/uh fillers, capitalise 'i' and sentence starts."""
    t = FILLER.sub(r"\1", text)
    t = re.sub(r"\s{2,}", " ", t).strip(" ,")
    t = re.sub(r"\bi\b", "I", t)
    t = re.sub(r"\bi'(m|ll|ve|d)\b", lambda m: "I'" + m.group(1), t)
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return t


# ---------------------------------------------------------------- voice detection
# Loudness alone can't tell a voice from keyboard clicks, a fan, breathing or
# music, and every noise burst that starts an utterance comes back as a junk
# line ("I.", "Oh."). Silero VAD (v5: a 2 MB ONNX model, CPU, ONNX Runtime)
# scores each 32 ms of audio with a speech probability. The model is
# recurrent, so each person has their own VadStream (state + leftover
# samples); the model itself is loaded once, in the background at startup,
# and shared. Until it is ready, or if it can't be loaded, Segmenter falls
# back to the loudness gate alone.
VAD_REPO, VAD_FILE = "onnx-community/silero-vad", "onnx/model.onnx"
VAD_WINDOW, VAD_CONTEXT = 512, 64   # samples per score (32 ms at 16 kHz); samples of the previous window in front
# Loudness bars, as multiples of the speech threshold (which sensitivity sets).
# The VAD tells speech from noise, not this person from a quiet voice in their
# room (TV, a roommate, other people leaking back through their speakers), so
# an utterance still has to start as loud as the threshold; once it is going,
# quieter audio may continue it (soft words, trailing off), but only down to
# VAD_REL of the utterance's own level.
VAD_START = 1.0                     # x threshold: an utterance starts only above this (and with speech)
VAD_FLOOR = 0.25                    # x threshold: below this nothing continues an utterance
VAD_WARM = 0.5                      # x threshold: between utterances, quieter than this isn't scored
VAD_REL = 0.15                      # x the utterance's level (-16 dB): quieter audio doesn't continue it
VAD_MIN_VOICED_MS = 160             # an utterance needs this much speech-scored audio to be transcribed (see flush)


class Vad:
    model = None                    # the shared scoring function once loaded
    state = "off"                   # off -> loading -> ready | failed

    @classmethod
    def load(cls):
        cls.state = "loading"
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(VAD_REPO, VAD_FILE, local_files_only=True)   # cached: no network
            except Exception:
                path = hf_hub_download(VAD_REPO, VAD_FILE)
            opts = ort.SessionOptions()
            # One thread: a 32 ms window is tiny, and threads would cost more than the math.
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
            sr = np.array(SR, dtype=np.int64)

            def score(x, state):
                out, state = sess.run(None, {"input": x[None, :], "state": state, "sr": sr})
                return float(out[0, 0]), state
            score(np.zeros(VAD_CONTEXT + VAD_WINDOW, np.float32), np.zeros((2, 1, 128), np.float32))
            cls.model, cls.state = score, "ready"
            print("  Speech detection: Silero VAD ready", flush=True)
        except Exception as e:
            cls.state = "failed"
            print(f"  Speech detection: Silero VAD unavailable ({type(e).__name__}: {e}); using loudness only", flush=True)

    @classmethod
    def load_async(cls):
        if cls.state == "off":
            cls.state = "loading"
            threading.Thread(target=cls.load, daemon=True, name="vad-load").start()


class VadStream:
    """One person's VAD: feed chunks of any length, get a probability per 32 ms."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), np.float32)
        self.ctx = np.zeros(VAD_CONTEXT, np.float32)
        self.rest = np.zeros(0, np.float32)      # samples short of a full window, kept for next time

    def feed(self, score, pcm: np.ndarray) -> list[float]:
        x = np.concatenate([self.rest, pcm]) if len(self.rest) else pcm
        n = len(x) // VAD_WINDOW
        probs = []
        for i in range(n):
            w = x[i * VAD_WINDOW:(i + 1) * VAD_WINDOW]
            p, self.state = score(np.concatenate([self.ctx, w]), self.state)
            self.ctx = w[-VAD_CONTEXT:]
            probs.append(p)
        self.rest = x[n * VAD_WINDOW:]
        return probs


# ---------------------------------------------------------------- segmenter
class Segmenter:
    """One person's stream -> utterances. Loudness is a cheap first gate; what
    passes it is scored by the VAD (once loaded), which decides speech."""

    def __init__(self, out: queue.Queue, threshold: float, label: str, uid: str = "",
                 silence_ms=700, max_ms=25000, preroll_ms=300, vad_threshold: float | None = 0.5):
        self.out, self.threshold, self.label, self.uid = out, threshold, label, uid
        self.vad_threshold = vad_threshold        # None: loudness only
        self.channel: dict | None = None
        self.silence_ms, self.max_ms, self.preroll_ms = silence_ms, max_ms, preroll_ms
        self.pre: deque[tuple[float, np.ndarray]] = deque()
        self.cur: list[tuple[float, np.ndarray]] = []
        self.silent_ms = 0.0
        self.noise = threshold
        self.last_seen = time.time()
        self.last_end = 0.0          # audio-clock end of the previous chunk (ms)
        self.vad = VadStream()
        self.voiced_ms = 0.0         # VAD-scored speech in the current utterance
        self.cur_vad = False         # the current utterance was started by the VAD
        self.level = 0.0             # typical loudness of the current utterance's speech
        self.carry = False           # the last utterance was cut at max_ms: what follows continues it
        self.cur_carry = False       # the current utterance is such a continuation
        self.dropped = 0             # utterances the VAD found too little speech in

    def push(self, t: float, pcm: np.ndarray):
        self.last_seen = time.time()
        dur = len(pcm) / SR * 1000
        # The browser only sends speech, so a jump in the audio timeline is
        # silence that was skipped: count it, and end the utterance if it's long.
        gap = t - self.last_end if self.last_end else 0
        self.last_end = t + dur
        if gap > 300:
            self.vad.reset()                     # unrelated audio follows: start the VAD fresh
            self.pre.clear()                     # and the sound before the gap is no lead-in to it
            if self.cur or self.carry:
                self.silent_ms += gap
                if self.silent_ms >= self.silence_ms:
                    self.flush()
                    self.carry = False           # (a pause this long ends a carry too)
        rms = float(np.sqrt(np.mean(pcm * pcm))) if len(pcm) else 0.0
        score = Vad.model if self.vad_threshold is not None else None
        voiced = 0
        going = bool(self.cur) or self.carry
        if score is None:
            speech = rms > max(self.threshold, self.noise * 3)
        elif rms > self.threshold * VAD_WARM or going:
            # Near a voice's loudness (or mid-utterance, where the VAD's
            # memory of the voice matters): ask the model. Silero is slow to
            # rise at onsets, so once speaking, a lower bar keeps it going.
            probs = self.vad.feed(score, pcm)
            voiced = sum(p > self.vad_threshold for p in probs)
            stay = self.vad_threshold - 0.15 if going else self.vad_threshold
            bar = max(self.threshold * VAD_FLOOR, self.level * VAD_REL) if going else self.threshold * VAD_START
            speech = rms > bar and any(p > stay for p in probs)
            if speech:
                self.level = 0.8 * self.level + 0.2 * rms if going else rms
        else:
            self.vad.reset()                     # quiet, between utterances: nothing worth remembering
            speech = False
        if not speech:
            self.noise = 0.98 * self.noise + 0.02 * rms   # (only the loudness-only gate uses it)
        if self.cur:
            self.cur.append((t, pcm))
            self.voiced_ms += voiced * VAD_WINDOW / SR * 1000
            self.silent_ms = 0 if speech else self.silent_ms + dur
            if self.silent_ms >= self.silence_ms:
                self.flush()
            elif self.cur[-1][0] + dur - self.cur[0][0] >= self.max_ms:
                self.flush()
                self.carry = True                # mid-speech cut: the rest is the same person still talking
        elif speech:
            # the lead-in: only the sound just before this, never something left from long ago
            self.cur = [(pt, pp) for pt, pp in self.pre if t - pt <= self.preroll_ms + 50] + [(t, pcm)]
            self.pre.clear()
            self.silent_ms = 0
            self.voiced_ms = voiced * VAD_WINDOW / SR * 1000
            self.cur_vad = score is not None
            self.cur_carry, self.carry = self.carry, False
        else:
            if self.carry:                       # a pause after the cut: still the same talk until it's a long one
                self.silent_ms += dur
                self.carry = self.silent_ms < self.silence_ms
            self.pre.append((t, pcm))
            while self.pre and t - self.pre[0][0] > self.preroll_ms:
                self.pre.popleft()

    def flush(self):
        if not self.cur:
            return
        t0 = self.cur[0][0]
        audio = np.concatenate([p for _, p in self.cur])
        # (sample offset in `audio`, clock ms) per chunk: skipped silence means
        # audio time != clock time, and lines are split back onto the clock.
        marks, off = [], 0
        for t, p in self.cur:
            marks.append((off, t)); off += len(p)
        self.cur, self.silent_ms = [], 0
        self.carry = False
        # A quick "no" or "yes" scores about 250 ms of speech; a cough, a click
        # or a breath that fooled the VAD for a window or two doesn't. (The
        # tail of a talk cut at max_ms is exempt: it can be one short word.)
        if self.cur_vad and not self.cur_carry and self.voiced_ms < VAD_MIN_VOICED_MS:
            self.dropped += 1
            print(f"  Speech detection: dropped a {len(audio) / SR:.1f} s sound with only "
                  f"{self.voiced_ms:.0f} ms of speech ({self.dropped} so far for this person)", flush=True)
            return
        if self.cur_vad and self.cur_carry and len(audio) < SR * 0.4:
            # The last word of a talk cut at max_ms can be a single chunk:
            # pad it with silence rather than lose it to the length minimum.
            audio = np.pad(audio, (0, int(SR * 0.4) - len(audio)))
        if len(audio) >= SR * 0.4:
            self.out.put((t0, audio, self.label, self.uid, self.channel, marks))


# ---------------------------------------------------------------- output
class Writer:
    """Appends each line to the readable .md transcript and to a .jsonl data
    file (used by the dashboard's analytics), then pushes it to the dashboard."""

    def __init__(self, hub: "Hub"):
        self.hub = hub
        self.q = None                    # the utterance queue (set by main)
        self.pending: list = []          # .md lines waiting for anything said earlier
        self.lock = threading.Lock()

    def _paths(self, when):
        TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
        md = TRANSCRIPTS / f"{when:%Y-%m-%d}.md"
        if not md.exists():
            md.write_text(f"# Voice call transcript, {when:%A %B} {when.day}, {when.year}\n\n", encoding="utf-8")
        return md, TRANSCRIPTS / f"{when:%Y-%m-%d}.jsonl"

    def write(self, t_ms: float, dur_s: float, uid: str, speaker: str, text: str, ch: dict | None = None, bg: bool = False):
        when = dt.datetime.fromtimestamp(t_ms / 1000)
        md, jl = self._paths(when)
        rec = {"id": os.urandom(5).hex(), "t": int(t_ms), "end": int(t_ms + dur_s * 1000), "user": uid, "speaker": speaker, "text": text}
        un = self.hub.username_of(uid)
        if un:
            rec["username"] = un         # their Discord username: the same whatever name they show
        if uid:
            self.hub.people_dirty.add(uid)
        if bg:
            rec["bg"] = True             # not the person's own voice (a video, someone in their room)
        with self.lock:
            self.pending.append((t_ms, when, speaker, text, (ch or {}).get("name"), time.time(), rec["id"], bg))
        self.flush_md()
        if ch and ch.get("id"):
            rec.update({"ch": ch["id"], "chName": ch.get("name"), "guild": ch.get("guild"), "guildName": ch.get("guildName")})
        with jl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{when:%H:%M:%S}] {speaker}: {text}", flush=True)
        self.hub.publish({"type": "line", **rec})
        return rec

    def replace(self, old: list[dict], parts: list[tuple[float, float, str]]):
        """The 2B pass rewrote an utterance: new lines replace the live ones, in
        the data file, the dashboard and (if not written yet) the .md file."""
        if not old or not parts:
            return
        base = old[0]
        ids = [r["id"] for r in old]
        # Deleted (or archived away) since? Then there is nothing to rewrite.
        days = {dt.datetime.fromtimestamp(r["t"] / 1000).strftime("%Y-%m-%d") for r in old}
        alive = {r.get("id") for d in days for r in load_day(d)}
        if not all(i in alive for i in ids):
            return
        new = []
        for i, (t_ms, end_ms, text) in enumerate(parts):
            rec = {"id": os.urandom(5).hex(), "t": int(t_ms), "end": int(max(end_ms, t_ms + 300)), "user": base["user"],
                   "speaker": base["speaker"], "text": text, "refined": True}
            for k in ("ch", "chName", "guild", "guildName", "bg", "username"):   # the 2B rewrite keeps the background flag
                if k in base:
                    rec[k] = base[k]
            new.append(rec)
        # A note in every day file that held an old line (a call can cross
        # midnight), then each new line in its own day's file.
        for d in sorted(days):
            _, jl = self._paths(dt.datetime.strptime(d, "%Y-%m-%d"))
            with jl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"tomb": True, "t": int(min(r["t"] for r in old)), "replaces": ids}) + "\n")
        for rec in new:
            _, jl = self._paths(dt.datetime.fromtimestamp(rec["t"] / 1000))
            with jl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with self.lock:
            if all(any(x[6] == i for x in self.pending) for i in ids):      # not in the .md yet: swap them
                self.pending = [x for x in self.pending if x[6] not in ids]
                for rec in new:
                    w = dt.datetime.fromtimestamp(rec["t"] / 1000)
                    self.pending.append((rec["t"], w, rec["speaker"], rec["text"], rec.get("chName"), time.time(), rec["id"], bool(rec.get("bg"))))
        self.hub.publish({"type": "replace", "ids": ids, "lines": new})
        self.flush_md()

    def watermark(self) -> float:
        """Clock time (ms) of the earliest speech that could still produce a line:
        someone mid-sentence, an utterance waiting in the queue, or the one being
        transcribed now. Everything said before it is final."""
        marks = []
        try:
            marks += [p.cur[0][0] for p in list(self.hub.people.values()) if p.cur]
        except Exception:
            pass
        try:
            marks += [item[0] for item in list(self.q.queue)] if self.q else []
        except Exception:
            pass
        if getattr(self.hub, "busy_t0", None) is not None:
            marks.append(self.hub.busy_t0)
        try:      # lines waiting for the 2B rewrite (gives up on ones older than a minute)
            marks += [it["t0"] for it in list(getattr(self.hub, "refine_q", ())) if time.time() - it["queued"] < 60]
        except Exception:
            pass
        return min(marks) if marks else float("inf")

    def flush_md(self, force: bool = False):
        """Write .md lines in the order they were said, not the order they were
        transcribed (a long sentence finishes after a short reply that came later)."""
        with self.lock:
            if not self.pending:
                return
            wm, now = self.watermark(), time.time()
            self.pending.sort(key=lambda x: x[0])
            ready = [x for x in self.pending if force or x[0] < wm or now - x[5] > 90]
            if not ready:
                return
            cut = max(i for i, x in enumerate(self.pending) if x in ready) + 1
            ready, self.pending = self.pending[:cut], self.pending[cut:]
        for t_ms, when, speaker, text, chname, _, _, bg in ready:
            md, _ = self._paths(when)
            with md.open("a", encoding="utf-8") as f:
                # Background voices are marked, so the readable transcript doesn't put a video's words in their mouth.
                f.write(f"**[{when:%H:%M:%S}] {speaker}{f' · {chname}' if chname else ''}:** {BG_MARK if bg else ''}{text}\n\n")


BG_MARK = "(background voice) "      # .md prefix for lines that didn't sound like the speaker
MD_LINE = re.compile(r"^\*\*\[(\d\d):(\d\d):(\d\d)\] (.+?):\*\* (.*)$")


def load_day(day: str) -> list[dict]:
    """All lines for a day. Older lines that predate the .jsonl data file are
    recovered from the .md transcript (duration estimated from word count)."""
    recs: list[dict] = []
    jl, md = TRANSCRIPTS / f"{day}.jsonl", TRANSCRIPTS / f"{day}.md"
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except ValueError:
                pass
    first = min((r["t"] for r in recs), default=float("inf"))
    if md.exists():
        base = dt.datetime.strptime(day, "%Y-%m-%d")
        for line in md.read_text(encoding="utf-8").splitlines():
            m = MD_LINE.match(line)
            if not m:
                continue
            h, mi, se, who, text = m.groups()
            t = int((base.replace(hour=int(h), minute=int(mi), second=int(se))).timestamp() * 1000)
            if t >= first - 1000:
                continue
            words = max(1, len(text.split()))
            recs.append({"t": t, "end": t + int(words / 2.6 * 1000), "user": "", "speaker": who, "text": text})
    # Lines written before the lead-in fix could start minutes early (a scrap of old
    # sound was glued to the front): a few words can't take minutes, so such a line
    # starts where its words could have (about 2.6 words a second, at least 1.5 s).
    for r in recs:
        span = r.get("end", r["t"]) - r["t"]
        if span > 30000 and r.get("text"):
            need = max(1500, len(r["text"].split()) / 2.6 * 1000)
            if span > need * 4:
                r["t"] = int(r["end"] - need)
    gone = {x for r in recs for x in (r.get("replaces") or [])}   # lines the 2B pass rewrote
    if gone or any(r.get("tomb") for r in recs):
        recs = [r for r in recs if not r.get("tomb") and r.get("id") not in gone]
    recs.sort(key=lambda r: r["t"])
    return recs


GAP_MS = 15 * 60 * 1000
_day_cache: dict[str, tuple[float, list, list]] = {}


def day_data(day: str) -> tuple[list[dict], list[dict]]:
    """(lines, calls) for a day, cached until its files change."""
    files = [TRANSCRIPTS / f"{day}.jsonl", TRANSCRIPTS / f"{day}.md"]
    stamp = sum(f.stat().st_mtime for f in files if f.exists())
    hit = _day_cache.get(day)
    if hit and hit[0] == stamp:
        return hit[1], hit[2]
    recs = load_day(day)
    calls = build_calls(day, recs)
    _day_cache[day] = (stamp, recs, calls)
    return recs, calls


def build_calls(day: str, recs: list[dict]) -> list[dict]:
    """Split a day into calls: a new call starts when the voice channel
    changes or after 15 quiet minutes."""
    calls: list[dict] = []
    open_by_ch: dict = {}              # several calls can run at once, one per voice channel
    for r in recs:
        ch = r.get("ch")
        cur = open_by_ch.get(ch) if ch else (calls[-1] if calls else None)
        if ch and cur is None and calls and not calls[-1]["ch"] and r["t"] - calls[-1]["end"] < GAP_MS:
            cur = calls[-1]                # older lines without a channel: the call they belong to
        if cur and r["t"] - cur["end"] < GAP_MS and (not ch or not cur["ch"] or ch == cur["ch"]):
            cur["lines"].append(r); cur["end"] = max(cur["end"], r["end"])
            if ch and not cur["ch"]:
                cur.update(ch=ch, chName=r.get("chName"), guild=r.get("guild"), guildName=r.get("guildName"))
                open_by_ch[ch] = cur
        else:
            calls.append({"day": day, "start": r["t"], "end": r["end"], "ch": ch, "chName": r.get("chName"),
                          "guild": r.get("guild"), "guildName": r.get("guildName"), "lines": [r]})
            if ch:
                open_by_ch[ch] = calls[-1]
    out = []
    for c in calls:
        talk: dict[str, float] = {}
        words = 0
        own = [r for r in c["lines"] if not r.get("bg")]     # background voices aren't the person talking
        for r in own:
            talk[r["speaker"]] = talk.get(r["speaker"], 0) + (r["end"] - r["t"]) / 1000
            words += len(r["text"].split())
        people = sorted(talk.items(), key=lambda kv: -kv[1])
        out.append({"id": f"{day}@{c['start']}", "day": day, "start": c["start"], "end": c["end"],
                    "ch": c["ch"], "chName": c["chName"], "guild": c["guild"], "guildName": c["guildName"],
                    "people": [p for p, _ in people], "lines": len(own), "words": words,
                    "total": len(c["lines"]),          # every line, background included (what a delete removes)
                    "preview": next((r["text"] for r in reversed(own) if len(r["text"]) > 20), own[-1]["text"] if own else "")[:120]})
    return out


def srv_key(c: dict) -> str:
    if c.get("guild"):
        return "g:" + c["guild"]
    if c.get("guildName"):
        return "gn:" + c["guildName"]
    return "dm" if c.get("chName") == "Private call" else "unknown"


def ch_key(c: dict) -> str:
    return srv_key(c) + "|" + (c.get("ch") or c.get("chName") or "?")


def group_match(c: dict, kind: str, key: str, parts: dict | None = None) -> bool:
    if kind == "call":                     # a merged call matches every one of its parts
        return c["id"] == key or (parts or {}).get(c["id"]) == key
    return (kind == "channel" and ch_key(c) == key) or (kind == "server" and srv_key(c) == key)


# ---------------------------------------------------------------- merged calls
# Calls are split automatically (a new channel, 15 quiet minutes, midnight). You
# can merge several into one: the groups are kept in merges.json as lists of the
# calls' own ids, so merging changes no transcript and is undone by removing
# the group. A merged call takes the id of its earliest part.
def merges_path() -> Path:
    return TRANSCRIPTS.parent / "merges.json"


def merge_groups() -> list[list[str]]:
    try:
        groups = json.loads(merges_path().read_text(encoding="utf-8")).get("groups", [])
        return [[str(i) for i in g] for g in groups if isinstance(g, list) and len(g) >= 2]
    except Exception:
        return []


def save_merge_groups(groups: list[list[str]]):
    tmp = merges_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"groups": [g for g in groups if len(g) >= 2]}, indent=1), encoding="utf-8")
    tmp.replace(merges_path())


def merge_calls(ids: list[str]) -> dict:
    """Merge these calls (and any merged calls they already belong to) into one."""
    raw = {c["id"] for d in all_days() for c in day_data(d)[1]}
    ids = [i for i in dict.fromkeys(ids) if i in raw]
    groups = resolved_groups()
    owner = {i: g[0] for g in groups for i in g}
    if len({owner.get(i, i) for i in ids}) < 2:             # two different calls (or merged calls) at least
        return {"error": "pick at least two calls"}
    joined = set(ids)
    for g in [g for g in groups if joined & set(g)]:      # merging into an already merged call
        joined |= set(g); groups.remove(g)
    if len(joined) < 2:
        return {"error": "pick at least two calls"}
    parts = sorted(joined, key=lambda i: (i.split("@")[0], int(i.split("@")[1])))
    groups.append(parts)
    save_merge_groups(groups)
    return {"ok": True, "id": parts[0], "parts": len(parts)}


def unmerge_call(cid: str) -> dict:
    groups = resolved_groups()
    hit = [g for g in groups if cid in g]
    save_merge_groups([g for g in groups if cid not in g])
    return {"ok": bool(hit), "parts": len(hit[0]) if hit else 0}


def resolved_groups() -> list[list[str]]:
    """merge groups in today's call ids. A call's id is the day and start of its
    first line, so it changes if an earlier line arrives late: a stored id that no
    longer exists is matched to the call on that day that now covers its time."""
    calls = {d: day_data(d)[1] for d in all_days()}
    ids = {c["id"] for cs in calls.values() for c in cs}
    def resolve(i):
        if i in ids:
            return i
        try:
            d, t = i.split("@"); t = int(t)
        except ValueError:
            return None
        hit = next((c for c in calls.get(d, []) if c["start"] - 60000 <= t <= c["end"]), None)
        return hit["id"] if hit else None
    out = []
    for g in merge_groups():
        r = list(dict.fromkeys(x for x in map(resolve, g) if x))
        if len(r) >= 2:
            out.append(r)
    return out


def merged_id_of() -> dict[str, str]:
    """call id -> id of the merged call it is part of, as /api/calls shows it."""
    return {p["id"]: c["id"] for c in all_calls() if c.get("merged") for p in c["parts"]}


def prune_merges():
    """after a delete: keep only groups that still have two calls, in current ids."""
    save_merge_groups(resolved_groups())


def all_calls() -> list[dict]:
    """Every call, newest first, with merged calls folded into one entry.
    Each call lists its parts ({id, day, start, end, ch}); a plain call has one."""
    raw = [c for d in all_days() for c in reversed(day_data(d)[1])]
    by_id = {c["id"]: c for c in raw}
    part = lambda c: {"id": c["id"], "day": c["day"], "start": c["start"], "end": c["end"], "ch": c["ch"]}
    merged, used = {}, set()
    for g in resolved_groups():
        cs = sorted((by_id[i] for i in g if i in by_id), key=lambda c: c["start"])
        if len(cs) < 2:
            continue
        first, talk = cs[0], {}
        for c in cs:
            for rank, p in enumerate(c["people"]):
                talk[p] = talk.get(p, 0) + 1 / (rank + 1)        # people ranked by how high they were in each part
        latest = max(cs, key=lambda c: c["end"])             # channel and server of where it is now
        merged[first["id"]] = {**first, **{k: latest[k] for k in ("ch", "chName", "guild", "guildName")},
                               "start": first["start"], "end": max(c["end"] for c in cs),
                               "people": sorted(talk, key=lambda p: -talk[p]),
                               "lines": sum(c["lines"] for c in cs), "words": sum(c["words"] for c in cs),
                               "total": sum(c["total"] for c in cs), "preview": cs[-1]["preview"],
                               "merged": True, "parts": [part(c) for c in cs],
                               "chNames": list(dict.fromkeys(c["chName"] or "" for c in cs))}
        used |= {c["id"] for c in cs[1:]}
    out = []
    for c in raw:
        if c["id"] in used:
            continue
        out.append(merged.get(c["id"]) or {**c, "parts": [part(c)]})
    out.sort(key=lambda c: -c["parts"][-1]["start"])          # newest first, by each call's latest part
    return out


def archive_path() -> Path:
    return TRANSCRIPTS.parent / "archive.json"


def archived_keys() -> set[str]:
    try:
        return set(json.loads(archive_path().read_text(encoding="utf-8")).get("keys", []))
    except Exception:
        return set()


def set_archived(key: str, on: bool):
    keys = archived_keys()
    (keys.add if on else keys.discard)(key)
    archive_path().write_text(json.dumps({"keys": sorted(keys)}, indent=1), encoding="utf-8")


def delete_group(kind: str, key: str) -> dict:
    """Remove every line of a call / channel / server from the transcripts.
    Removed lines go to transcripts/.trash/ (recoverable), not into the void."""
    trash = TRANSCRIPTS / ".trash"
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    n_lines = n_calls = 0
    parts = merged_id_of()
    for d in all_days():
        _, calls = day_data(d)
        targets = [c for c in calls if group_match(c, kind, key, parts)]
        if not targets:
            continue
        n_calls += len(targets)
        ranges = [(c["start"], c["end"], c["ch"]) for c in targets]
        def hit(t, ch, slack=0):
            return any(s0 - slack <= t <= e0 + slack and (not ch or not cc or ch == cc) for s0, e0, cc in ranges)
        trash.mkdir(parents=True, exist_ok=True)
        jl, md = TRANSCRIPTS / f"{d}.jsonl", TRANSCRIPTS / f"{d}.md"
        if jl.exists():
            keep, gone = [], []
            for line in jl.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    keep.append(line); continue
                (gone if hit(r["t"], r.get("ch")) else keep).append(line)
            if gone:
                with (trash / f"{d}-deleted-{stamp}.jsonl").open("a", encoding="utf-8") as f:
                    f.write("\n".join(gone) + "\n")
                tmp = jl.with_suffix(".jsonl.tmp"); tmp.write_text("".join(l + "\n" for l in keep), encoding="utf-8"); tmp.replace(jl)
                n_lines += len(gone)
        if md.exists():
            base = dt.datetime.strptime(d, "%Y-%m-%d")
            out, gone, skip_blank = [], [], False
            for line in md.read_text(encoding="utf-8").splitlines():
                m = MD_LINE.match(line)
                if m:
                    h, mi, se = (int(x) for x in m.groups()[:3])
                    t = int(base.replace(hour=h, minute=mi, second=se).timestamp() * 1000)
                    if hit(t, None, slack=1000):         # .md has 1-second timestamps
                        gone.append(line); skip_blank = True; continue
                if skip_blank and line == "":
                    skip_blank = False; continue
                skip_blank = False
                out.append(line)
            if gone:
                with (trash / f"{d}-deleted-{stamp}.md").open("a", encoding="utf-8") as f:
                    f.write("\n\n".join(gone) + "\n")
                tmp = md.with_suffix(".md.tmp"); tmp.write_text("\n".join(out) + "\n", encoding="utf-8"); tmp.replace(md)
                if not jl.exists():
                    n_lines += len(gone)
        _day_cache.pop(d, None)
    set_archived(f"{kind}:{key}", False)
    if kind == "call":
        unmerge_call(key)
    prune_merges()
    return {"calls": n_calls, "lines": n_lines}


def parse_parts(raw: str) -> list | None:
    """'day~start~end~channel;...' from the dashboard: the stretches a (merged) call covers."""
    from urllib.parse import unquote_plus
    out = []
    for sp in unquote_plus(raw or "").split(";"):
        bits = sp.split("~")
        if len(bits) == 4 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", bits[0]) and bits[1].isdigit() and bits[2].isdigit():
            out.append((bits[0], int(bits[1]), int(bits[2]), bits[3]))
    return out or None


# ---------------------------------------------------------------- people
# Everyone ever heard, by their Discord id (it never changes, unlike display
# names). people.json holds the usernames and names each id has used; every
# person with a known username also gets people/<username>.md with all their lines.
def safe_file_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("._")[:60] or "person"


def person_lines(uid: str) -> list[dict]:
    out = []
    for d in sorted(all_days()):
        recs, calls = day_data(d)
        for r in recs:
            if r.get("user") == uid and not r.get("bg") and r.get("text"):
                call = next((c for c in calls if c["start"] <= r["t"] <= c["end"]), None)
                out.append({"day": d, "t": r["t"], "end": r.get("end", r["t"]), "text": r["text"], "speaker": r.get("speaker"),
                            "chName": r.get("chName") or (call or {}).get("chName"), "guildName": r.get("guildName") or (call or {}).get("guildName"),
                            "call": call and call["id"]})
    return out


_people_cache: tuple | None = None


def people_summary(hub) -> list[dict]:
    """Everyone with at least one line: id, username, every name used, counts, first and last time."""
    global _people_cache
    days = all_days()
    stamp = (tuple((d, day_data(d)[0].__len__()) for d in days), len(hub.people_dir))
    if _people_cache and _people_cache[0] == stamp:
        return _people_cache[1]
    agg: dict = {}
    for d in days:
        for r in day_data(d)[0]:
            uid = r.get("user")
            if not uid or r.get("bg") or not r.get("text") or not re.fullmatch(r"\d{15,22}", uid):
                continue
            a = agg.setdefault(uid, {"id": uid, "lines": 0, "words": 0, "first": r["t"], "last": r["t"], "spoke": {}})
            a["lines"] += 1; a["words"] += len(r["text"].split())
            a["first"] = min(a["first"], r["t"]); a["last"] = max(a["last"], r["t"])
            if r.get("speaker") and not re.fullmatch(r"user-\d{4}", r["speaker"]):
                a["spoke"][r["speaker"]] = max(a["spoke"].get(r["speaker"], 0), r["t"])
    out = []
    for uid, a in agg.items():
        p = hub.people_dir.get(uid) or {}
        shown = sorted(a.pop("spoke").items(), key=lambda kv: -kv[1])
        out.append({**a, "username": p.get("username"), "usernames": p.get("usernames", []),
                    "names": list(dict.fromkeys([n for n, _ in shown] + p.get("names", []) + p.get("nicks", []))),
                    "display": (p.get("names") or [None])[-1], "nick": (p.get("nicks") or [None])[-1],
                    "name": shown[0][0] if shown else (p.get("names") or [uid])[0]})
    out.sort(key=lambda x: -x["last"])
    _people_cache = (stamp, out)
    return out


def person_markdown(hub, uid: str) -> str:
    p = hub.people_dir.get(uid) or {}
    lines = person_lines(uid)
    names = list(dict.fromkeys([r["speaker"] for r in reversed(lines) if r.get("speaker")] + p.get("names", []) + p.get("nicks", [])))
    title = p.get("username") and "@" + p["username"] or (names[0] if names else uid)
    head = [f"# {title}", "", f"Discord id: {uid}"]
    if p.get("usernames"):
        head.append("Usernames: " + ", ".join("@" + u for u in p["usernames"]))
    if names:
        head.append("Names shown: " + ", ".join(names))
    head.append(f"Lines: {len(lines)}")
    body, last_key = [], None
    for r in lines:
        key = (r["day"], r.get("call"))
        if key != last_key:
            where = " · ".join(x for x in (r.get("chName"), r.get("guildName")) if x)
            body += ["", f"## {r['day']}" + (f" · {where}" if where else "")]
            last_key = key
        when = dt.datetime.fromtimestamp(r["t"] / 1000).strftime("%H:%M:%S")
        body.append(f"- {when} **{r.get('speaker') or title}:** {r['text']}")
    return "\n".join(head + body) + "\n"


def write_people_files(hub, uids) -> int:
    folder = TRANSCRIPTS.parent / "people"
    n = 0
    for uid in list(uids):
        un = hub.username_of(uid)
        if not un:
            continue
        try:
            folder.mkdir(exist_ok=True)
            f = folder / f"{safe_file_name(un)}.md"
            tmp = f.with_suffix(".md.tmp"); tmp.write_text(person_markdown(hub, uid), encoding="utf-8"); tmp.replace(f)
            n += 1
        except OSError as e:
            print("people file:", e, flush=True)
    return n


def all_days() -> list[str]:
    return sorted({f.stem for f in TRANSCRIPTS.glob("*.md")} | {f.stem for f in TRANSCRIPTS.glob("*.jsonl")}, reverse=True)


# ---------------------------------------------------------------- updates
# Updates come from GitHub release tags (vX.Y.Z) via git, which is already
# signed in on each machine (the repo is private: no tokens in the app).
def git(*args, timeout=60):
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return 1, str(e)


def vtuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v or "")[:3]) or (0,)


class Updater:
    def __init__(self, hub: "Hub"):
        self.hub = hub
        self.lock = threading.Lock()
        self.state = {"current": self.current(), "latest": None, "available": False, "notes": "",
                      "checked": None, "error": None, "busy": False, "applied": None}

    @staticmethod
    def current() -> str:
        try:
            return (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        except OSError:
            return "0.0.0"

    def check(self) -> dict:
        st = self.state
        st["current"] = self.current()
        if not (ROOT / ".git").exists():
            st.update(error="This copy isn't a git checkout, so it can't update itself. Install with git clone to get updates.")
            return st
        rc, out = git("fetch", "--tags", "--force", "--quiet", "origin", timeout=90)
        if rc != 0:
            st.update(error="Couldn't reach GitHub: " + out[-200:], checked=time.time())
            return st
        rc, out = git("tag", "-l", "v*", "--sort=-v:refname")
        latest = next((t for t in out.splitlines() if re.fullmatch(r"v\d+\.\d+\.\d+", t.strip())), None)
        notes = git("tag", "-l", "--format=%(contents)", latest)[1] if latest else ""
        st.update(latest=latest, notes=notes.strip()[:4000], checked=time.time(), error=None,
                  available=bool(latest) and vtuple(latest) > vtuple(st["current"]))
        return st

    def apply(self) -> dict:
        with self.lock:
            st = self.check()
            if st.get("error"):
                return {"ok": False, "error": st["error"]}
            if not st["available"]:
                return {"ok": False, "error": "Already up to date."}
            st["busy"] = True
            try:
                tag = st["latest"]
                rc, dirty = git("status", "--porcelain", "--untracked-files=no")
                if dirty:
                    return {"ok": False, "error": "This copy has local code changes; not touching them. (git status shows them.)"}
                rc, branch = git("rev-parse", "--abbrev-ref", "HEAD")
                if branch.strip() != "main":
                    return {"ok": False, "error": f"On branch {branch.strip()!r}, expected main."}
                rc, changed = git("diff", "--name-only", "HEAD", tag)
                files = [f for f in changed.splitlines() if f.strip()]
                rc, out = git("merge", "--ff-only", "--quiet", tag)
                if rc != 0:
                    return {"ok": False, "error": "Couldn't fast-forward to " + tag + ": " + out[-200:]}
                result = {"ok": True, "from": st["current"], "version": self.current(), "files": len(files),
                          "rebuildMac": any(f.startswith("app/mac/") or f.startswith("app/desktop/package") for f in files),
                          "app": any(f.startswith("app/desktop/package") for f in files),   # new Electron / npm packages
                          "deps": any(f in ("server/pyproject.toml", "server/uv.lock") for f in files),
                          "server": any(f.startswith("server/") for f in files)}
                st.update(applied=result, available=False, current=self.current())
                return result
            finally:
                st["busy"] = False

    def loop(self):
        time.sleep(20)
        while True:
            try:
                if current_settings()["autoUpdate"]:
                    self.check()
            except Exception as e:
                self.state["error"] = str(e)[:200]
            time.sleep(6 * 3600)


class Hub:
    """Live state for the dashboard: who's connected, who's talking, new lines."""

    def __init__(self, model: str):
        self.model, self.model_ready, self.started = model, False, time.time()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.clients: set = set()
        self.extensions = 0
        self.ext_sockets: set = set()
        self.ext_reports: dict = {}       # connection -> (status, time)
        # One entry per Discord window (account): who it is and whether it is the
        # one transcribing its channel (two of your accounts in one channel must
        # not transcribe everything twice).
        self.conns: dict = {}             # ws -> {account, role, userId, label, since, capturing}
        self.uid_owner: dict = {}         # person -> (connection, last chunk time): one source per person
        self.res: dict = {}
        self.parent_pid = 0
        self.audio_s = self.proc_s = 0.0             # transcription throughput
        self.latency: deque = deque(maxlen=50)        # s from end of speech to text
        self.people: dict[str, "Segmenter"] = {}
        self.queue: queue.Queue | None = None
        self.load_people()
        self.names_file = TRANSCRIPTS.parent / "names.json"
        try:
            self.names: dict[str, str] = json.loads(self.names_file.read_text(encoding="utf-8"))
        except Exception:
            self.names = {}

    # ---- people: everyone's Discord account, with every name they have used
    def load_people(self):
        self.people_file = TRANSCRIPTS.parent / "people.json"
        self.people_dirty: set = set()
        try:
            self.people_dir: dict = json.loads(self.people_file.read_text(encoding="utf-8"))
        except Exception:
            self.people_dir = {}

    def username_of(self, uid: str) -> str | None:
        return (self.people_dir.get(uid) or {}).get("username") if uid else None

    def learn_account(self, uid: str, a: dict):
        """{u: username, g: display name, n: server nickname} for a Discord id."""
        if not uid or not re.fullmatch(r"\d{15,22}", uid) or not isinstance(a, dict) or not a.get("u"):
            return
        p = self.people_dir.setdefault(uid, {"username": None, "usernames": [], "names": [], "nicks": [], "first": None})
        now = int(time.time() * 1000)
        changed = p.get("username") != a["u"]
        p["username"] = a["u"]
        for key, val in (("usernames", a["u"]), ("names", a.get("g")), ("nicks", a.get("n"))):
            if val and val not in p[key]:
                p[key].append(str(val)[:80]); changed = True
        p["first"] = p.get("first") or now
        if changed:
            self.people_dirty.add(uid)
            try:
                tmp = self.people_file.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(self.people_dir, ensure_ascii=False, indent=1), encoding="utf-8")
                tmp.replace(self.people_file)
            except OSError:
                pass

    def learn_name(self, uid: str, name: str):
        """Remember the real name behind a Discord id (relabels old user-1234 lines)."""
        if not name or re.fullmatch(r"user-\d{4}", name) or self.names.get(uid) == name:
            return
        self.names[uid] = name
        try:
            self.names_file.write_text(json.dumps(self.names, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass

    def own_ids(self) -> set:
        ids = {c.get("userId") for c in self.conns.values() if c.get("userId")}
        for st, _ in self.ext_reports.values():
            if (st.get("me") or {}).get("id"):
                ids.add(st["me"]["id"])
        return ids

    def elect(self) -> list:
        """Per voice channel, pick the one connection that transcribes it:
        a listener account before your main one, then whoever joined first.
        Returns [(ws, on)] for connections whose role changed."""
        now, groups = time.time(), {}
        for ws, c in self.conns.items():
            st, at = self.ext_reports.get(ws, ({}, 0.0))
            ch = (st.get("channel") or {}).get("id") if now - at < 15 and st.get("pcs") and st.get("tracks") else None
            if ch:
                groups.setdefault(ch, []).append((ws, c))
        chosen, grouped = set(), set()
        for members in groups.values():
            members.sort(key=lambda m: (m[1].get("role") != "listener", m[1]["since"]))
            chosen.add(members[0][0])
            grouped.update(m[0] for m in members)
        changes = []
        for ws, c in self.conns.items():
            want = ws in chosen or ws not in grouped
            if c["capturing"] != want:
                c["capturing"] = want
                changes.append((ws, want))
        return changes

    def accounts_state(self, now: float) -> list:
        own, out = self.own_ids(), []
        for ws, c in self.conns.items():
            st, at = self.ext_reports.get(ws, ({}, 0.0))
            fresh = now - at < 15
            me = st.get("me") or {}
            self_st = (me.get("self") or {}) if fresh else {}
            out.append({"account": c.get("account"), "role": c.get("role"), "label": c.get("label") or me.get("name"),
                        "userId": c.get("userId") or me.get("id"), "capturing": c["capturing"],
                        "inCall": fresh and bool(st.get("pcs")) and bool(st.get("tracks")),
                        "channel": st.get("channel") if fresh else None,
                        "streams": len([x for x in (st.get("liveIds") or []) if x not in own]) if fresh else 0,
                        "present": [x for x in (st.get("liveIds") or []) if x not in own] if fresh else [],
                        "deafened": bool(self_st.get("deaf") or self_st.get("selfDeaf")),
                        # call audio lost: Discord kept failing to decrypt it over the last 30 s and
                        # nobody has been audible for 30 s (counted from joining if nobody has been yet)
                        "audioLost": bool(fresh and st.get("pcs") and st.get("tracks") and (st.get("decryptFails") or 0) > 100
                                          and st.get("heardAgo") is not None and st.get("heardAgo") > 30),
                        "lastReport": int(at * 1000) if at else None})
        return out

    def state(self) -> dict:
        now = time.time()
        ppl = [{"user": uid, "name": p.label, "talking": bool(p.cur) and now - p.last_seen < 1.5,
                "lastSeen": int(p.last_seen * 1000)} for uid, p in self.people.items() if now - p.last_seen < 120]
        # Report from the connection that matters: in a call first, then the app, then the latest.
        reports = [r for r in self.ext_reports.values() if now - r[1] < 15] or \
                  sorted(self.ext_reports.values(), key=lambda r: r[1])[-1:]
        best = max(reports, key=lambda r: (bool(r[0].get("pcs")) and bool(r[0].get("tracks")),
                                           r[0].get("client") == "app", r[1]), default=({}, 0.0))
        st, at = best
        age = now - at
        ext = {"connected": self.extensions > 0,
               "discordTab": self.extensions > 0 and age < 15,          # a Discord tab reports every 5 s
               "paused": bool(st.get("paused")) if age < 15 else False,
               "inCall": bool(st.get("pcs")) and bool(st.get("tracks")),
               "streams": len([x for x in (st.get("liveIds") or []) if x not in self.own_ids()]) if st.get("liveIds") is not None else len(st.get("live") or []),   # in the channel now
               "named": sum(1 for n in (st.get("live") or []) if not re.fullmatch(r"\d{15,22}", str(n))),
               "audio": st.get("ctx"), "hook": st.get("v"), "errors": st.get("errors") or [],
               "channel": st.get("channel") if age < 15 else None,
               "present": [x for x in (st.get("liveIds") or []) if x not in self.own_ids()] if age < 15 else [],   # in the call now (not your accounts)
               "client": st.get("client") or "chrome",           # "app": Discord inside the desktop app
               "lastReport": int(at * 1000) if at else None}
        return {"type": "state", "model": self.model, "engine": getattr(self, "engine", None), "modelReady": self.model_ready,
                "modelId": getattr(self, "model_id", None), "modelLoading": getattr(self, "loading", None),
                "modelError": getattr(self, "model_error", None),
                "refine": getattr(self, "refine_id", None), "refineLoading": getattr(self, "refine_loading", None),
                "refineQueue": len(getattr(self, "refine_q", ()) or ()), "refined": getattr(self, "refined", 0),
                "extensions": self.extensions, "ext": ext, "people": ppl,
                "accounts": self.accounts_state(now), "own": sorted(self.own_ids()), "queue": self.queue.qsize() if self.queue else 0,
                "uptime": int(now - self.started), "now": int(now * 1000), "pid": os.getpid(),
                "res": {**self.res, "speed": round(self.audio_s / self.proc_s, 1) if self.proc_s else None,
                        "latency": round(sum(self.latency) / len(self.latency), 2) if self.latency else None,
                        "audioMin": round(self.audio_s / 60, 1)}}

    def sample_resources(self):
        """Every 2 s, what Earshot uses, measured the way the OS reports it:
        the speech server, the app and its call windows (the app is this
        server's parent process and its helpers), model / GPU memory, and the
        machine as a whole."""
        import psutil
        procs: dict[int, psutil.Process] = {}
        cores = os.cpu_count() or 1
        last_gpu, gpu = 0.0, {}

        def mem(p: psutil.Process) -> float:
            """Bytes of memory a process really uses (Activity Monitor's / Task Manager's number)."""
            if sys.platform == "darwin":
                fp = phys_footprint(p.pid)
                if fp:
                    return fp
            mi = p.memory_info()
            return getattr(mi, "private", 0) or mi.rss          # Windows: private bytes

        def tracked(pid: int):
            if pid not in procs:
                p = psutil.Process(pid)
                p.cpu_percent(None)                                 # first call primes the counter
                procs[pid] = p
            return procs[pid]

        while True:
            time.sleep(2)
            try:
                me = tracked(os.getpid())
                app_pids: list[int] = []
                if self.parent_pid and psutil.pid_exists(self.parent_pid):
                    parent = psutil.Process(self.parent_pid)
                    app_pids = [parent.pid] + [c.pid for c in parent.children(recursive=True) if c.pid != me.pid
                                               and c.pid not in {x.pid for x in me.children(recursive=True)}]
                server_pids = [me.pid] + [c.pid for c in me.children(recursive=True)]
                live = set(server_pids + app_pids)
                for pid in list(procs):
                    if pid not in live:
                        procs.pop(pid, None)
                cpu_s = mem_s = cpu_a = mem_a = 0.0
                for pid in server_pids:
                    try:
                        p = tracked(pid); cpu_s += p.cpu_percent(None); mem_s += mem(p)
                    except psutil.Error:
                        pass
                for pid in app_pids:
                    try:
                        p = tracked(pid); cpu_a += p.cpu_percent(None); mem_a += mem(p)
                    except psutil.Error:
                        pass
                vm = psutil.virtual_memory()
                model = vram_used = vram_total = None
                if "mlx.core" in sys.modules:                         # Apple Silicon: GPU memory is shared memory
                    mx = sys.modules["mlx.core"]
                    model = mx.get_active_memory() + mx.get_cache_memory()
                elif "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
                    torch = sys.modules["torch"]
                    model = torch.cuda.memory_reserved()
                    free, total = torch.cuda.mem_get_info()
                    vram_used, vram_total = total - free, total       # every program's use of the card
                if time.time() - last_gpu > 5 and shutil.which("nvidia-smi"):
                    last_gpu = time.time()
                    try:
                        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,name",
                                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.splitlines()[0]
                        u, used, tot, name = [x.strip() for x in out.split(",", 3)]
                        gpu = {"gpuLoad": float(u), "gpuName": name}
                        vram_used, vram_total = float(used) * 2**20, float(tot) * 2**20
                    except Exception:
                        gpu = {}
                elif gpu and vram_used is None:
                    pass
                MB = lambda b: round(b / 2**20) if b else None
                self.res = {
                    "cores": cores,
                    "cpu": round(cpu_s, 1),                                         # speech server, % of one core
                    "cpuServer": round(cpu_s / cores, 1), "cpuApp": round(cpu_a / cores, 1),
                    "cpuTotal": round((cpu_s + cpu_a) / cores, 1),                  # % of the whole processor
                    "memServerMB": MB(mem_s), "memAppMB": MB(mem_a), "memTotalMB": MB(mem_s + mem_a),
                    "rssMB": MB(mem_s + mem_a),
                    "modelMB": MB(model), "unified": "mlx.core" in sys.modules,
                    "vramUsedMB": MB(vram_used), "vramTotalMB": MB(vram_total), **gpu,
                    "sysMemUsedMB": MB(vm.total - vm.available), "sysMemTotalMB": MB(vm.total),
                    "appProcs": len(app_pids),
                }
            except Exception as e:
                self.res = {**self.res, "error": str(e)[:120]}

    def tell_extensions(self, msg: dict):   # safe to call from any thread
        data = json.dumps(msg)
        async def send():
            for w in list(self.ext_sockets):
                try:
                    await w.send(data)
                except Exception:
                    pass
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(send()))

    def publish(self, msg: dict):           # safe to call from any thread
        if self.loop:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send(msg)))

    async def _send(self, msg: dict):
        data = json.dumps(msg, ensure_ascii=False)
        for c in list(self.clients):
            try:
                await c.send(data)
            except Exception:
                self.clients.discard(c)


# ---------------------------------------------------------------- model
# ---------------------------------------------------------------- engines
# mlx  : Apple Silicon (Metal) via mlx-audio — the Mac default.
# torch: Hugging Face transformers on an NVIDIA GPU (CUDA) or the CPU.
# onnx : ONNX Runtime (onnx-asr) — light, fast on any CPU; the Windows default
#        when there is no NVIDIA GPU.
ENGINE_DEFAULTS = {
    "mlx": "ibm-granite/granite-speech-5.0-470m-turboctc",
    "torch": "ibm-granite/granite-speech-5.0-470m-turboctc",
    "onnx": "nemo-parakeet-tdt-0.6b-v2",
}


# Models the picker offers. "repos" maps engine -> model id for that runtime;
# "bench" holds our measured results on real meeting speech (AMI, 120 clips):
# word error rate % and speed (x real time) on the machines we tested.
MODELS = [
    {"id": "granite-470m", "name": "Granite Speech 5.0 · 470M", "default": True,
     "about": "Fast and light: transcribes live. English only.", "langs": "English",
     "repos": {"mlx": "ibm-granite/granite-speech-5.0-470m-turboctc", "torch": "ibm-granite/granite-speech-5.0-470m-turboctc"},
     "bench": {"mlx": {"wer": 5.5, "speed": 249, "mem": 1.6}, "torch": {"wer": 5.5, "speed": 82, "mem": 0.9}}},
    {"id": "granite-2b", "name": "Granite Speech 5.0 · 2B",
     "about": "The most accurate; about 5x the memory and much slower. English only.", "langs": "English",
     "repos": {"mlx": "ibm-granite/granite-speech-5.0-2b", "torch": "ibm-granite/granite-speech-5.0-2b"},
     "bench": {"mlx": {"wer": 5.4, "speed": 13, "mem": 9.1}, "torch": {"wer": 5.4, "speed": 11}}},
    # Not offered: used automatically on PCs without an NVIDIA GPU, where Granite can't keep up.
    {"id": "parakeet-v3", "name": "Parakeet TDT 0.6B v3", "fallback": True,
     "about": "Used on this PC because it has no NVIDIA graphics card: Granite needs one to keep up.", "langs": "25 European",
     "repos": {"onnx": "nemo-parakeet-tdt-0.6b-v3"},
     "bench": {"onnx": {"wer": 7.4, "speed": 30, "mem": 2.0}}},
]
GRANITE = ("granite-470m", "granite-2b")


def engines_here() -> list[str]:
    """Speech runtimes this machine can use, best first."""
    out = []
    import platform
    if sys.platform == "darwin" and platform.machine() == "arm64":
        try:
            import mlx_audio  # noqa: F401
            out.append("mlx")
        except Exception:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            out.append("torch")
    except Exception:
        pass
    try:
        import onnx_asr  # noqa: F401
        out.append("onnx")
    except Exception:
        pass
    return out


def models_here() -> list[dict]:
    have = engines_here()
    out = []
    for m in MODELS:
        eng = next((e for e in have if e in m["repos"]), None)
        out.append({**{k: v for k, v in m.items() if k != "repos"}, "engine": eng, "repo": m["repos"].get(eng) if eng else None,
                    "available": bool(eng), "stats": m.get("bench", {}).get(eng) if eng else None})
    return out


def phys_footprint(pid: int) -> int | None:
    """macOS: a process's physical footprint, the memory figure Activity Monitor
    shows (RSS misses GPU buffers that MLX models live in)."""
    try:
        import ctypes, ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        class RUsageV2(ctypes.Structure):
            _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
                "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size", "resident_size",
                "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time", "child_system_time",
                "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins", "child_elapsed_abstime",
                "diskio_bytesread", "diskio_byteswritten")]
        buf = RUsageV2()
        if lib.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(2), ctypes.byref(buf)) == 0:
            return int(buf.phys_footprint)
    except Exception:
        pass
    return None


def model_plan() -> dict:
    """Which models run: the Granite ones you switched on (470M live, 2B live, or
    470M live + 2B rewriting each line), or Parakeet when Granite can't run here."""
    here = {m["id"]: m for m in models_here() if m["available"]}
    granite = [g for g in GRANITE if g in here]
    if not granite:
        fb = here.get("parakeet-v3") or next(iter(here.values()), None)
        return {"live": fb, "refine": None, "selected": [], "fallback": True}
    st = load_settings()
    sel = [g for g in (st.get("models") or []) if g in granite]
    if not sel and st.get("model") in granite:          # the old single choice
        sel = [st["model"]]
    if not sel:
        sel = ["granite-470m"] if "granite-470m" in granite else granite[:1]
    live = here["granite-470m"] if "granite-470m" in sel else here[sel[0]]
    refine = here["granite-2b"] if len(sel) == 2 else None
    return {"live": live, "refine": refine, "selected": sel, "fallback": False}


def settings_path() -> Path:
    return TRANSCRIPTS.parent / "settings.json"


def load_settings() -> dict:
    try:
        return json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(**kw):
    st = load_settings(); st.update(kw)
    settings_path().write_text(json.dumps(st, indent=1), encoding="utf-8")


# Settings the dashboard can change (stored with the model choice in settings.json).
SETTING_DEFAULTS = {"splitLines": True, "personality": True, "qaLinks": True, "nameStyle": "nick", "autoUpdate": True, "sensitivity": "normal", "language": "en",
                    "background": "dim"}
SETTING_CHOICES = {"nameStyle": ["nick", "display", "username"],   # how people are named: server nickname, display name, username
                   "sensitivity": ["high", "normal", "low"], "background": ["show", "dim", "hide"],   # how the dashboard shows "bg" lines
                   "language": ["en"]}      # Earshot transcribes English only: every model is held to it
SENSITIVITY = {"high": 0.5, "normal": 1.0, "low": 2.0}     # x the speech threshold: high hears quieter voices
VAD_SENSITIVITY = {"high": 0.35, "normal": 0.5, "low": 0.65}  # speech probability the VAD must reach (Silero's default is 0.5)


PLUGINS_DIR = ROOT / "plugins"
PLUGIN_TYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
                ".json": "application/json", ".png": "image/png", ".svg": "image/svg+xml", ".webp": "image/webp", ".html": "text/html; charset=utf-8"}


def list_plugins() -> list[dict]:
    """Plugins: one folder each in plugins/, described by manifest.json. Off until turned on in Settings."""
    on = set(load_settings().get("plugins") or [])
    out = []
    if PLUGINS_DIR.is_dir():
        for d in sorted(PLUGINS_DIR.iterdir()):
            mf = d / "manifest.json"
            if not d.is_dir() or not mf.is_file() or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", d.name):
                continue
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
            except ValueError:
                continue
            out.append({"id": d.name, "name": str(m.get("name") or d.name)[:80], "version": str(m.get("version") or "")[:20],
                        "description": str(m.get("description") or "")[:300], "main": str(m.get("main") or "main.js"),
                        "style": m.get("style"), "enabled": d.name in on})
    return out


def plugin_file(rel: str):
    """A file inside plugins/<id>/, or None (never outside that folder)."""
    try:
        p = (PLUGINS_DIR / rel).resolve()
        p.relative_to(PLUGINS_DIR.resolve())
    except (ValueError, OSError):
        return None
    if not p.is_file() or p.suffix.lower() not in PLUGIN_TYPES:
        return None
    return p


def current_settings() -> dict:
    st = load_settings()
    out = {k: st.get(k, v) for k, v in SETTING_DEFAULTS.items()}
    out["language"] = "en"           # English only, whatever an older settings.json says
    return out


def set_setting(key: str, raw: str) -> dict:
    if key not in SETTING_DEFAULTS:
        return {"error": "unknown setting"}
    if isinstance(SETTING_DEFAULTS[key], bool):
        val = raw in ("1", "true", "on")
    else:
        val = raw if raw in SETTING_CHOICES.get(key, []) else None
        if val is None:
            return {"error": "not one of " + ", ".join(SETTING_CHOICES.get(key, []))}
    save_settings(**{key: val})
    return {"ok": True, key: val}


# ---------------------------------------------------------------- personality read
# A quick, for-fun personality sketch of one person from what they said in a
# call, written by a small local language model (Liquid AI LFM3-1.7B). Loaded
# on first use, unloaded after 10 idle minutes.
LLM_REPOS = {"mlx": "LiquidAI/LFM3-1.7B-MLX-8bit", "torch": "LiquidAI/LFM3-1.7B"}
LLM_NAME = "LFM3 1.7B"
PERSONA_SYSTEM = """You read how one person comes across in a voice call between friends playing a game, using only things they said. This is an impression of how they talk with others, not a medical or psychological diagnosis.

Rate each item from their lines. If the lines don't show something, give it a low score. Do not invent; base every rating and note on what they actually said, and quote short bits.
Call them only by the name you are given. Other names that come up in their lines belong to other people they are talking to or about, never to them.

Big Five, each 1 to 5 (3 is average):
- O openness: curious, imaginative vs. plain, routine
- C conscientiousness: organised, careful vs. careless, impulsive
- E extraversion: talkative, energetic vs. quiet, reserved
- A agreeableness: warm, cooperative vs. blunt, combative
- S emotional stability: calm, steady vs. tense, reactive

Interpersonal signals, each 0 to 3 (0 none, 1 a little, 2 clear, 3 strong):
- self-focus: centres talk on themselves, brags, wants admiration or praise, dismisses others (a narcissistic style)
- antagonism: hostile, contemptuous, insulting, puts people down
- manipulation: flatters or guilt-trips to get their way, says one thing and means another
- impulsiveness: reactive, volatile, blurts things out, can't wait
- warmth: notices others, supportive, includes people, kind

Reply with exactly these lines, a number then a short reason from their words (at most 12 words, no second thoughts), and nothing else:
SUMMARY: <one sentence on how they come across>
self-focus: <0-3> <evidence>
antagonism: <0-3> <evidence>
manipulation: <0-3> <evidence>
impulsiveness: <0-3> <evidence>
warmth: <0-3> <evidence>
O: <1-5> <reason>
C: <1-5> <reason>
E: <1-5> <reason>
A: <1-5> <reason>
S: <1-5> <reason>
QUOTE: <a short quote>
QUOTE: <a short quote>"""


PERSONA_SIGNALS = [("self-focus", "Self-focus (narcissistic style)"), ("antagonism", "Antagonism"),
                   ("manipulation", "Manipulation"), ("impulsiveness", "Impulsiveness"), ("warmth", "Warmth")]
PERSONA_BIG5 = [("O", "Openness"), ("C", "Conscientiousness"), ("E", "Extraversion"),
                ("A", "Agreeableness"), ("S", "Emotional stability")]
_SIG_TRAIT = {"self-focus": "self-focused", "antagonism": "combative", "manipulation": "calculating",
              "impulsiveness": "impulsive", "warmth": "warm"}


def _persona_parse(raw: str):
    summary, big5, signals, quotes = "", {}, {}, []
    for line in (raw or "").splitlines():
        t = line.strip().strip("-*• ").strip()
        m = re.match(r"(?i)^summary\s*[:\-]\s*(.+)$", t)
        if m and not summary:
            summary = m.group(1).strip(); continue
        m = re.match(r"(?i)^quote\s*[:\-]\s*(.+)$", t)
        if m:
            q = m.group(1).strip().strip('"“”\'').strip()
            if q and q != "<a short quote>" and len(quotes) < 3:
                quotes.append(q)
            continue
        m = re.match(r"(?i)^(emotional stability|[OCEAS])\s*[:\-]\s*([1-5])\b[.\)\s]*(.*)$", t)
        if m:
            k = "S" if m.group(1).upper() not in ("O", "C", "E", "A", "S") else m.group(1).upper()
            big5[k] = (int(m.group(2)), m.group(3).strip()); continue
        m = re.match(r"(?i)^(self[\s-]?focus|antagonism|manipulation|impulsiveness|warmth)\s*[:\-]\s*([0-3])\b[.\)\s]*(.*)$", t)
        if m:
            k = "self-focus" if m.group(1).lower().replace(" ", "").replace("-", "") == "selffocus" else m.group(1).lower()
            signals[k] = (int(m.group(2)), m.group(3).strip()); continue
    return summary, big5, signals, quotes


def _persona_clean(ev: str) -> str:
    """Tidy a model's reason: drop a left-over <0-3>/<1-5> placeholder, a (12) line
    reference, a leading dash/colon/bracket, and cap the length."""
    ev = (ev or "").strip()
    ev = re.sub(r"^[\s\-–—•*:>]+", "", ev)            # leading dashes, colons, bullets
    ev = re.sub(r"^<?\s*[0-9]\s*-\s*[0-9]\s*>\s*", "", ev)   # <0-3>, 0-3>, <1-5> ...
    ev = re.sub(r"^<[^>]*>\s*", "", ev)               # any other leading <tag>
    ev = re.sub(r"^\([0-9]+\)[\s:]*", "", ev)         # (12) line reference
    ev = re.sub(r"^[\s\-–—•*:<>]+", "", ev).strip()   # whatever leading punctuation is left
    return (ev[:157].rstrip() + "…") if len(ev) > 160 else ev


def persona_build(raw: str) -> dict:
    """Turn the model's rated lines into structured data for the report graphic
    (and a plain-text version for Copy and as a fallback). Signals are ranked in
    code, so the order never depends on the small model."""
    summary, big5, signals, quotes = _persona_parse(raw)
    summary = re.sub(r"\s+\([^()]{1,40}\)(?=\s+(?:is|comes|seems|sounds|was)\b)", "", summary, count=1)   # "Jo(h)n (Gator) comes across" -> "Jo(h)n comes across"
    if not big5 and not signals:
        return {"text": (raw or "").strip()[:1500], "data": None}   # unparseable: show what came back

    b = lambda k: big5.get(k, (3, ""))[0]
    traits = [_SIG_TRAIT[k] for k, _ in PERSONA_SIGNALS if signals.get(k, (0, ""))[0] >= 2]
    for cond, word in [(b("E") >= 4, "outgoing"), (b("E") <= 2, "reserved"), (b("A") >= 4, "easygoing"),
                       (b("A") <= 2, "blunt"), (b("C") >= 4, "organised"), (b("S") <= 2, "tense"),
                       (b("S") >= 4, "steady"), (b("O") >= 4, "curious")]:
        if cond and word not in traits and "combative" not in (traits if word == "blunt" else []):
            traits.append(word)
    traits = traits[:3]

    labels = dict(PERSONA_SIGNALS)
    # every signal in the graphic (self-focus/narcissism always visible), strongest first
    sig = [{"key": k, "label": labels[k], "score": signals[k][0] if k in signals else None,
            "note": _persona_clean(signals.get(k, (0, ""))[1]), "good": k == "warmth"}
           for k, _ in PERSONA_SIGNALS]                     # (a signal the model didn't rate is None: "not rated")
    sig.sort(key=lambda s: (s["good"], s["score"] is None, -(s["score"] or 0)))   # warmth last; otherwise strongest first
    big = [{"key": k, "label": lbl, "score": big5[k][0], "note": _persona_clean(big5[k][1])}
           for k, lbl in PERSONA_BIG5 if k in big5]
    data = {"summary": summary, "traits": traits, "signals": sig, "big5": big, "quotes": quotes[:3]}

    out = []
    if summary:
        out.append(f"**In a sentence:** {summary}")
    if traits:
        out.append(f"**Comes across as:** {', '.join(traits)}")
    shown = [s for s in sig if s["key"] == "self-focus" or (s["score"] or 0) >= 1]
    if shown:
        out.append("**Signals (0 = none, 3 = strong):**")
        for s in shown:
            if s["score"] is None:
                out.append(f"- {s['label']}: not rated"); continue
            bar = "●" * s["score"] + "○" * (3 - s["score"])
            out.append(f"- {s['label']}: {bar} {s['score']}/3{(': ' + s['note']) if s['note'] else ''}")
    if big:
        out.append("**Big Five (1-5):**")
        out.extend(f"- {x['label']}: {x['score']}/5{(': ' + x['note']) if x['note'] else ''}" for x in big)
    if quotes:
        out.append("**In their words:**")
        out.extend(f'- "{q}"' for q in quotes[:3])
    return {"text": "\n".join(out), "data": data}


class PersonalityReader:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = self.proc = None
        self.engine = None
        self.last_used = 0.0
        self.jobs: dict[str, dict] = {}
        self.cache: dict[tuple, tuple[int, str]] = {}    # (day, call start, ch, who) -> (words, job id)
        threading.Thread(target=self._idle_unload, daemon=True).start()

    _engine_lock = threading.Lock()
    _engine_ok: tuple | None = None

    @classmethod
    def engine_here(cls) -> tuple[str | None, str]:
        # One check at a time, remembered once it works: two threads importing the
        # library at once could see it half-loaded and wrongly report it missing.
        if cls._engine_ok:
            return cls._engine_ok
        with cls._engine_lock:
            if cls._engine_ok:
                return cls._engine_ok
            res = cls._find_engine()
            if res[0]:
                cls._engine_ok = res
            return res

    @staticmethod
    def _find_engine() -> tuple[str | None, str]:
        import platform
        if sys.platform == "darwin" and platform.machine() == "arm64":
            try:
                import mlx_vlm  # noqa: F401
                return "mlx", ""
            except ImportError:
                return None, "Install the Mac add-on: cd server && uv sync"
        try:
            import torch
            if torch.cuda.is_available():
                return "torch", ""
        except Exception:
            pass
        return None, "Needs an Apple Silicon Mac or an NVIDIA graphics card."

    def info(self) -> dict:
        eng, why = self.engine_here()
        return {"available": bool(eng), "engine": eng, "model": LLM_NAME, "reason": why, "loaded": self.model is not None}

    def _load(self, eng: str):
        if self.model is not None:
            return
        repo = LLM_REPOS[eng]
        print(f"Personality read: loading {repo} …", flush=True)
        if eng == "mlx":
            from mlx_vlm import load
            self.model, self.proc = load(repo)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.proc = AutoTokenizer.from_pretrained(repo)
            self.model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to("cuda").eval()
        self.engine = eng

    def _generate(self, messages: list[dict], max_tokens: int = 420, temperature: float = 0.3) -> str:
        if self.engine == "mlx":
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(self.proc, self.model.config, messages, num_images=0)
            out = generate(self.model, self.proc, prompt, max_tokens=max_tokens, temperature=temperature, repetition_penalty=1.1, verbose=False)
            return (out.text if hasattr(out, "text") else str(out)).strip()
        import torch
        ids = self.proc.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                            return_dict=True, enable_thinking=False).to("cuda")
        with torch.inference_mode():
            gen = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-3), top_p=0.9,
                                      repetition_penalty=1.05)
        return self.proc.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def think_then_answer(self, messages: list[dict], prefix: str = "", think_tokens: int = 350, answer_tokens: int = 16) -> str:
        """LFM3 reasons inside <think>...</think> before it answers. Let it think for
        at most think_tokens, close the thinking, and start its answer with prefix so a
        small model writes the answer itself rather than more thinking. Returns prefix + answer."""
        if self.engine == "mlx":
            from mlx_vlm import generate
            from mlx_vlm.prompt_utils import apply_chat_template
            prompt = apply_chat_template(self.proc, self.model.config, messages, num_images=0, enable_thinking=True)
            gen = lambda p, n: (lambda o: o.text if hasattr(o, "text") else str(o))(
                generate(self.model, self.proc, p, max_tokens=n, temperature=0.0, repetition_penalty=1.1, verbose=False))
        else:
            import torch
            prompt = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=True)
            def gen(p, n):
                ids = self.proc(p, return_tensors="pt", add_special_tokens=False).to("cuda")
                with torch.inference_mode():
                    out = self.model.generate(**ids, max_new_tokens=n, do_sample=False, repetition_penalty=1.05)
                return self.proc.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        if prompt.rstrip().endswith("<think></think>"):          # template turned thinking off: turn it on
            prompt = prompt.rstrip()[:-len("</think>")]
        thought = gen(prompt, think_tokens).split("</think>", 1)[0]
        return prefix + gen(prompt + thought + "\n</think>\n" + prefix, answer_tokens)

    def _idle_unload(self):
        while True:
            time.sleep(30)
            with self.lock:
                if self.model is not None and time.time() - self.last_used > 600:
                    self.model = self.proc = None
                    gc.collect()
                    print("Personality read: model unloaded (idle).", flush=True)

    def start(self, day: str, t0: int, t1: int, ch: str, user: str, name: str, fresh: bool = False, parts: list | None = None) -> dict:
        if not current_settings()["personality"]:
            return {"error": "Personality reads are turned off in Settings."}
        eng, why = self.engine_here()
        if not eng:
            return {"error": why}
        recs = [r for d in day.split(",") if d for r in day_data(d)[0]]     # a merged call can span days
        inside = (lambda r: any(a <= r["t"] <= b and (not c or not r.get("ch") or r["ch"] == c) for _, a, b, c in parts)) if parts \
            else (lambda r: t0 <= r["t"] <= t1 and (not ch or not r.get("ch") or r.get("ch") == ch))
        mine = [r for r in recs if inside(r) and not r.get("bg")
                and ((user and r.get("user") == user) or (not user and r.get("speaker") == name))]
        said = [r["text"].strip() for r in mine if r.get("text", "").strip()]
        words = sum(len(x.split()) for x in said)
        if len(said) < 4 or words < 30:
            return {"error": f"Not enough to go on yet: {name} has said {words} words in this call. Try again once they've talked a bit more."}
        picked, total = [], 0                      # most recent lines, up to ~2500 words
        for x in reversed(said):
            total += len(x.split())
            if total > 2500:
                break
            picked.append(x)
        picked.reverse()
        # Reuse the last read of this person in this call; redo it every 10 minutes while they keep talking.
        ck = (day, t0, ch, user or name)
        hit = self.cache.get(ck)
        stale = hit and time.time() - self.jobs.get(hit[1], {}).get("started", 0) >= 600 and words - hit[0] >= 20
        if hit and not fresh and not stale:
            old = self.jobs.get(hit[1])
            if old and old["status"] in ("loading", "thinking", "done"):
                return {"id": old["id"], **{k: old[k] for k in ("status", "lines", "words")}, "cached": True}
        jid = os.urandom(6).hex()
        self.cache[ck] = (words, jid)
        job = {"id": jid, "status": "loading" if self.model is None else "thinking", "name": name,
               "lines": len(picked), "words": sum(len(x.split()) for x in picked), "model": LLM_NAME, "started": time.time()}
        self.jobs[jid] = job
        messages = [{"role": "system", "content": PERSONA_SYSTEM},
                    {"role": "user", "content": f"Name: {name}\nThings {name} said in this call, oldest first:\n"
                                                + "\n".join("- " + x for x in picked)}]

        def run():
            try:
                with self.lock:
                    self._load(eng)
                    job["status"] = "thinking"
                    raw = self.think_then_answer(messages, prefix="SUMMARY:", think_tokens=420, answer_tokens=800)
                    self.last_used = time.time()
                built = persona_build(raw)
                job.update(status="done", text=built["text"], data=built["data"], seconds=round(time.time() - job["started"], 1))
            except Exception as e:
                job.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        threading.Thread(target=run, daemon=True).start()
        for old in [k for k, v in self.jobs.items() if time.time() - v["started"] > 3600]:
            self.jobs.pop(old, None)
        return {"id": jid, **{k: job[k] for k in ("status", "lines", "words")}}


# ---------------------------------------------------------------- questions and answers
# Which lines are real questions, and which later line (if any) answers each
# one, judged from what was said around it by the same small local model as
# personality reads (LFM3 1.7B), thinking it through before it answers. Results
# are kept per day in transcripts/<day>.qa.json as line id -> {"real", "answer"}.
QA_SYSTEM = """You read part of a voice-chat transcript between friends playing a game together. You get what was said before, one QUESTION line, and the numbered lines said after it.
1. Is the QUESTION a real question: the speaker asks someone else something and expects a reply? NOT real questions: rhetorical questions, jokes, exclamations, quoted speech, filler ("right?", "huh?"), and questions the speaker answers themselves.
2. If it is real, which numbered later line, said by a DIFFERENT person, answers it? It must respond to what was asked (a yes or no, a choice, the information asked for). Most later lines are not answers. If none answers it, the answer is null.
Think about what is being asked and who replies to it, then reply with JSON only: {"answer": number of the later line, or null}"""
QA_REAL = """You read a moment from a voice chat between friends playing a game. Is the QUESTION line a real question, where the speaker wants someone else to reply with an answer?
Real questions (answer yes) ask the others for information, a choice, help, or a yes or no, even when short or rough: "which fob are you in?", "can you hear me?", "who needs a hammer?", "what accent is that?", "is the grenade launcher support XP?".
Not real questions (answer no): exclamations and swearing ("what the hell?", "bro, what the fuck?"), rhetorical questions, jokes and teasing, quoted speech, filler ("right?", "huh?", "you know what?"), and questions the speaker answers themselves.
Think it through, then reply with one word: yes or no."""
QA_VERIFY = """You read two lines from a voice chat. Q is a question. Does line A answer Q: does it respond to what Q asked? A line that only follows Q, changes the subject, or answers something else does not.
Think it through, then reply with one word: yes or no."""
QA_AFTER = 10           # an answer is looked for in the next 10 lines; none of them answers it: "no answer"
QA_WAIT_S = 45          # or, in a quiet stretch, once 45 s have passed


def is_question_text(text: str) -> bool:
    t = (text or "").strip().rstrip("\"')]” ").strip()
    return bool(re.search(r"\?[!?]*$", t)) and len(t.split()) >= 2     # "?", "??", "?!" like the dashboard


class QAJudge:
    def __init__(self, reader: "PersonalityReader"):
        self.reader = reader                          # shares its model, lock and idle unloading
        self.todo: list[tuple[str, str]] = []         # (day, line id), newest first
        self.pending: set[tuple[str, str]] = set()
        self.res: dict[str, dict] = {}                # day -> {line id: {"real", "answer"}}
        self.cv = threading.Condition()
        self.error = None
        self.fails = 0                                # model errors in a row
        self.off_until = 0.0                          # after repeated errors, rest a while
        self.tries: dict[tuple[str, str], int] = {}   # unclear replies per question
        self.wanted: dict[tuple[str, str], float] = {}   # when the dashboard last asked about it
        threading.Thread(target=self._worker, daemon=True).start()

    def available(self) -> bool:
        return (current_settings().get("qaLinks", True) and self.reader.engine_here()[0] is not None
                and time.time() >= self.off_until)

    def _path(self, day: str) -> Path:
        return TRANSCRIPTS / f"{day}.qa.json"

    def day_results(self, day: str) -> dict:
        if day not in self.res:
            try:
                self.res[day] = json.loads(self._path(day).read_text(encoding="utf-8"))
            except Exception:
                self.res[day] = {}
        return self.res[day]

    def request(self, days: list[str], t0: int, t1: int, ch: str, parts: list | None = None) -> dict:
        """What is known for the questions in this stretch; queues the rest."""
        judged, waiting, now = {}, 0, time.time() * 1000
        ok = self.available()
        spans = parts or [(d, t0, t1, ch) for d in days]
        for d in {sp[0] for sp in spans}:
            got = self.day_results(d)
            recs = day_data(d)[0]
            ids = {r.get("id") for r in recs}
            for r in recs:
                if r.get("bg") or not r.get("id") or not is_question_text(r.get("text", "")):
                    continue
                if not any(sd == d and a <= r["t"] <= b and (not c or not r.get("ch") or r["ch"] == c) for sd, a, b, c in spans):
                    continue
                res = got.get(r["id"])
                if res and res.get("answer") and res["answer"] not in ids:
                    got.pop(r["id"], None); res = None           # its answer was rewritten since: judge again
                if res:
                    judged[r["id"]] = res
                elif ok:
                    waiting += 1
                    key = (d, r["id"])
                    self.wanted[key] = time.time()
                    later = sum(1 for x in recs if x["t"] > r["t"] and not x.get("bg") and (not r.get("ch") or not x.get("ch") or x["ch"] == r["ch"]))
                    if later >= QA_AFTER or now - r["end"] > QA_WAIT_S * 1000:
                        with self.cv:
                            if key not in self.pending and len(self.todo) < 400:
                                self.pending.add(key)
                                self.todo.insert(0, key)
                                self.cv.notify()
        return {"available": ok, "model": LLM_NAME, "judged": judged, "waiting": waiting, "error": self.error}

    def _think(self, system: str, user: str, prefix: str = "Answer (yes or no):") -> str:
        """The model's answer after a short, capped stretch of thinking (the thinking is dropped)."""
        return self.reader.think_then_answer([{"role": "system", "content": system}, {"role": "user", "content": user}], prefix)

    def judge(self, day: str, qid: str) -> dict | None:
        recs = [r for r in day_data(day)[0] if not r.get("bg") and r.get("text")]
        qi = next((i for i, r in enumerate(recs) if r.get("id") == qid), None)
        if qi is None:
            return None
        q = recs[qi]
        same = lambda r: not q.get("ch") or not r.get("ch") or r["ch"] == q["ch"]
        before = [r for r in recs[max(0, qi - 30):qi] if same(r) and q["t"] - r["t"] < 60000][-6:]
        # what was said after it: the next 10 lines, including a reply that started a moment
        # before the question ended (people answer while the asker is still finishing)
        after = [r for r in recs[max(0, qi - 6):qi + 40] if r is not q and same(r) and r["t"] - q["t"] < 45000
                 and (r["t"] > q["t"] or (r["t"] >= q["t"] - 2000 and r["end"] > q["end"]))][:QA_AFTER]
        before = [r for r in before if r not in after]
        text = ("Said before:\n" + "\n".join(f"{r['speaker']}: {r['text']}" for r in before) + "\n\n" if before else "") \
            + f"QUESTION ({q['speaker']}): {q['text']}\n\nSaid after, numbered:\n" \
            + ("\n".join(f"{n}. {r['speaker']}: {r['text']}" for n, r in enumerate(after, 1)) or "(nothing)")
        # 1. a real question at all? Clear cases by rule, the rest by the model (its own
        #    step: easier for a small model than deciding everything at once)
        words = re.findall(r"[a-z']+", q["text"].lower())
        if len(words) <= 6 and re.search(r"\b(the (hell|fuck|heck)|wtf)\b", q["text"].lower()):
            return {"real": False, "answer": None}               # "what the hell?", "bro, what the fuck?"
        asks_others = bool(set(words) & {"you", "your", "you're", "y'all", "someone", "somebody", "anyone", "anybody",
                                          "everyone", "guys", "we", "we're", "us"})
        near = "\n".join(f"{r['speaker']}: {r['text']}" for r in before[-3:])
        soon = "\n".join(f"{r['speaker']}: {r['text']}" for r in after[:3])
        if not asks_others:
            v = self._think(QA_REAL, (f"Before:\n{near}\n\n" if near else "") + f"QUESTION ({q['speaker']}): {q['text']}"
                            + (f"\n\nAfter:\n{soon}" if soon else ""))
            yn = re.findall(r"\b(yes|no)\b", v.lower())
        if not asks_others and yn and yn[0] == "no":
            return {"real": False, "answer": None}
        # 2. which later line answers it
        out = self._think(QA_SYSTEM, text, prefix='{"answer":')
        m = re.match(r'\{"answer":\s*(\d+|null|none)', out, re.I)
        if not m:
            return {"unsure": True}                     # unclear reply: not saved, tried again
        j = {"answer": int(m.group(1)) if m.group(1).isdigit() else None}
        real = True                         # settled in step 1; this step only picks the answer
        ans = j.get("answer")
        a = after[ans - 1] if isinstance(ans, int) and 1 <= ans <= len(after) else None
        who = lambda r: r.get("user") or r.get("speaker")
        # an answer comes after the question, from someone else, and responds to it
        if a is not None and (a["end"] <= q["end"] or who(a) == who(q)):
            a = None
        if a is not None:
            v = self._think(QA_VERIFY, f"Q ({q['speaker']}): {q['text']}\nA ({a['speaker']}): {a['text']}")
            first = re.findall(r"\b(yes|no)\b", v.lower())
            if not first or first[0] != "yes":
                a = None
        return {"real": real, "answer": a["id"] if a is not None else None}

    def _worker(self):
        while True:
            with self.cv:
                while not self.todo:
                    self.cv.wait(timeout=30)
                day, qid = self.todo.pop(0)
            if time.time() - self.wanted.get((day, qid), 0) > 300:    # nobody is looking at that call any more
                with self.cv:
                    self.pending.discard((day, qid))
                continue
            try:
                eng, why = self.reader.engine_here()
                if not eng:
                    raise RuntimeError(why)
                with self.reader.lock:                 # one model, shared with personality reads
                    self.reader._load(eng)
                    res = self.judge(day, qid)
                    self.reader.last_used = time.time()
                self.error, self.fails = None, 0
                if res and res.get("unsure"):
                    n = self.tries[(day, qid)] = self.tries.get((day, qid), 0) + 1
                    res = {"real": True, "answer": None} if n >= 2 else None     # twice unclear: call it unanswered
                if res is not None:
                    got = self.day_results(day)
                    got[qid] = res
                    tmp = self._path(day).with_suffix(".json.tmp")
                    tmp.write_text(json.dumps(got), encoding="utf-8"); tmp.replace(self._path(day))
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"[:200]
                self.fails += 1
                print("Questions and answers:", self.error, flush=True)
                if self.fails >= 3:                        # keep failing: stop for 5 minutes, say so
                    self.off_until = time.time() + 300
                    with self.cv:
                        self.todo.clear(); self.pending.clear()
                    self.fails = 0
                time.sleep(5)
            finally:
                with self.cv:
                    self.pending.discard((day, qid))


def pick_engine(requested: str) -> str:
    if requested != "auto":
        return requested
    import platform
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "mlx"
    try:
        import torch
        if torch.cuda.is_available():
            return "torch"
    except Exception:
        pass
    return "onnx"


# ---------------------------------------------------------------- timing
# One utterance can hold several sentences while someone else replies in
# between. Split it at sentence ends and pauses, and put each part at the time
# it was actually said, so the transcript interleaves who said what when.
ENGLISH_SYMBOLS = set("£€°’‘“”—–…")              # non-ASCII that English text still uses


def english_mask(tokenizer, size: int, symbols: bool = False, ban: set | None = None):
    """Tokens a model may emit when the call is in English: only ones made of
    plain ASCII, so it can't drift into another language's words or script.
    symbols=True also allows tokens that decode to ASCII plus ENGLISH_SYMBOLS
    (currency, degrees, curly quotes, dashes), for the 2B that writes them."""
    ok = np.zeros(size, dtype=bool)
    markers = str.maketrans("", "", "\u2581\u0120\u010a")      # SentencePiece / byte-BPE space and newline marks
    name = getattr(tokenizer, "convert_ids_to_tokens", None) or getattr(tokenizer, "id_to_token")
    for i in range(size):
        try:
            t = name(i)
        except Exception:
            t = None
        if t is None:
            continue
        raw = str(t)
        t = raw.translate(markers)
        ok[i] = all(ord(c) < 128 for c in t)
        if ban and ok[i]:
            # a whole foreign word: after a space mark, or capitalised (the first word of a line);
            # a lower-case piece with no space mark is the middle of a word ("chem|ist") and stays
            w = t.strip().lower()
            if w in ban and (raw[:1] in "\u2581\u0120" or t.strip().istitle()):
                ok[i] = False
                continue
        if symbols and not ok[i]:
            # byte-level BPE names these by their bytes ("Â£"): look at the decoded text
            # (a piece of a multi-byte character decodes to U+FFFD and stays forbidden)
            try:
                d = tokenizer.decode([i])
            except Exception:
                continue
            # (only tokens with a symbol in them: not tabs and control bytes, which the byte names forbid)
            ok[i] = any(c in ENGLISH_SYMBOLS for c in d) and all(32 <= ord(c) < 127 or c in ENGLISH_SYMBOLS for c in d)
            if ban and ok[i] and d.strip().lower() in ban:
                ok[i] = False
    return ok


def en_token_cap(audio_len: int) -> int:
    """Most tokens the 2B may write, English only, for this much audio. English
    speech is under 5 tokens a second (AMI meetings: at most 4.9); with the mask
    on, speech in another script can make the model repeat itself until it runs
    out of tokens (a minute on a 10 s clip), so stop well before that."""
    return int(10 * audio_len / SR) + 20


def looping(ids) -> bool:
    """True when the output ends in the same 1-4 tokens 8 times over ("3 3 3 ...")."""
    for p in range(1, 5):
        if len(ids) >= 8 * p and all(ids[-k] == ids[-k - p] for k in range(1, 7 * p + 1)):
            return True
    return False


def ctc_pieces(ids, audio_len: int, decode, blank: int) -> list[tuple[str, float, float]]:
    """Greedy CTC frames -> [(text added by this token, start s, end s)]."""
    n = len(ids)
    spf = audio_len / SR / max(1, n)                  # seconds per output frame
    toks, prev = [], None
    for i, t in enumerate(ids):
        t = int(t)
        if t != blank and t != prev:
            toks.append([t, i, i])
        elif t != blank and t == prev and toks:
            toks[-1][2] = i
        prev = t
    out, before = [], ""
    for k in range(len(toks)):
        now = decode([x[0] for x in toks[:k + 1]])
        out.append((now[len(before):] if now.startswith(before) else now[len(before):], toks[k][1] * spf, (toks[k][2] + 1) * spf))
        before = now
    return out


def split_timed(pieces: list[tuple[str, float, float]], pause_s: float = 0.6) -> list[tuple[str, float, float]]:
    """Group timed pieces into parts at sentence ends and pauses."""
    parts, buf, start, end = [], "", None, 0.0
    for i, (txt, a, b) in enumerate(pieces):
        nxt = pieces[i + 1] if i + 1 < len(pieces) else None
        if start is None and txt.strip():
            start = a
        buf += txt; end = b
        sentence_end = buf.rstrip().endswith((".", "?", "!")) and (nxt is None or nxt[0][:1].isspace())
        pause = nxt is not None and nxt[1] - b > pause_s and nxt[0][:1].isspace()
        if (sentence_end or pause or nxt is None) and buf.strip():
            parts.append([buf.strip(), start if start is not None else a, end]); buf, start = "", None
    merged = []                                       # don't leave one-word fragments on their own
    for p in parts:
        if merged and len(p[0].split()) < 2 and not p[0].endswith(("?", "!")):
            merged[-1][0] += " " + p[0]; merged[-1][2] = p[2]
        else:
            merged.append(p)
    return [tuple(p) for p in merged]


def split_by_text(text: str, audio_s: float) -> list[tuple[str, float, float]]:
    """No timing from the model: split at sentence ends, time by share of characters."""
    sents = [x for x in re.split(r"(?<=[.?!])\s+", text.strip()) if x]
    if len(sents) <= 1:
        return [(text.strip(), 0.0, audio_s)]
    total, pos, out = sum(len(x) for x in sents), 0, []
    for x in sents:
        a = audio_s * pos / total; pos += len(x)
        out.append((x, a, audio_s * pos / total))
    merged = []
    for p in out:
        if merged and len(p[0].split()) < 2:
            merged[-1] = (merged[-1][0] + " " + p[0], merged[-1][1], p[2])
        else:
            merged.append(p)
    return merged


def to_clock(marks: list[tuple[int, float]], sec: float) -> float:
    """Audio seconds within an utterance -> clock ms (skipped silence restored)."""
    if not marks:
        return 0.0
    pos = sec * SR
    best = marks[0]
    for m in marks:
        if m[0] <= pos:
            best = m
        else:
            break
    return best[1] + (pos - best[0]) / SR * 1000


# The token masks keep other scripts and accents out, but Spanish, French or
# German speech can still come out as unaccented words in that language. Those
# lines are dropped: a line counts as another language when its common short
# words are clearly that language's ("que", "la", "und", "die", "est"...) and
# not English's. Few such words, or as many English ones, and the line stays.
EN_COMMON = set("""the a an and or but to of in on at for with is are was were be been it its this that these those
i you he she we they me him her us them my your our their do does did not no yes so if then than there here what
who how why when where can could would should will just like about have has had get got go going know think yeah
okay ok right well oh uh um hey up out all one some any my as by from""".split())
FOREIGN_COMMON = set("""que la las los el lo una unos del al por para con como pero porque es esta este estoy estas
muy hay yo tu nosotros vamos todos bueno gracias hola si ahora tambien donde cuando
le les des du au aux une est sont avec pour pas mais je tu il elle nous vous ils sur dans tres bien merci bonjour oui
der die das und ist nicht ein eine einen mit auf fur ich du wir sie es sind auch aber noch nur wie was wenn dann
ja nein danke guten morgen bitte zu von dem den im
di il gli della che sono per non una questo ciao grazie
de het een en niet ik je wij zijn van voor met op dat ook
nao sim obrigado voce com uma para mas muito""".split()) - EN_COMMON


# Words Granite 2B is kept from writing as whole words (a strong push, not a
# guarantee: a word can still come out in pieces; not_english() catches the rest):
# common German, Spanish, French, Italian, Dutch and Portuguese words. With them
# blocked, speech the model half-hears as another language comes out as the
# English words it most likely was.
BAN_WORDS = set("""
nicht ist und ich wir sind auch aber noch wenn dann nein danke bitte heute wieder mein meine meinen dein deine seine ihre
unser unsere euch mich dich sich ihm ihn ihnen haben habe hast hatte hatten wird werden wurde wurden kann kannst können
konnte muss musst müssen soll sollen willst wollen mehr sehr schon jetzt hier dort immer oder weil dass doch nach über
zwischen gegen ohne durch für zum zur beim alles nichts etwas viel viele gute guten morgen abend nacht jahr leute machen
mache macht gemacht gehen gehe geht ging kommen komme kommt sehen sehe sieht sagen sage sagt gesagt wissen weiß weiss
glaube denke geben gibt nehmen bleiben warum wieso woher wohin welche welcher welches einem einer eines keine kein keinen
genau stimmt richtig wirklich vielleicht natürlich naturlich eigentlich bisschen ganz gerade erst schnell rechts oben
unten draußen draussen hallo scheiße scheisse verdammt geil komm achtung hilfe ach nee jawohl gesehen
está estás estoy están eso esto pero porque cuando donde dónde cómo muy también tambien ahora hola gracias bueno buena
vamos todos nada nosotros ellos ellas usted ustedes tengo tienes tiene hacer hace puedo puede quiero quieres sabes creo
mucho poco aquí aqui allí entonces siempre nunca hoy mañana manana noche verdad claro oye pues contigo conmigo qué quién
avec pour mais nous vous elle très bien merci bonjour oui sont voilà alors aussi encore toujours jamais maintenant demain
beaucoup rien chose quoi pourquoi comment parce faire avoir être suis sommes êtes peux veux
questo questa sono della grazie ciao perché anche molto adesso ancora sempre niente cosa allora andiamo dove quando
niet het een zijn voor ook maar nog wel heel goed dank waarom wanneer
não você obrigado obrigada muito então agora isso isto""".split())
# ...minus ones that are also English words, names or places, or the start of
# common English words in the model's vocabulary ("Mich|ael", "ach|e", "nach|os")
BAN_WORDS -= {"immer", "nach", "ach", "erst", "und", "muss", "mich", "nada", "mucho", "vamos", "bueno",
              "comment", "pour", "sage", "chose", "dove", "encore", "heel", "dank", "nee", "elle", "agora", "della",
              "poco", "morgen", "claro", "hoy", "nog", "seine", "ging", "het"}
FOREIGN_COMMON |= BAN_WORDS - EN_COMMON
# words that also turn up in English chat (gaming slang, names, loans) never count as foreign
ENGLISHY = {"op", "die", "per", "con", "met", "van", "al", "lo", "la", "sim", "het", "dat", "dem", "den", "so", "me", "mi",
            "yo", "si", "en", "de", "du", "des", "le", "il", "es", "tu", "ja", "na", "wo", "am", "ok", "nada", "ciao",
            "hola", "merci", "gracias", "mucho", "bueno", "amigo", "vamos", "pronto", "bien", "oui", "una", "el", "los"}
FOREIGN_COMMON -= ENGLISHY - {"ja", "oui", "bien"}       # still foreign on a line of their own (see SHORT_FOREIGN)


# one- or two-word lines ("Ja.", "Es ist...", "Oui") are dropped when every word is
# one of these; words that are also English slang or English words are left out
SHORT_FOREIGN = (FOREIGN_COMMON | {"ja", "oui", "bien"}) - (ENGLISHY - {"ja", "oui", "bien"}) - {"via", "mit"}


# How common each word is in English and six other languages (server/langwords.txt.gz,
# built by tools/build_langwords.py from wordfreq data). A line is in another
# language when its words, all together, fit that language clearly better than
# English: laughter and filler ("ha", "um", "no") count for nothing.
LANG_NEUTRAL = set("ha hah haha hahaha hahahaha he hehe heh hi ho hey oh ah aha uh um umm hm hmm mm mhm no nope yeah yep yup "
                   "yes ok okay bah ugh eh huh wow yo bro man sus gg lol lmao da ta la na o a i".split())
_langwords: dict | None = None


def langwords() -> dict:
    global _langwords
    if _langwords is None:
        _langwords = {}
        try:
            import gzip
            with gzip.open(Path(__file__).with_name("langwords.txt.gz"), "rt", encoding="utf-8") as f:
                for line in f:
                    if not line.startswith("#"):
                        w, *v = line.split()
                        _langwords[w] = [float(x) for x in v]
        except Exception as e:
            print("Language data missing, using the word lists only:", e, flush=True)
    return _langwords


def foreign_score(text: str) -> tuple[float, int]:
    """(how much better the best other language fits than English, known words)"""
    z = langwords()
    ws = [w for w in re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)?", text.lower()) if w not in LANG_NEUTRAL]
    tot, known = [0.0] * 7, 0
    for w in ws:
        v = z.get(w)
        if v:
            known += 1
            for i in range(7):
                tot[i] += max(v[i], 1.0)
    return (max(tot[1:]) - tot[0], known) if known else (0.0, 0)


def not_english(text: str) -> bool:
    if langwords():
        d, known = foreign_score(text)
        return d >= 2.5 and d >= 0.8 * known       # tuned on real calls: catches foreign lines, keeps English and filler
    words = re.findall(r"[a-z']+", text.lower())
    if len(words) < 3:
        return bool(words) and all(w in SHORT_FOREIGN for w in words)
    f = sum(w in FOREIGN_COMMON for w in words)
    e = sum(w in EN_COMMON for w in words)
    distinct = len({w for w in words if w in FOREIGN_COMMON})
    if distinct >= 3 and f >= 0.25 * len(words) and f > e:   # plenty of another language, even mixed with some English
        return True
    return distinct >= 2 and f > 2 * e and f >= 0.25 * len(words)


def load_engine(engine: str, repo: str, language: str | None):
    """Returns run(audio float32 16 kHz) -> text (or {"text", "pieces"}), English only."""
    run = _load_engine(engine, repo, language)
    def english_run(audio):
        out = run(audio)
        text = out["text"] if isinstance(out, dict) else out
        if text and not_english(text):
            print(f"Not English, dropped: {text[:60]!r}", flush=True)
            return {"text": "", "pieces": [], "foreign": text} if isinstance(out, dict) else ""
        return out
    return english_run


def _load_engine(engine: str, repo: str, language: str | None):
    if "granite" in repo.lower() or "parakeet" in repo.lower():
        language = "en"          # Earshot's models only ever write English
    if engine == "mlx" and "whisper" in repo.lower():      # Whisper: its own MLX package
        import mlx_whisper
        kw = {"language": language} if language else {}
        return lambda audio: mlx_whisper.transcribe(audio, path_or_hf_repo=repo, condition_on_previous_text=False, **kw)["text"]
    if engine == "mlx":
        import mlx.core as mx
        from mlx_audio.stt import load
        model = load(repo)
        kw = {"language": language} if language and ("whisper" in repo.lower() or "qwen" in repo.lower()) else {}
        if "granite-speech" in repo.lower() and "turboctc" not in repo.lower():      # Granite 2B
            kw = {"punctuate": True}      # (a language hint made 2B less accurate on English: 6.7% vs 5.3% words wrong)
        if "turboctc" in repo.lower():                   # Granite 470M (CTC): word timing from its frames
            feat = sys.modules[type(model).__module__].compute_features
            blank = model.config.pad_token_id
            tok = model._tokenizer
            mask = None
            def run_ctc(audio):
                nonlocal mask
                x = feat(model._load_audio(mx.array(audio)), num_mel_bins=model.config.encoder_config.num_mel_bins)[None]
                logits = model(x)[0]
                if language == "en":                     # English only: never pick another language's tokens
                    if mask is None:
                        m = english_mask(tok, logits.shape[-1]); m[blank] = True
                        mask = mx.array(np.where(m, 0.0, -1e9).astype(np.float32))
                    logits = logits + mask
                ids = np.array(mx.argmax(logits, axis=-1))
                pieces = ctc_pieces(ids, len(audio), lambda t: tok.decode(t, skip_special_tokens=True), blank)
                return {"text": "".join(p[0] for p in pieces).strip(), "pieces": pieces}
            return run_ctc
        if "granite-speech" in repo.lower() and language == "en":      # Granite 2B, English only
            # model.generate takes no logits processors, so this runs its own
            # (non-streaming, greedy) decoding steps with a mask that forbids
            # tokens which aren't plain ASCII: no other scripts or accented
            # letters (Latin-script words of another language can still slip
            # through, just without accents). Same prompt as generate().
            # When the speech isn't English the mask can make it repeat itself
            # instead of ending: such output is dropped (returns ""), see en_token_cap.
            from mlx_audio.lm.generate import generate_step
            gs = sys.modules[type(model).__module__]
            tok = model._tokenizer
            ask = gs._resolve_prompt("asr", None, None, punctuate=True)
            system = gs.PLUS_SYSTEM_PROMPT if model.is_plus else None
            mask = None
            def only_english(tokens, logits):
                nonlocal mask
                if mask is None:
                    m = english_mask(tok, logits.shape[-1], symbols=True, ban=BAN_WORDS)
                    for i in [tok.eos_token_id, *getattr(tok, "all_special_ids", [])]:
                        if i is not None and i < len(m):
                            m[i] = True
                    mask = mx.array(np.where(m, 0.0, -np.inf).astype(np.float32))
                return logits.astype(mx.float32) + mask
            def run_en(audio):
                feats, n_audio = model._extract_features(model._load_audio(mx.array(audio)))
                audio_feats = model.get_audio_features(feats)
                ids = model._build_prompt(n_audio, ask, system_prompt=system)
                embeds = model._build_inputs_embeds(ids, audio_feats)
                out, ended = [], False
                for t, _ in generate_step(prompt=ids, input_embeddings=embeds.squeeze(0), model=model,
                                          max_tokens=en_token_cap(len(audio)), logits_processors=[only_english]):
                    if t == tok.eos_token_id:
                        ended = True
                        break
                    out.append(int(t))
                    if looping(out):
                        break
                return tok.decode(out, skip_special_tokens=True) if ended else ""
            return run_en
        def run(audio):
            out = model.generate(mx.array(audio), **kw)
            return out.text if hasattr(out, "text") else out if isinstance(out, str) else ""
        return run
    if engine == "torch":
        import torch
        from transformers import AutoProcessor
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        half = torch.float16 if dev == "cuda" else torch.float32
        low = repo.lower()
        proc = AutoProcessor.from_pretrained(repo)
        def to_dev(inp, dtype):
            return {k: (v.to(dev, dtype=dtype) if v.is_floating_point() else v.to(dev)) if hasattr(v, "to") else v for k, v in inp.items()}
        if "turboctc" in low:                                # Granite 470M (CTC)
            from transformers import AutoModelForCTC
            model = AutoModelForCTC.from_pretrained(repo, dtype=half).to(dev).eval()
            blank = model.config.pad_token_id
            tk = getattr(proc, "tokenizer", proc)
            mask = None
            def run(audio):
                nonlocal mask
                with torch.inference_mode():
                    inp = to_dev(proc([audio], sampling_rate=SR), model.dtype)
                    logits = model(**inp).logits[0]
                    if language == "en":                 # English only: never pick another language's tokens
                        if mask is None:
                            m = english_mask(tk, logits.shape[-1]); m[blank] = True
                            mask = torch.tensor(np.where(m, 0.0, -1e4), dtype=logits.dtype, device=logits.device)
                        logits = logits + mask
                ids = logits.argmax(-1).cpu().numpy()
                pieces = ctc_pieces(ids, len(audio), lambda t: tk.decode(t, skip_special_tokens=True), blank)
                return {"text": "".join(p[0] for p in pieces).strip(), "pieces": pieces}
            return run
        if "granite-speech" in low:                          # Granite 2B (speech LLM)
            from transformers import AutoModelForSpeechSeq2Seq
            model = AutoModelForSpeechSeq2Seq.from_pretrained(repo, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
            tok = proc.tokenizer
            ask = "<|audio|>transcribe the speech with proper punctuation and capitalization."
            prompt = tok.apply_chat_template([{"role": "user", "content": ask}],
                                             tokenize=False, add_generation_prompt=True)
            gen = {}
            if language == "en":                         # English only: forbid tokens that aren't plain ASCII
                from transformers import LogitsProcessor, LogitsProcessorList
                class OnlyEnglish(LogitsProcessor):
                    bad = None
                    def __call__(self, input_ids, scores):
                        if self.bad is None:
                            m = english_mask(tok, scores.shape[-1], symbols=True, ban=BAN_WORDS)
                            for i in [tok.eos_token_id, *tok.all_special_ids]:
                                if i is not None and i < len(m):
                                    m[i] = True
                            self.bad = torch.from_numpy(~m).to(scores.device)
                        return scores.masked_fill(self.bad, -float("inf"))
                gen["logits_processor"] = LogitsProcessorList([OnlyEnglish()])
            def run(audio):
                with torch.inference_mode():
                    inp = proc(prompt, torch.from_numpy(audio).unsqueeze(0), device=dev, return_tensors="pt").to(dev)
                    cap = min(200, en_token_cap(len(audio))) if gen else 200
                    out = model.generate(**inp, max_new_tokens=cap, do_sample=False, num_beams=1, **gen)
                    new = out[0, inp["input_ids"].shape[-1]:].tolist()
                    if gen and (looping([t for t in new if t != tok.pad_token_id]) or
                                (tok.eos_token_id not in new and len(new) >= cap)):
                        return ""                    # English only, but it looped: not English speech
                    return tok.decode(new, skip_special_tokens=True)
            return run
        if "qwen3-asr" in low:                               # Qwen3-ASR (52 languages)
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(repo, dtype=half).to(dev).eval()
            def run(audio):
                with torch.inference_mode():
                    inp = proc.apply_transcription_request(audio=audio, sampling_rate=SR).to(model.device, model.dtype)
                    out = model.generate(**inp, max_new_tokens=256)
                    return proc.decode(out[:, inp["input_ids"].shape[1]:], return_format="transcription_only")[0]
            return run
        if "whisper" in low:                                 # Whisper
            from transformers import pipeline
            pipe = pipeline("automatic-speech-recognition", model=repo, dtype=half, device=dev)
            kw = {"language": language} if language else {}
            return lambda audio: pipe({"raw": audio, "sampling_rate": SR}, generate_kwargs=kw)["text"]
        raise SystemExit(f"No torch loader for {repo}")
    if engine == "onnx":
        import onnx_asr
        model = onnx_asr.load_model(repo, quantization="int8")
        asr = getattr(model, "asr", None)
        if language == "en" and asr is not None and hasattr(asr, "_decode") and getattr(asr, "_vocab", None):
            # Parakeet (TDT): add the English mask to the token logits at every
            # decoding step, so it can only pick plain-ASCII pieces (or blank)
            vocab, blank = asr._vocab, asr._blank_idx
            ok = np.array([i == blank or all(ord(c) < 128 for c in vocab.get(i, "")) for i in range(asr._vocab_size)])
            mask = np.where(ok, 0.0, -1e9).astype(np.float32)
            dec = asr._decode
            def decode_en(prev_tokens, prev_state, encoder_out):
                logits, step, state = dec(prev_tokens, prev_state, encoder_out)
                return logits + mask[: logits.shape[-1]], step, state
            asr._decode = decode_en
        return lambda audio: model.recognize(audio, sample_rate=SR)
    raise SystemExit(f"Unknown engine {engine!r} (use mlx, torch, onnx or auto)")


def transcriber(q: queue.Queue, writer: Writer, hub: Hub, language: str | None, language_forced: bool = False):
    """Transcribes queued utterances. Switches models live when hub.want changes
    (utterances wait in the queue while the new model loads)."""
    import gc
    from concurrent.futures import ThreadPoolExecutor
    # Own-voice detection runs on the CPU next to transcription (on the GPU
    # with MLX/torch), so it adds no wait; one worker keeps it in order.
    try:
        from voiceprint import Voiceprints
        voices = Voiceprints()
    except Exception as e:               # missing file or onnxruntime: transcribe without it
        print("Own-voice detection off:", e, flush=True)
        voices = None
    voice_pool = ThreadPoolExecutor(1)
    current, model_run = None, None
    refine_cur, refine_run = None, None
    refine_lang = None                   # the language the rewrite model was loaded for (2B is English-only on "en", else the same)
    loader: dict = {}                    # background load of the rewrite model
    def free_memory(engine):
        gc.collect()
        try:
            if engine == "mlx":
                sys.modules["mlx.core"].clear_cache()
            elif engine == "torch":
                sys.modules["torch"].cuda.empty_cache()
        except Exception:
            pass
    while True:
        lang = "en"      # English only (the --language flag still reaches Whisper/Qwen test models below)
        want = (*hub.want, lang)                          # (id, engine, repo, language)
        if want != current and refine_cur and want[:3] == refine_cur and refine_run \
                and (refine_lang == "en") == (lang == "en"):
            # Both -> 2B only: the rewrite model is already loaded; make it the live one.
            model_run, refine_run, refine_cur, hub.refine_id = refine_run, None, None, None
            hub.refine_q.clear()
            current = want
            hub.model_id, hub.engine, hub.model = want[0], want[1], want[2]
            hub.model_ready, hub.loading, hub.model_error = True, None, None
            free_memory(hub.engine)
            print("Model ready (the 2B rewrite model is now live).", flush=True)
        if want != current:
            mid, engine, repo, lang = want
            language = None if lang == "auto" else lang
            if refine_run is not None and (hub.want_refine is None or (refine_lang == "en") != (language == "en")):
                # free the rewrite model first (also when English only was turned on or off:
                # it reloads for that; 2B is the same for every other language)
                refine_run, refine_cur, hub.refine_id = None, None, None
                hub.refine_q.clear()
                free_memory(engine)
            hub.model_ready, hub.loading = False, mid
            print(f"Loading {repo} on {engine} (first run downloads it)…", flush=True)
            try:
                new_run = load_engine(engine, repo, language)
                new_run(np.zeros(SR, dtype=np.float32))  # warm up
            except Exception as e:
                print("model load failed:", e, flush=True)
                hub.model_error = f"{mid}: {str(e)[:200]}"
                if current is None:
                    raise
                hub.want = current[:3]                    # fall back to the previous model
                hub.model_ready, hub.loading = True, None
                continue
            model_run = new_run
            current = want
            hub.model_id, hub.engine, hub.model = mid, engine, repo
            hub.model_ready, hub.loading, hub.model_error = True, None, None
            hub.audio_s = hub.proc_s = 0.0; hub.latency.clear()
            gc.collect()
            try:
                if engine == "mlx":
                    sys.modules["mlx.core"].clear_cache()
                elif engine == "torch":
                    sys.modules["torch"].cuda.empty_cache()
            except Exception:
                pass
            print("Model ready.", flush=True)
        # The 2B rewrite pass: loaded when both models are on, dropped when not.
        target = hub.want_refine
        if target != refine_cur:
            if target is None:
                if refine_run is not None or refine_cur is not None:
                    refine_run, refine_cur, hub.refine_id = None, None, None
                    hub.refine_q.clear()
                    free_memory(hub.engine)
            elif loader.get("target") != target:
                # Load (and on first use download) 2B in the background: live lines keep coming.
                loader.clear(); loader["target"] = target
                hub.refine_loading = target[0]
                def load_refine(tg=target, lg=language):
                    try:
                        print(f"Loading {tg[2]} for the rewrite pass…", flush=True)
                        if tg[1] == "mlx":
                            # MLX ties a model to the thread that loads it: only download here
                            # (the slow part); the transcriber thread loads it, in seconds.
                            from huggingface_hub import snapshot_download
                            snapshot_download(tg[2])
                            loader.update(done=True, run=None, target=tg)
                        else:
                            loader.update(done=True, run=load_engine(tg[1], tg[2], lg), target=tg, lang=lg)
                    except Exception as e:
                        loader.update(done=True, error=f"{tg[0]}: {str(e)[:200]}", target=tg)
                threading.Thread(target=load_refine, daemon=True).start()
            elif loader.get("done"):
                if loader.get("error"):
                    print("rewrite model failed:", loader["error"], flush=True)
                    hub.model_error = loader["error"]
                    hub.want_refine = None
                elif loader.get("run") is not None and (loader.get("lang") == "en") != (language == "en"):
                    loader.clear()                                         # English only changed while loading: load again
                else:
                    try:
                        run = loader["run"] or load_engine(target[1], target[2], language)
                        run(np.zeros(SR, dtype=np.float32))                # warm up here, on this thread
                        refine_run, refine_cur, hub.refine_id, refine_lang = run, target, target[0], language
                        print("Rewrite pass ready.", flush=True)
                    except Exception as e:
                        hub.model_error = f"{target[0]}: {str(e)[:200]}"; hub.want_refine = None
                loader.clear(); hub.refine_loading = None
        elif loader and loader.get("target") != target:
            loader.clear(); hub.refine_loading = None      # turned off again while it was loading
        writer.flush_md()
        # Live lines first; the rewrite pass uses the time in between.
        if refine_run and hub.refine_q and q.empty():
            it = hub.refine_q.popleft()
            if time.time() - it["queued"] < 60:
                try:
                    res2 = refine_run(it["audio"]) or ""
                    text2 = tidy((res2["text"] if isinstance(res2, dict) else res2).strip())
                    old_text = " ".join(r["text"] for r in it["lines"])
                    if text2 and text2.lower() not in HALLUCINATIONS and text2 != old_text:
                        writer.replace(it["lines"], refined_parts(text2, it))
                        hub.refined += 1
                except Exception as e:
                    print("rewrite error:", e, flush=True)
            writer.flush_md()
            continue
        try:
            item = q.get(timeout=0.5)
            t0, audio, who, uid, ch = item[:5]
            marks = item[5] if len(item) > 5 else [(0, t0)]
            hub.busy_t0 = t0
        except queue.Empty:
            continue
        voice = voice_pool.submit(voices.embed, audio) if voices else None
        try:
            started = time.time()
            res = model_run(audio) or ""
            used_2b = False
            if isinstance(res, dict) and res.get("foreign") and refine_run:
                used_2b = True
                # 470M heard another language: ask 2B, which can only write English, right away
                res = refine_run(audio) or ""
                if res:
                    print(f"  2B instead: {(res['text'] if isinstance(res, dict) else res)[:60]!r}", flush=True)
            text = (res["text"] if isinstance(res, dict) else res).strip()
            hub.proc_s += time.time() - started
            hub.audio_s += len(audio) / SR
            hub.latency.append(max(0.0, time.time() - (t0 / 1000 + len(audio) / SR)))
        except Exception as e:  # keep running on one bad chunk
            print("transcribe error:", e, flush=True)
            hub.busy_t0 = None
            continue
        if not text or tidy(text).lower() in HALLUCINATIONS:
            hub.busy_t0 = None
            continue
        if not current_settings()["splitLines"]:
            parts = [(text, 0.0, len(audio) / SR)]
        else:
            parts = split_timed(res["pieces"]) if isinstance(res, dict) and res.get("pieces") else split_by_text(text, len(audio) / SR)
        keep = [(p, a, b) for p, a, b in ((tidy(p), a, b) for p, a, b in parts) if p and p.lower() not in HALLUCINATIONS]
        bg = False
        if voice and keep:              # only speech that became a line shapes the voices (not music or noise)
            try:
                bg = voices.judge(uid or who, voice.result(timeout=10), len(audio) / SR)
            except Exception as e:      # never lose a line over this
                print("voiceprint error:", e, flush=True)
        written = []
        for part, a, b in keep:
            start = to_clock(marks, a)
            written.append(writer.write(start, max(0.3, (to_clock(marks, b) - start) / 1000), uid, who, part, ch, bg))
        if refine_run and written and not used_2b:     # (2B already wrote it)
            hub.refine_q.append({"t0": t0, "audio": audio, "marks": marks, "lines": written, "queued": time.time()})
        hub.busy_t0 = None


def refined_parts(text: str, it: dict) -> list[tuple[float, float, str]]:
    """Times for the rewritten text: if it splits into as many sentences as the
    live lines, keep each live line's timing; otherwise share the utterance's
    span by text length."""
    lines, audio_s = it["lines"], len(it["audio"]) / SR
    sents = split_by_text(text, audio_s) if current_settings()["splitLines"] else [(text, 0.0, audio_s)]
    if len(sents) == len(lines):
        return [(r["t"], r["end"], s[0]) for r, s in zip(lines, sents)]
    lo, hi = min(r["t"] for r in lines), max(r["end"] for r in lines)
    clamp = lambda x: min(max(x, lo), hi)
    return [(clamp(to_clock(it["marks"], a)), clamp(to_clock(it["marks"], b)), s) for s, a, b in sents]


# ---------------------------------------------------------------- websocket
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--engine", default="auto", choices=["auto", "mlx", "torch", "onnx"],
                    help="speech runtime (auto: mlx on Apple Silicon, torch with an NVIDIA GPU, else onnx)")
    ap.add_argument("--model", default=None, help="model id for the engine (default depends on the engine)")
    ap.add_argument("--language", default="en", help="for Whisper/Qwen models; 'auto' to detect")
    ap.add_argument("--threshold", type=float, default=0.008, help="speech RMS threshold (0-1)")
    ap.add_argument("--no-vad", action="store_true", help="decide speech by loudness only (no Silero VAD)")
    ap.add_argument("--out", default=None, help="transcript folder (default: ../transcripts)")
    ap.add_argument("--parent-pid", type=int, default=0, help="exit when this process (the app) goes away")
    args = ap.parse_args()
    if args.out:
        globals()["TRANSCRIPTS"] = Path(args.out)
    # Which model: --model (any repo, for experiments) > saved choice > default.
    if args.model:
        engine = pick_engine(args.engine)
        want = ("custom", engine, args.model)
    else:
        forced = None if args.engine == "auto" else args.engine
        plan = model_plan()
        choice = plan["live"] if plan["live"] and (not forced or plan["live"]["engine"] == forced) else None
        if choice:
            want = (choice["id"], choice["engine"], choice["repo"])
        else:
            engine = pick_engine(args.engine)
            want = ("default", engine, ENGINE_DEFAULTS[engine])
    engine = want[1]
    language = None if args.language == "auto" else args.language

    # One way to stop, on every OS: set this event (from any thread). Signals
    # aren't reliable here (a background process may have SIGINT ignored, and
    # Windows has no Ctrl-C for windowless processes).
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    request_stop = lambda why: (print(why, flush=True), loop.call_soon_threadsafe(stop.set))

    if args.parent_pid:
        def watch_parent():
            import psutil
            while True:
                time.sleep(2)
                if not psutil.pid_exists(args.parent_pid):
                    request_stop("App is gone; shutting down.")
                    return
        threading.Thread(target=watch_parent, daemon=True).start()

    q: queue.Queue = queue.Queue()
    hub = Hub(want[2])
    hub.engine, hub.want, hub.model_id, hub.loading, hub.model_error = engine, want, want[0], want[0], None
    rp = None if args.model else model_plan()["refine"]
    hub.want_refine = (rp["id"], rp["engine"], rp["repo"]) if rp and rp["engine"] == engine else None
    hub.refine_q = deque()               # utterances waiting for the 2B rewrite
    hub.refine_id, hub.refine_loading, hub.refined = None, None, 0
    hub.queue = q
    hub.loop = asyncio.get_running_loop()
    hub.parent_pid = args.parent_pid
    threading.Thread(target=hub.sample_resources, daemon=True).start()
    if not args.no_vad:
        Vad.load_async()                 # speech detection; loudness only until it's ready
    writer = Writer(hub)
    writer.q = q
    updater = Updater(hub)
    def people_files_loop():
        """Keep people/<username>.md current for everyone who spoke (every 30 s)."""
        write_people_files(hub, list(hub.people_dir))          # everyone known, once at start
        while True:
            time.sleep(30)
            dirty, hub.people_dirty = hub.people_dirty, set()
            if dirty:
                write_people_files(hub, dirty)
    threading.Thread(target=people_files_loop, daemon=True).start()
    persona = PersonalityReader()
    qa_judge = QAJudge(persona)
    threading.Thread(target=updater.loop, daemon=True).start()
    lang_cli = "--language" in sys.argv                  # an explicit --language wins over the setting
    threading.Thread(target=transcriber, args=(q, writer, hub, language, lang_cli), daemon=True).start()

    async def http(connection, request):
        # Plain HTTP (the dashboard); WebSocket upgrades fall through to handle().
        # Only answer requests addressed to this machine (blocks DNS-rebinding
        # tricks where another website pretends to be 127.0.0.1).
        host = request.headers.get("Host", "")
        if host.split(":")[0] not in ("127.0.0.1", "localhost"):
            return Response(403, "Forbidden", Headers([("Content-Length", "0")]), b"")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            origin = request.headers.get("Origin")
            if origin and not re.match(r"^http://(127\.0\.0\.1|localhost)(:\d+)?$", origin):   # the app (no Origin) or the dashboard
                return Response(403, "Forbidden", Headers([("Content-Length", "0")]), b"")
            return None
        path, _, query = request.path.partition("?")
        params = dict(kv.split("=", 1) for kv in query.split("&") if "=" in kv)
        def respond(body, ctype="application/json", status=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False)
            if isinstance(body, str):
                body = body.encode("utf-8")
            return Response(status, {200: "OK", 400: "Bad Request", 403: "Forbidden"}.get(status, "Not Found"),
                            Headers([("Content-Type", ctype), ("Cache-Control", "no-store"),
                                     ("Content-Length", str(len(body)))]), body)
        if path == "/logo.png":
            return respond((DASHBOARD / "logo.png").read_bytes(), "image/png")
        if path in ("/", "/index.html"):
            return respond((DASHBOARD / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/state":
            return respond(hub.state())
        if path == "/api/names":
            return respond(hub.names)
        if path == "/api/people":
            return respond(people_summary(hub))
        if path == "/api/person":
            from urllib.parse import unquote_plus
            uid = unquote_plus(params.get("id", ""))
            if not re.fullmatch(r"\d{15,22}", uid):
                return respond({"error": "bad request"}, status=400)
            if params.get("format") == "md":
                return respond(person_markdown(hub, uid), ctype="text/markdown; charset=utf-8")
            p = hub.people_dir.get(uid) or {}
            return respond({"id": uid, "username": p.get("username"), "usernames": p.get("usernames", []),
                            "names": p.get("names", []), "nicks": p.get("nicks", []), "lines": person_lines(uid)})
        if path == "/api/calls":
            arch = archived_keys()
            calls = []
            for c in all_calls():
                hit = next((k for k in (f"call:{c['id']}", f"channel:{ch_key(c)}", f"server:{srv_key(c)}") if k in arch), None)
                calls.append({**c, "srvKey": srv_key(c), "chKey": ch_key(c), "archived": hit})
            return respond(calls)
        if path == "/api/models":
            plan = model_plan()
            return respond({"current": hub.model_id, "loading": hub.loading or hub.refine_loading, "error": hub.model_error,
                            "selected": plan["selected"], "fallback": plan["fallback"], "refine": hub.refine_id,
                            "models": [m for m in models_here() if not m.get("fallback") or plan["fallback"]]})
        if path == "/api/models/select":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            from urllib.parse import unquote_plus
            ids = [x for x in unquote_plus(params.get("ids", params.get("id", ""))).split(",") if x]   # the dashboard encodes the comma
            avail = {m["id"]: m for m in models_here() if m["available"] and m["id"] in GRANITE}
            if not ids or any(x not in avail for x in ids):
                return respond({"error": "Choose Granite 470M, 2B or both (at least one)."}, status=400)
            save_settings(models=[g for g in GRANITE if g in ids])
            plan = model_plan()
            hub.want = (plan["live"]["id"], plan["live"]["engine"], plan["live"]["repo"])
            r = plan["refine"]
            hub.want_refine = (r["id"], r["engine"], r["repo"]) if r else None
            return respond({"ok": True, "selected": plan["selected"]})
        if path == "/api/plugins":
            return respond(list_plugins())
        if path == "/api/plugins/enable":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            pid, on = params.get("id", ""), params.get("on") == "1"
            if pid not in {p["id"] for p in list_plugins()}:
                return respond({"error": "no such plugin"}, status=404)
            cur = [x for x in (load_settings().get("plugins") or []) if x != pid] + ([pid] if on else [])
            save_settings(plugins=cur)
            return respond({"ok": True, "enabled": cur})
        if path.startswith("/plugins/"):
            from urllib.parse import unquote
            f = plugin_file(unquote(path[len("/plugins/"):]))
            if not f:
                return respond({"error": "not found"}, status=404)
            return respond(f.read_bytes(), PLUGIN_TYPES[f.suffix.lower()])
        if path == "/api/settings":
            if request.headers.get("X-CT") == "1" and params.get("key"):
                from urllib.parse import unquote
                res = set_setting(params["key"], unquote(params.get("value", "")))
                if res.get("ok") and params["key"] == "sensitivity":
                    for p in hub.people.values():
                        p.threshold = args.threshold * SENSITIVITY[res["sensitivity"]]
                        if p.vad_threshold is not None:
                            p.vad_threshold = VAD_SENSITIVITY[res["sensitivity"]]
                return respond(res, status=400 if "error" in res else 200)
            return respond({**current_settings(), "choices": SETTING_CHOICES})
        if path == "/api/qa":
            from urllib.parse import unquote_plus
            days = [d for d in unquote_plus(params.get("days", "")).split(",") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
            try:
                t0, t1 = int(params.get("from", 0)), int(params.get("to", 0)) or int(time.time() * 1000)
            except ValueError:
                return respond({"error": "bad request"}, status=400)
            return respond(qa_judge.request(days, t0, t1, unquote_plus(params.get("ch", "")), parse_parts(params.get("parts", ""))))
        if path == "/api/persona/info":
            return respond(persona.info())
        if path == "/api/persona/start":
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            try:
                from urllib.parse import unquote_plus
                day_q = ",".join(d for d in unquote_plus(params.get("day", "")).split(",") if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d))
                res = persona.start(day_q, int(params.get("from", 0)), int(params.get("to", 0)),
                                    unquote_plus(params.get("ch", "")), unquote_plus(params.get("user", "")), unquote_plus(params.get("name", "")),
                                    fresh=params.get("fresh") == "1", parts=parse_parts(params.get("parts", "")))
            except (ValueError, OSError) as e:
                res = {"error": str(e)}
            return respond(res, status=400 if "error" in res else 200)
        if path == "/api/persona":
            job = persona.jobs.get(params.get("id", ""))
            return respond(job or {"error": "unknown job"}, status=200 if job else 404)
        if path == "/api/update":
            return respond(updater.state)
        if path in ("/api/update/check", "/api/update/apply"):
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            # git can take seconds: never block the audio event loop on it
            if path == "/api/update/check":
                return respond(await asyncio.to_thread(updater.check))
            return respond(await asyncio.to_thread(updater.apply))
        if path == "/api/shutdown":
            # The app asks the server to stop cleanly (Windows has no Ctrl-C for
            # background processes). Same guard as other changes.
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            threading.Timer(0.2, request_stop, args=("Shutdown requested by the app.",)).start()
            return respond({"ok": True})
        if path in ("/api/merge", "/api/unmerge"):
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            from urllib.parse import unquote_plus
            if path == "/api/merge":
                res = merge_calls([i for i in unquote_plus(params.get("ids", "")).split(",") if i])
            else:
                res = unmerge_call(unquote_plus(params.get("id", "")))
            return respond(res, status=400 if "error" in res else 200)
        if path in ("/api/archive", "/api/delete"):
            # Changes data: only the dashboard itself may ask (custom header =>
            # other websites can't send it without a CORS preflight, which fails).
            if request.headers.get("X-CT") != "1":
                return respond({"error": "forbidden"}, status=403)
            from urllib.parse import unquote_plus
            kind, key = params.get("kind", ""), unquote_plus(params.get("key", ""))
            if kind not in ("call", "channel", "server") or not key:
                return respond({"error": "bad request"}, status=400)
            if path == "/api/archive":
                set_archived(f"{kind}:{key}", params.get("on", "1") == "1")
                return respond({"ok": True})
            if params.get("dry") == "1":
                parts = merged_id_of()
                n_calls = sum(1 for d in all_days() for c in day_data(d)[1] if group_match(c, kind, key, parts))
                n_lines = sum(c["total"] for d in all_days() for c in day_data(d)[1] if group_match(c, kind, key, parts))
                return respond({"calls": n_calls, "lines": n_lines})
            return respond(delete_group(kind, key))
        if path == "/api/search":
            from urllib.parse import unquote_plus
            q = unquote_plus(params.get("q", "")).strip().lower()
            if len(q) < 2:
                return respond([])
            hits, parts = [], merged_id_of()
            for d in all_days():
                recs, calls = day_data(d)
                for r in reversed(recs):
                    if q in r["text"].lower() or q in r["speaker"].lower():
                        call = next((c for c in calls if c["start"] <= r["t"] <= c["end"]), None)
                        hits.append({"day": d, "t": r["t"], "speaker": r["speaker"], "text": r["text"], "bg": bool(r.get("bg")),
                                     "chName": r.get("chName") or (call or {}).get("chName"),
                                     "call": call and parts.get(call["id"], call["id"])})
                        if len(hits) >= 150:
                            return respond(hits)
            return respond(hits)
        if path == "/api/days":
            days = sorted({f.stem for f in TRANSCRIPTS.glob("*.md")} | {f.stem for f in TRANSCRIPTS.glob("*.jsonl")}, reverse=True)
            return respond(days)
        if path == "/api/day":
            day = params.get("d", dt.date.today().isoformat())
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
                return respond({"error": "bad day"}, status=404)
            return respond(load_day(day))
        return respond({"error": "not found"}, status=404)

    async def dashboard(ws):
        hub.clients.add(ws)
        try:
            await ws.send(json.dumps(hub.state()))
            async for _ in ws:
                pass
        finally:
            hub.clients.discard(ws)

    async def tick():
        while True:
            await asyncio.sleep(1)
            if hub.clients:
                await hub._send(hub.state())
    asyncio.create_task(tick())

    async def handle(ws):
        if ws.request.path.startswith("/live"):
            return await dashboard(ws)
        people = hub.people
        hub.extensions += 1
        hub.ext_sockets.add(ws)
        conn = hub.conns[ws] = {"account": None, "role": "main", "userId": None, "label": None,
                                "since": time.time(), "capturing": True}

        async def reelect():
            for w, on in hub.elect():
                try:
                    await w.send(json.dumps({"type": "capture", "on": on}))
                except Exception:
                    pass
        last_status = time.time()
        print("Discord window connected.", flush=True)

        async def flush_quiet():
            # Someone left or went silent without trailing quiet audio: flush them.
            while True:
                await asyncio.sleep(1)
                for p in list(people.values()):
                    if p.cur and time.time() - p.last_seen > 2.0:
                        p.flush()
        flusher = asyncio.create_task(flush_quiet())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                ev = json.loads(msg)
                if ev["type"] == "track-audio":
                    if not conn["capturing"]:          # another of your accounts is transcribing this channel
                        continue
                    uid = ev["user"]
                    # Two of your accounts in one call both hear this person: take their
                    # audio from one connection only (until it goes quiet for 2 s).
                    own = hub.uid_owner.get(uid)
                    now_t = time.time()
                    if (own and own[0] is not ws and own[0] in hub.conns and hub.conns[own[0]]["capturing"]
                            and now_t - own[1] < 2.0):
                        continue                       # (hands over at once when that one stands by)
                    hub.uid_owner[uid] = (ws, now_t)
                    if uid not in people:
                        sens = current_settings()["sensitivity"]
                        people[uid] = Segmenter(q, args.threshold * SENSITIVITY.get(sens, 1.0),
                                                label=f"user-{uid[-4:]}", uid=uid,
                                                vad_threshold=None if args.no_vad else VAD_SENSITIVITY.get(sens, 0.5))
                    if ev.get("name"):
                        people[uid].label = ev["name"]
                        hub.learn_name(uid, ev["name"])
                    if ev.get("ch"):
                        people[uid].channel = ev["ch"]
                    pcm = np.frombuffer(base64.b64decode(ev["pcm"]), dtype="<i2").astype(np.float32) / 32768.0
                    people[uid].push(ev["t"], pcm)
                elif ev["type"] == "hello":
                    conn.update(account=ev.get("account"), role="listener" if ev.get("role") == "listener" else "main",
                                userId=ev.get("userId") or conn["userId"], label=ev.get("label") or conn["label"])
                    await reelect()
                elif ev["type"] == "hook-status":
                    hub.ext_reports[ws] = ({**ev["s"], "paused": ev.get("paused")}, time.time())
                    for uid, acc in (ev["s"].get("accounts") or {}).items():
                        hub.learn_account(uid, acc)
                    if ev["s"].get("role") in ("main", "listener"):
                        conn["role"] = ev["s"]["role"]
                    await reelect()
                    with (TRANSCRIPTS.parent / "status.log").open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"),
                                            **ev["s"], "paused": ev.get("paused")}) + "\n")
                if time.time() - last_status > 30:
                    last_status = time.time()
                    live = [p.label for p in people.values() if time.time() - p.last_seen < 30]
                    print(f"  STATUS {len(live)} people talking recently: {', '.join(live) or '-'}  "
                          f"(queue {q.qsize()})", flush=True)
        finally:
            hub.extensions -= 1
            hub.ext_sockets.discard(ws)
            hub.ext_reports.pop(ws, None)
            hub.conns.pop(ws, None)
            for w, on in hub.elect():          # someone else in that channel takes over
                try:
                    await w.send(json.dumps({"type": "capture", "on": on}))
                except Exception:
                    pass
            flusher.cancel()
            for p in people.values():
                p.flush()
            print("Discord window disconnected.", flush=True)

    async with websockets.serve(handle, "127.0.0.1", args.port, max_size=2**22, process_request=http):
        print(f"Dashboard: http://127.0.0.1:{args.port}/  ·  transcripts → {TRANSCRIPTS}", flush=True)
        await stop.wait()
    writer.flush_md(force=True)                   # write any lines still waiting for earlier speech
    print("Server stopped.", flush=True)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
