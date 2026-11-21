"""Words to listen for (Settings): the list is saved cleanly, lines are respelled
("hot zone" -> "hotzone", "C. W. I. S." -> "CWIS"), Granite 470M's beam search
picks a listed word the greedy decode missed and changes nothing else, and
Granite 2B gets a push only once it is partway through a listed word.
  cd server && uv run python ../test/vocab_test.py                 (no models: tokenizers only)
  cd server && uv run python ../test/vocab_test.py --engine mlx    (Mac: also speech made with `say`, both models)"""
import argparse, os, re, subprocess, sys, tempfile, numpy as np
sys.argv, args = sys.argv[:1], argparse.ArgumentParser().parse_known_args()[1]
sys.path.insert(0, ".")
import server as S
eng = args[args.index("--engine") + 1] if "--engine" in args else None
fails = []
def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok: fails.append(what)

WORDS = "hotzone, CWIS, EU, nade"
settings = {"vocabulary": WORDS, "boost": "normal"}
S.load_settings = lambda: dict(settings)
S.save_settings = lambda **kw: settings.update(kw)

# 1. the setting
r = S.set_setting("vocabulary", "hotzone,  CWIS = sea whiz / see wiz\nEU; nade,,  " + "x" * 50)
check(r.get("vocabulary") == "hotzone\nCWIS = sea whiz / see wiz\nEU\nnade", f"list saved one word a line, sounds kept, junk dropped: {r.get('vocabulary')!r}")
check(S.set_setting("boost", "loud").get("error") and S.set_setting("boost", "high").get("ok"), "boost takes off/low/normal/high only")
settings.update(vocabulary=WORDS, boost="normal")

# 2. respelling (runs on every line, through tidy)
for got, want in [("Into the hot zone. Then out.", "Into the hotzone. Then out."),
                  ("Go for the C. W. I. S. First, then the radar.", "Go for the CWIS. First, then the radar."),   # (can't tell from a sentence end; never lowercases a name)
                  ("The C.W.I.S. shot me.", "The CWIS shot me."),
                  ("We left the E.U. Then we came back.", "We left the EU. Then we came back."),
                  ("The eu servers are lagging.", "The EU servers are lagging."),
                  ("Nade over there, two nades.", "Nade over there, two nades."),
                  ("I grew up near the Hudson.", "I grew up near the Hudson."),
                  ("Europe and the euro, a grenade.", "Europe and the euro, a grenade."),
                  ('Get out of the hot zone now!"', "Get out of the hotzone now!"),
                  ("Visit the E. U.", "Visit the EU."),
                  ("The C. W. I. S. Then we pushed.", "The CWIS. Then we pushed."),
                  ("NADE incoming.", "NADE incoming."),
                  ("I'm 6'2\" tall", "I'm 6'2\" tall")]:
    check(S.tidy(got) == want, f"{got!r} -> {S.tidy(got)!r}")
settings["vocabulary"] = "CWIS = sea whiz"
check(S.tidy("Their sea whiz shot it down.") == "Their CWIS shot it down.", "a word written how it sounds is respelled")
settings["vocabulary"] = "Ashe, Alot, Ian, Omen, gg"
for t in ["I waited as he left.", "I like it a lot.", "Ian and I an", "an o men thing"]:
    check(S.tidy(t) == t[:1].upper() + t[1:], f"short words aren't joined into a listed word: {t!r} -> {S.tidy(t)!r}")
check(S.tidy("GG everyone.") == "GG everyone.", "an all-caps line keeps a lowercase listed word in caps")
v2 = S.vocab_now()
st = ("", 0.0)
for p in [" as", " he", " left"]: st = v2.advance(st, p)
check(v2.score(st, final=True) == 0, "the 470M bonus doesn't join ' as' ' he' into Ashe")
settings["vocabulary"] = WORDS
settings["boost"] = "off"
check(S.tidy("the hot zone") == "The hotzone", "respelling still runs with boost off")
settings["boost"] = "normal"

# 3. the matcher behind both boosts
v = S.vocab_now()
def walk(pieces):
    st = ("", 0.0)
    for p in pieces: st = v.advance(st, p)
    return st
check(v.score(walk([" hot", " zone"]), final=True) == v.weight, "' hot' ' zone' spells hotzone (spaces ignored)")
check(v.score(walk([" n", "ade", " over"])) == v.weight, "' n' 'ade' then a new word keeps the bonus")
check(v.score(walk([" n", "ade", "s", ","])) == v.weight, "plural, then punctuation, keeps it")
check(v.score(walk([" gre", "n", "ade"]), final=True) == 0, "no bonus inside another word (gre|n|ade)")
check(v.score(walk([" hot", " dog"]), final=True) == 0, "a started word that isn't finished gives its bonus back")
check(v.score(walk([" E", "U", "rope"]), final=True) == 0, "EU then more letters (Europe) gets nothing")

# 4. Granite 470M: beam search on made-up frames, with its real tokenizer
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("ibm-granite/granite-speech-5.0-470m-turboctc")
blank = tk.pad_token_id
V = max(len(tk), blank + 1)
def frames(rows):
    """rows: one dict per frame {token piece: log-prob} (None = blank); everything else ~impossible."""
    lp = np.full((len(rows), V), -30.0, np.float32)
    for f, row in enumerate(rows):
        for piece, l in row.items():
            lp[f, blank if piece is None else tk.convert_tokens_to_ids(piece)] = l
    return lp
B = None
def decode(ids):
    return "".join(p[0] for p in S.ctc_pieces(ids, 16000 * 2, lambda t: tk.decode(t, skip_special_tokens=True, group_tokens=False), blank)).strip()
texts, starts, cont = v.table(tk, V)
lp = frames([{B: -0.1}, {"▁an": -0.4, "▁a": -1.2}, {B: -0.1}, {"▁aid": -0.4, "▁n": -1.2},
             {B: -0.4, "ade": -1.2}, {B: -0.1}, {"▁over": -0.1}, {B: -0.1}])
greedy = decode(lp.argmax(1))
boosted = decode(S.ctc_boosted(lp, blank, v, texts, starts, cont))
check(greedy == "an aid over" and boosted.endswith("nade over"), f"470M: greedy {greedy!r}, boosted {boosted!r}")
plain = frames([{B: -0.1}, {"▁the": -0.2, "▁a": -1.8}, {B: -0.1}, {"▁ping": -0.3, "▁p": -1.5},
                {B: -0.2, "ing": -2.0}, {"▁is": -0.2}, {B: -0.1}])
check(decode(S.ctc_boosted(plain, blank, v, texts, starts, cont)) == decode(plain.argmax(1)), "470M: a line with no listed word comes out as the greedy decode")
far = frames([{B: -0.1}, {"▁an": -0.05, "▁a": -9.0}, {B: -0.1}, {"▁aid": -0.05, "▁n": -9.0},
              {B: -0.05, "ade": -9.0}, {B: -0.1}])
check(decode(S.ctc_boosted(far, blank, v, texts, starts, cont)) == "an aid", "470M: a listed word the audio doesn't support isn't forced in")
ids = S.ctc_boosted(lp, blank, v, texts, starts, cont)
check(len(ids) == lp.shape[0] and all(0 <= i < lp.shape[1] for i in ids), "470M: one token id per frame, for word timing")
settings["vocabulary"] = "hothot"
vg = S.vocab_now(); tg = vg.table(tk, V)
# the held path ("hot" over two frames) is likelier at frame 1, but the bonus makes "hot hot" win:
# its two tokens must keep a blank between them, or the timing pass would read them as one
rep_ = frames([{"\u2581hot": -0.05}, {"\u2581hot": -0.6, B: -0.8}, {"\u2581hot": -0.05}, {B: -0.05}])
got = decode(S.ctc_boosted(rep_, blank, vg, *tg))
check(got == "hot hot", f"470M: a token said twice stays twice: {got!r}")
i_no = tk.convert_tokens_to_ids("\u2581no")
three = np.full(7, blank); three[[1, 3, 5]] = i_no
check(decode(three) == "no no no", "torch 470M decode keeps 'no no no' (its tokenizer merges repeats by default)")
settings["vocabulary"] = WORDS
held = frames([{B: -0.1}, {"▁hot": -0.2}, {"▁hot": -0.2}, {"▁hot": -0.2}, {B: -0.1}, {"▁zone": -0.2}, {B: -0.1}])
pieces = S.ctc_pieces(S.ctc_boosted(held, blank, v, texts, starts, cont), 16000 * 7 // 10, lambda t: tk.decode(t, skip_special_tokens=True, group_tokens=False), blank)
check([p[0].strip() for p in pieces] == ["hot", "zone"] and pieces[0][2] > pieces[0][1] + 0.15, f"470M: a token held over frames keeps its length {pieces}")

# 5. Granite 2B: the push, with its real tokenizer, and the prompt
tk2 = AutoTokenizer.from_pretrained("ibm-granite/granite-speech-5.0-2b")
vb = S.VocabBias(tk2)
size = len(tk2)
enc = lambda s: tk2.encode(s, add_special_tokens=False)
b = vb.bias(enc(" The hot"), size)
zone = enc(" zone")[0]
check(b is not None and b[zone] > 0, "2B: after ' hot' the token ' zone' gets a push")
check(vb.bias(enc(" The E"), size) is None, "2B: no push from a single letter ('E' -> 'EU')")
check(vb.bias(enc(" The weather is"), size) is None, "2B: no push in an ordinary sentence")
settings["boost"] = "off"
check(vb.bias(enc(" The hot"), size) is None, "2B: boost off means no push")
settings["boost"] = "normal"
b = vb.bias(enc(" Throw a n"), size)
check(b is not None and b[enc("ade")[0]] > 0, "2B: after ' n' the rest of 'nade' gets a push (a 4-letter word)")

# 6. real speech (Mac): accents from `say`, both models
if eng:
    VOICES = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa"]
    SAY = [("Watch out, they're pushing the hotzone from the left side.", "hotzone"),
           ("I only play on E U, the ping is better.", "EU"),
           ("Throw a nade over the wall before we push.", "nade"),
           ("Go for the C W I S first, then the radar.", "CWIS"),
           ("I grew up near the Hudson river in New York.", None),
           ("Can you see what's in the corner over there?", None)]
    def clip(voice, text):
        d = tempfile.mkdtemp(); aiff, wav = os.path.join(d, "a.aiff"), os.path.join(d, "a.wav")
        subprocess.run(["say", "-v", voice, "-o", aiff, text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", aiff, wav], check=True)
        return np.frombuffer(open(wav, "rb").read()[44:], dtype=np.int16).astype(np.float32) / 32768
    clips = [(voice, text, want, clip(voice, text)) for voice in VOICES for text, want in SAY]
    for mid in ["granite-470m", "granite-2b"]:
        m = next(x for x in S.MODELS if x["id"] == mid)
        run = S.load_engine(eng, m["repos"][eng], "en")
        hit = tot = 0; wrong = []
        for voice, text, want, a in clips:
            out = run(a); out = S.tidy((out["text"] if isinstance(out, dict) else out) or "")
            heard = re.findall(r"\b(hotzone|EU|nades?|CWIS)\b", out)
            if want:
                tot += 1; hit += want in out
            elif heard:
                wrong.append(out)
        check(hit >= (0.85 if mid == "granite-470m" else 0.75) * tot, f"{mid}: {hit}/{tot} listed words heard across {len(VOICES)} accents")
        check(not wrong, f"{mid}: no listed word in lines without one {wrong}")
        del run

print("\nFAILED: " + "; ".join(fails) if fails else "\nAll vocabulary checks passed.")
sys.exit(1 if fails else 0)
