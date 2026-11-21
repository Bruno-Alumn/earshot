"""Order test: a long sentence that started first but finishes transcribing last
must still come first in the .md transcript (written in spoken order).
  cd server && uv run python ../test/order_test.py"""
import sys, tempfile, time, queue, types
from pathlib import Path
sys.argv = ["x"]; sys.path.insert(0, ".")
import server as S
tmp = Path(tempfile.mkdtemp()); S.TRANSCRIPTS = tmp
hub = types.SimpleNamespace(people={}, busy_t0=None, publish=lambda m: None, people_dirty=set(), username_of=lambda uid: None)
w = S.Writer(hub); w.q = queue.Queue()
t0 = time.time() * 1000
# A started talking at t0 (20 s sentence, still being transcribed); B replied at t0+5 s and finished first.
hub.busy_t0 = t0
w.write(t0 + 5000, 1.5, "b", "Bob", "Short reply.")
w.write(t0 + 8000, 1.0, "c", "Cara", "Me too.")
md = next(tmp.glob("*.md"))
held = "Short reply" not in md.read_text()
hub.busy_t0 = None
w.write(t0, 20.0, "a", "Alice", "A long question that started first?")
text = md.read_text()
order = [text.index(x) for x in ("long question", "Short reply", "Me too")]
print(text)
ok = held and order == sorted(order)
print("PASS" if ok else "FAIL", "held while earlier speech was pending:", held, "order:", order)
sys.exit(0 if ok else 1)
