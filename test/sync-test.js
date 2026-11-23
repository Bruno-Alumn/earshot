// Sync between two installs, end to end: two servers with their own data folders,
// two sync links (app/desktop/sync.js, as the app runs it), paired with a code.
//   node test/sync-test.js [scratch dir]
const { spawn } = require('child_process');
const path = require('path'), fs = require('fs'), os = require('os');
const { createSync } = require('../app/desktop/sync.js');

const SCRATCH = path.resolve(process.argv[2] || fs.mkdtempSync(path.join(os.tmpdir(), 'ct-sync-')));
const PY = path.join(__dirname, '..', 'server', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
let failed = false;
const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) failed = true; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const jl = (recs) => recs.map((r) => JSON.stringify(r)).join('\n') + '\n';

const day1 = '2026-11-20', day2 = '2026-11-21';
const T1 = new Date(2026, 10, 20, 20, 0, 0).getTime(), T2 = new Date(2026, 10, 21, 20, 0, 0).getTime();
const U1 = '111111111111111111', U2 = '222222222222222222', U3 = '333333333333333333';
const ch = { ch: '900000000000000001', chName: 'Voice 1', guild: '800000000000000001', guildName: 'Test' };
const chX = { ch: '900000000000000009', chName: 'Voice 9', guild: '800000000000000001', guildName: 'Test' };
const legacy = { t: T1 - 60000, end: T1 - 58000, user: U1, speaker: 'Alice', text: 'an old line without an id' };

function seed(dir, recs1, recs2, extra) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(path.join(dir, 'transcripts'), { recursive: true });
  fs.writeFileSync(path.join(dir, 'transcripts', `${day1}.jsonl`), jl(recs1));
  if (recs2) fs.writeFileSync(path.join(dir, 'transcripts', `${day2}.jsonl`), jl(recs2));
  for (const [f, v] of Object.entries(extra || {})) fs.writeFileSync(path.join(dir, f), JSON.stringify(v));
  fs.writeFileSync(path.join(dir, 'settings.json'), JSON.stringify({ personality: false, qaLinks: false }));
}

function server(dir, port) {
  const p = spawn(PY, [path.join(__dirname, '..', 'server', 'server.py'), '--port', String(port), '--out', path.join(dir, 'transcripts'), ...(process.env.SERVER_ARGS || '').split(' ').filter(Boolean)], { cwd: path.join(__dirname, '..', 'server') });
  let log = ''; p.stdout.on('data', (d) => (log += d)); p.stderr.on('data', (d) => (log += d));
  p.log = () => log;
  return p;
}
const get = (port, p) => fetch(`http://127.0.0.1:${port}${p}`, { headers: { 'X-CT': '1' } }).then((r) => r.json());
const lines = (dir, d) => { try { return fs.readFileSync(path.join(dir, 'transcripts', `${d}.jsonl`), 'utf8').trim().split('\n').filter(Boolean).map(JSON.parse); } catch { return []; } };

(async () => {
  const A = path.join(SCRATCH, 'A'), B = path.join(SCRATCH, 'B');
  seed(A, [
    legacy,
    { id: 'a000000001', t: T1, end: T1 + 2000, user: U1, speaker: 'Alice', text: 'hello from the mac', ...ch },
    { id: 'a0000000d1', t: T1 + 5000, end: T1 + 7000, user: U2, speaker: 'Bob', text: 'Get in the car now', ...ch },
  ], [
    { id: 'a0000000x1', t: T2 + 1000, end: T2 + 2000, user: U2, speaker: 'Bob', text: 'line in the call that gets deleted', ...chX },
  ], { 'names.json': { [U1]: 'Alice' }, 'personas.json': { [U1]: { reads: [{ at: 100, lines: 50, text: 'read on A' }] } },
       'people.json': { [U1]: { username: 'alice', usernames: ['alice'], names: ['Alice'], nicks: [], first: 1 } } });
  seed(B, [
    legacy,
    { id: 'b000000001', t: T1 + 1000, end: T1 + 3000, user: U3, speaker: 'Carol', text: 'hello from the pc', ...ch },
    { id: 'b0000000d1', t: T1 + 5800, end: T1 + 7600, user: U2, speaker: 'Bob', text: 'get in the car now!', ...ch },
  ], [
    { id: 'b0000000x1', t: T2, end: T2 + 1500, user: U3, speaker: 'Carol', text: 'this call is deleted on the pc', ...chX },
    { id: 'b0000000x2', t: T2 + 3000, end: T2 + 4000, user: U2, speaker: 'Bob', text: 'and so is this', ...chX },
  ], { 'names.json': { [U3]: 'Carol' } });

  const sA = server(A, 8790), sB = server(B, 8791);
  for (const s of [sA, sB]) { for (let i = 0; i < 200 && !s.log().includes('Dashboard:'); i++) await sleep(300); }
  ok(sA.log().includes('Dashboard:') && sB.log().includes('Dashboard:'), 'both servers up');

  // delete the call on B before they ever sync: A's copy of that call must go too
  const callsB = await get(8791, '/api/calls');
  const x = callsB.find((c) => c.ch === chX.ch);
  const del = await get(8791, `/api/delete?kind=call&key=${encodeURIComponent(x.id)}`);
  ok(del.lines === 2, `deleted the call on B (${del.lines} lines)`);

  const mk = (dir, port, sport) => createSync({ stateFile: path.join(dir, 'app', 'sync.json'), serverBase: `http://127.0.0.1:${port}`, port: sport, name: path.basename(dir) });
  const yA = mk(A, 8790, 8797), yB = mk(B, 8791, 8798);
  yA.start(); yB.start();
  const pc = await yA.startPairing();
  ok(/^\S+ [A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}$/.test(pc.code), `A shows a code (${pc.code.replace(/ .*/, ' ****')})`);
  const secret = pc.code.split(' ')[1];
  const bad = await yB.join(`127.0.0.1:8797 ${secret.replace(/^./, (c) => (c === 'A' ? 'B' : 'A'))}`);
  ok(!!bad.error, 'a wrong code is refused');
  const j = await yB.join(`127.0.0.1:8797 ${secret}`);
  ok(j.ok && j.name === 'A', 'B pairs with A');
  ok(yA.view().peers.length === 1 && !yA.view().pairing, 'A knows B, the code is used up');
  await sleep(1500);
  while (yB.view().busy) await sleep(300);
  const r1 = await yB.syncNow();
  const r = r1[yA.installId];
  ok(r && !r.error, `sync round: ${r?.error || `got ${r.got}, sent ${r.sent}`}`);

  const [mA, mB] = await Promise.all([get(8790, '/api/sync/manifest'), get(8791, '/api/sync/manifest')]);
  ok(JSON.stringify(mA.days) === JSON.stringify(mB.days), 'every day matches on both');
  for (const [n, dir] of [['A', A], ['B', B]]) {
    const L1 = lines(dir, day1), L2 = lines(dir, day2);
    ok(L1.some((x) => x.id === 'a000000001') && L1.some((x) => x.id === 'b000000001'), `${n}: has both machines' lines`);
    ok(L1.filter((x) => x.user === U2).length === 1, `${n}: the line both heard is kept once`);
    ok(L1.filter((x) => x.text === legacy.text).length === 1, `${n}: the old line without an id is there once`);
    ok(!L2.some((x) => x.ch === chX.ch), `${n}: the deleted call is gone`);
  }
  const nA = await get(8790, '/api/names'), nB = await get(8791, '/api/names');
  ok(nA[U3] === 'Carol' && nB[U1] === 'Alice', 'names went both ways');
  const pB = JSON.parse(fs.readFileSync(path.join(B, 'personas.json'), 'utf8'));
  ok(pB[U1]?.reads?.length === 1, 'personality reads went across');
  const r2 = (await yA.syncNow())[yB.installId];
  ok(r2 && !r2.error && r2.got === 0 && r2.sent === 0, `next round from A moves nothing (${r2?.error || `got ${r2.got}, sent ${r2.sent}`})`);

  // a line said later on A arrives on B at the next round
  fs.appendFileSync(path.join(A, 'transcripts', `${day1}.jsonl`), jl([{ id: 'a000000002', t: T1 + 9000, end: T1 + 9500, user: U1, speaker: 'Alice', text: 'one more thing', ...ch }]));
  await yB.syncNow();
  ok(lines(B, day1).some((x) => x.id === 'a000000002'), 'a new line reaches the other machine');

  // a 2B rewrite on A: its refined line replaces the live one on B too (not taken for a copy of it)
  fs.appendFileSync(path.join(A, 'transcripts', `${day1}.jsonl`), jl([
    { tomb: true, t: T1 + 9000, replaces: ['a000000002'] },
    { id: 'a000000003', t: T1 + 9000, end: T1 + 9500, user: U1, speaker: 'Alice', text: 'One more thing.', refined: true, ...ch },
  ]));
  await yB.syncNow();
  const dB = await get(8791, `/api/day?d=${day1}`).catch(() => null);
  const shownB = (dB?.lines || dB || []).filter?.((x) => x.user === U1 && /one more thing/i.test(x.text)) || [];
  ok(lines(B, day1).some((x) => x.id === 'a000000003') && shownB.length === 1 && shownB[0].id === 'a000000003', `the rewritten line shows once on B (${shownB.length})`);

  // B was behind: it writes its own copy of speech A already sent; B keeps one
  fs.appendFileSync(path.join(A, 'transcripts', `${day1}.jsonl`), jl([{ id: 'a000000004', t: Date.now() - 60000, end: Date.now() - 58000, user: U3, speaker: 'Carol', text: 'the late copy test line', ...ch }]));
  const today = new Date(); const dToday = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`;
  fs.appendFileSync(path.join(A, 'transcripts', `${dToday}.jsonl`), jl([{ id: 'a000000005', t: Date.now() - 30000, end: Date.now() - 28000, user: U3, speaker: 'Carol', text: 'said just now on the call', ...ch }]));
  await yB.syncNow();
  ok(lines(B, dToday).some((x) => x.id === 'a000000005'), 'a line from just now arrives');

  // archive on A, sync, unarchive on B, sync: both unarchived (three-way, not a union)
  const callsA = await get(8790, '/api/calls');
  const cA = callsA.find((c) => c.ch === ch.ch && c.id.startsWith(day1));
  await get(8790, `/api/archive?kind=call&key=${encodeURIComponent(cA.id)}&on=1`);
  await yA.syncNow();
  const arB = JSON.parse(fs.readFileSync(path.join(B, 'archive.json'), 'utf8')).keys;
  ok(arB.length === 1, `archiving reaches B (${arB})`);
  await get(8791, `/api/archive?kind=call&key=${encodeURIComponent(arB[0].slice(5))}&on=0`);
  await yB.syncNow(); await yA.syncNow();
  const arA2 = JSON.parse(fs.readFileSync(path.join(A, 'archive.json'), 'utf8')).keys, arB2 = JSON.parse(fs.readFileSync(path.join(B, 'archive.json'), 'utf8')).keys;
  ok(arA2.length === 0 && arB2.length === 0, `unarchiving on B undoes it on both (A ${arA2.length}, B ${arB2.length})`);

  // a channel deleted on A takes B's own lines in that channel too, and its .md lines
  const chY = { ch: '900000000000000007', chName: 'Voice 7', guild: '800000000000000001', guildName: 'Test' };
  const T3 = new Date(2026, 10, 22, 12, 0, 0).getTime(), day3 = '2026-11-22';
  fs.writeFileSync(path.join(B, 'transcripts', `${day3}.jsonl`), jl([
    { id: 'b0000000y1', t: T3, end: T3 + 1000, user: U3, speaker: 'Carol', text: 'only the pc heard this', ...chY },
    { id: 'b0000000y2', t: T3 + 600000, end: T3 + 601000, user: U3, speaker: 'Carol', text: 'pc only, later', ...chY }]));
  fs.writeFileSync(path.join(B, 'transcripts', `${day3}.md`), `# t\n\n**[12:00:00] Carol:** only the pc heard this\n`);
  fs.writeFileSync(path.join(A, 'transcripts', `${day3}.jsonl`), jl([{ id: 'a0000000y1', t: T3 + 300000, end: T3 + 301000, user: U2, speaker: 'Bob', text: 'the mac heard this', ...chY }]));
  await yA.syncNow();
  // B records more in that channel that A never sees before the delete
  fs.appendFileSync(path.join(B, 'transcripts', `${day3}.jsonl`), jl([{ id: 'b0000000y3', t: T3 + 1200000, end: T3 + 1201000, user: U3, speaker: 'Carol', text: 'pc only, not synced yet', ...chY }]));
  const chKey = (await get(8790, '/api/calls')).find((c) => c.ch === chY.ch).chKey;
  const delA = await get(8790, `/api/delete?kind=channel&key=${encodeURIComponent(chKey)}`);
  ok(delA.lines >= 3, `deleted the channel on A (${delA.lines} lines)`);
  await yA.syncNow();
  ok(!lines(B, day3).some((x) => x.ch === chY.ch), 'B lost every line in that channel, its own too');
  ok(!fs.readFileSync(path.join(B, 'transcripts', `${day3}.md`), 'utf8').includes('only the pc heard this'), 'and the .md lines');
  const callsB3 = await get(8791, '/api/calls');
  ok(!callsB3.some((c) => c.ch === chY.ch), 'the channel does not come back on B');
  await yB.syncNow();
  ok(!lines(A, day3).some((x) => x.ch === chY.ch), 'nor on A after B syncs');

  // both start a round at the same moment: one waits, nothing breaks, the next round is quiet
  const both = await Promise.all([yA.syncNow(), yB.syncNow()]);
  const errsBoth = both.flatMap((r) => Object.values(r)).filter((x) => x?.error);
  ok(errsBoth.every((x) => /same moment/.test(x.error)), `simultaneous rounds: ${errsBoth.length ? errsBoth.map((x) => x.error.slice(0, 40)).join(' | ') : 'both fine'}`);
  await yA.syncNow();
  const q = (await yB.syncNow())[yA.installId];
  ok(q && !q.error && q.got === 0 && q.sent === 0, `then a quiet round (${q?.error || `got ${q.got}, sent ${q.sent}`})`);

  // A has two calls in a channel with a gap; B heard chatter in the gap, so has one long call.
  // Deleting A's first call must not take B's second half, and A's second call must survive.
  const chZ = { ch: '900000000000000005', chName: 'Voice 5', guild: '800000000000000001', guildName: 'Test' };
  const T4 = new Date(2026, 10, 19, 18, 0, 0).getTime(), day4 = '2026-11-19', M = 60000;
  fs.writeFileSync(path.join(A, 'transcripts', `${day4}.jsonl`), jl([
    { id: 'a0000000z1', t: T4, end: T4 + 1000, user: U1, speaker: 'Alice', text: 'first call on the mac', ...chZ },
    { id: 'a0000000z2', t: T4 + 40 * M, end: T4 + 40 * M + 1000, user: U1, speaker: 'Alice', text: 'second call on the mac', ...chZ }]));
  fs.writeFileSync(path.join(B, 'transcripts', `${day4}.jsonl`), jl([
    { id: 'b0000000z1', t: T4 + 2000, end: T4 + 3000, user: U3, speaker: 'Carol', text: 'pc hears the first call', ...chZ },
    { id: 'b0000000z2', t: T4 + 14 * M, end: T4 + 14 * M + 1000, user: U3, speaker: 'Carol', text: 'pc hears chatter in the gap', ...chZ },
    { id: 'b0000000z3', t: T4 + 28 * M, end: T4 + 28 * M + 1000, user: U3, speaker: 'Carol', text: 'more chatter in the gap', ...chZ }]));
  const c4 = (await get(8790, '/api/calls')).filter((c) => c.ch === chZ.ch).sort((a, b) => a.start - b.start);
  ok(c4.length === 2, `A has two calls there (${c4.length})`);
  await get(8790, `/api/delete?kind=call&key=${encodeURIComponent(c4[0].id)}`);
  await yA.syncNow(); await yB.syncNow();
  const zA = lines(A, day4).map((x) => x.id), zB = lines(B, day4).map((x) => x.id);
  ok(!zB.includes('b0000000z1') && zB.includes('b0000000z3') && zB.includes('a0000000z2'), `B lost only its copy of the deleted call (${zB.join(',')})`);
  ok(zA.includes('a0000000z2') && zA.includes('b0000000z3') && !zA.includes('a0000000z1'), `A kept its second call and got B's later lines (${zA.join(',')})`);

  // archived on B in the middle of a round A runs: still archived on both afterwards
  const cY = (await get(8791, '/api/calls')).find((c) => c.ch === ch.ch && c.id.startsWith(day1));
  const yA2 = createSync({ stateFile: path.join(A, 'app', 'sync.json'), serverBase: 'http://127.0.0.1:8790', port: 8796, name: 'A',
    hooks: { afterPull: () => get(8791, `/api/archive?kind=call&key=${encodeURIComponent(cY.id)}&on=1`) } });
  await yA2.syncNow(); await yB.syncNow(); await yA.syncNow();
  const k1 = JSON.parse(fs.readFileSync(path.join(A, 'archive.json'), 'utf8')).keys, k2 = JSON.parse(fs.readFileSync(path.join(B, 'archive.json'), 'utf8')).keys;
  ok(k1.length === 1 && k2.length === 1, `an archive made mid-round sticks on both (A ${k1.length}, B ${k2.length})`);
  yA2.stop();

  // an unpaired computer is refused
  yA.unpair(yB.installId);
  const r3 = (await yB.syncNow())[yA.installId];
  ok(r3?.error && /Pair them again|Couldn't reach/.test(r3.error), `after unpairing, the other side gets nothing (${r3?.error})`);

  yA.stop(); yB.stop(); sA.kill(); sB.kill();
  console.log(failed ? '\nFAILED' : '\nPASS');
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
