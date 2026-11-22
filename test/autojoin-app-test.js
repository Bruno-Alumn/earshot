// Auto-join test: the real Electron app with one free account and Auto-join on.
// Three flagged channels in the fake Discord: "Standup" is a FAVOURITE but full,
// "Voice 4" is open, "Side room" is left unflagged. The controller must land in
// Voice 4 — proving it reads occupancy, skips the full channel even though it is
// the favourite, and ignores the unflagged one. Then everyone leaves Voice 4 and
// Standup's limit goes up to 5: the account, now alone, must move to Standup.
// (The silence-hop and the pure ranking are covered by test/autojoin-test.js.)
//   node test/autojoin-app-test.js <scratch-dir>
const { _electron: electron } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const SCRATCH = path.resolve(process.argv[2] || fs.mkdtempSync(path.join(require('os').tmpdir(), 'ct-autojoin-')));
fs.mkdirSync(SCRATCH, { recursive: true });
const DESKTOP = path.join(__dirname, '..', 'app', 'desktop');
const PORT = 8774;
const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const fail = (m) => { console.log('FAIL', m); failed++; };
const get = (p) => new Promise((r) => http.get({ host: '127.0.0.1', port: PORT, path: p, agent: false }, (res) => { let b = ''; res.on('data', (d) => (b += d)); res.on('end', () => { try { r(JSON.parse(b)); } catch { r(null); } }); }).on('error', () => r(null)));

const VOICE4 = '700000000000000001', STANDUP = '700000000000000003', SIDE = '700000000000000009';
const site = http.createServer((req, res) => {
  const u = req.url.split('?')[0];
  const f = path.join(__dirname, 'site', u === '/' ? 'index.html' : u.slice(1));
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': f.endsWith('.html') ? 'text/html' : 'audio/wav' }); fs.createReadStream(f).pipe(res);
}).listen(8899);

(async () => {
  const ud = path.join(SCRATCH, 'userdata'), out = path.join(SCRATCH, 'tx');
  fs.rmSync(ud, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([
    { id: 'l1', role: 'listener', partition: 'persist:test-l1', muted: true, testQuery: 'me=2' },
  ]));
  fs.writeFileSync(path.join(ud, 'settings.json'), JSON.stringify({ autoJoin: true }));
  fs.writeFileSync(path.join(ud, 'channels.json'), JSON.stringify({
    [STANDUP]: { auto: true, fav: true },   // favourite, but will be full -> must be skipped
    [VOICE4]: { auto: true },               // open -> the correct pick
    // Side room is deliberately left unflagged.
  }));
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP], timeout: 120000,
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_USER_DATA: ud, CT_DISCORD_URL: 'http://localhost:8899/',
      CT_AUTOJOIN_TICK: '1500', CT_AUTOJOIN_IDLE: '600', CT_AUTOJOIN_COOLDOWN: '5000', CT_AUTOJOIN_ALONE: '4',
      CT_SERVER_ARGS: `--out ${out} ${process.env.SERVER_ARGS || ''}` } });
  const logs = []; app.process().stdout.on('data', (d) => logs.push(String(d))); app.process().stderr.on('data', (d) => logs.push(String(d)));

  // Wait for the account's Discord window, then seed who is in each channel.
  let win = null;
  for (let i = 0; i < 150 && !win; i++) { win = app.windows().find((w) => w.url().startsWith('http://localhost:8899')) || null; if (!win) await sleep(200); }
  if (!win) { fail('account window missing'); console.log(logs.join('').split('\n').slice(-20).join('\n')); await app.close(); site.close(); process.exit(1); }
  for (let i = 0; i < 240 && !(await get('/api/state'))?.modelReady; i++) await sleep(500);

  let stage = 1;
  const seed = () => { const w = app.windows().find((x) => x.url().startsWith('http://localhost:8899')) || win;
    return w.evaluate(({ v, s, sr, stage }) => { window.__present = stage === 1
      ? { [v]: ['511111111111111111', '522222222222222222'], [s]: ['533333333333333333'], [sr]: ['544444444444444444', '511111111111111111'] }
      : { [v]: [], [s]: ['533333333333333333'], [sr]: ['544444444444444444', '511111111111111111'] }; if (stage === 2) window.__limit = { [s]: 5 }; }, { v: VOICE4, s: STANDUP, sr: SIDE, stage }); };
  await seed();
  // Re-seed periodically: a navigation (when the controller joins) reloads the page and clears it.
  const seeder = setInterval(() => seed().catch(() => {}), 1000);

  // Give directory + occupancy time to reach the controller, then it should join Voice 4.
  let ch = null;
  for (let i = 0; i < 60; i++) {
    const st = await get('/api/state');
    const a = (st?.accounts || []).find((x) => x.account === 'l1') || (st?.accounts || [])[0];
    ch = a?.channel?.name || null;
    if (ch) break;
    await sleep(1000);
  }
  console.log('LANDED IN', JSON.stringify(ch));
  if (ch !== 'Voice 4') fail('auto-join should land in Voice 4 (open, flagged), not the full favourite or an unflagged channel; got ' + ch);

  // Not alone (two people with it), so it must stay put past the alone window.
  await sleep(9000);
  { const st = await get('/api/state'); const a = (st?.accounts || []).find((x) => x.account === 'l1') || (st?.accounts || [])[0];
    if (a?.channel?.name !== 'Voice 4') fail('with people in Voice 4 the account should stay; it went to ' + a?.channel?.name); }

  // Everyone leaves Voice 4; Standup's limit goes up to 5, so it has room. Alone -> move.
  stage = 2; await seed().catch(() => {});
  let moved = null;
  for (let i = 0; i < 60; i++) {
    const st = await get('/api/state');
    const a = (st?.accounts || []).find((x) => x.account === 'l1') || (st?.accounts || [])[0];
    moved = a?.channel?.name || null;
    if (moved && moved !== 'Voice 4') break;
    await sleep(1000);
  }
  clearInterval(seeder);
  console.log('ALONE, MOVED TO', JSON.stringify(moved));
  if (moved !== 'Standup') fail('left alone in Voice 4, the account should move to Standup (open, flagged, has a person); got ' + moved);

  await app.evaluate(({ app }) => app.quit()); await sleep(2500); site.close();
  if (failed) { console.log(`\n${failed} FAILED`); console.log(logs.join('').split('\n').slice(-25).join('\n')); process.exit(1); }
  console.log('\nPASS'); process.exit(0);
})().catch((e) => { console.error(e); site.close(); process.exit(1); });
