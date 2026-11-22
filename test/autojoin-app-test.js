// Auto-join test: the real Electron app with one free account and Auto-join on.
// Three flagged channels in the fake Discord: "Standup" is a FAVOURITE but full,
// "Voice 4" is open, "Side room" is left unflagged. The controller must land in
// Voice 4 — proving it reads occupancy, skips the full channel even though it is
// the favourite, and ignores the unflagged one. Then everyone leaves Voice 4 and
// Standup's limit goes up to 5: the account, now alone, must move to Standup.
// Then its page reloads and it must go back; then it is taken out of voice with the
// page still up and it must stay out, with the dashboard's Auto switch saying so;
// turning Auto off and on lets it join again.
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
  const seed = () => app.evaluate(({ BrowserWindow }, { v, s, sr, stage }) => {
    const js = `window.__present = ${JSON.stringify(stage === 1
      ? { [v]: ['511111111111111111', '522222222222222222'], [s]: ['533333333333333333'], [sr]: ['544444444444444444', '511111111111111111'] }
      : { [v]: [], [s]: ['533333333333333333', '555555555555555555'], [sr]: ['544444444444444444', '511111111111111111'] })};` + (stage === 2 ? `window.__limit = ${JSON.stringify({ [s]: 5 })};` : '');
    for (const w of BrowserWindow.getAllWindows()) if (w.webContents.getURL().startsWith('http://localhost:8899')) w.webContents.executeJavaScript(js).catch(() => {});
  }, { v: VOICE4, s: STANDUP, sr: SIDE, stage });
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
  console.log('ALONE, MOVED TO', JSON.stringify(moved));
  if (moved !== 'Standup') fail('left alone in Voice 4, the account should move to Standup (open, flagged, two people in it); got ' + moved);

  const acc = async () => { const st = await get('/api/state'); return (st?.accounts || []).find((x) => x.account === 'l1') || (st?.accounts || [])[0]; };
  const waitFor = async (fn, secs) => { for (let i = 0; i < secs; i++) { if (await fn()) return true; await sleep(1000); } return false; };

  // The page reloads (as after a crash): it should go back to Standup by itself.
  await sleep(3000);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().find((x) => x.webContents.getURL().startsWith('http://localhost:8899'))?.webContents.reload());
  const dropped = await waitFor(async () => (await acc())?.channel?.name !== 'Standup', 20);
  if (!dropped) fail('the reload should take the account out of Standup for a moment');
  const back = await waitFor(async () => (await acc())?.channel?.name === 'Standup', 45);
  console.log('AFTER RELOAD, BACK IN STANDUP:', back);
  if (!logs.join('').includes('going back')) fail('the controller should log that it is going back after the reload');
  if (!back) fail('after its page reloaded the account should rejoin Standup; it is in ' + (await acc())?.channel?.name);

  // The page crashes: the app reloads it and it goes back again.
  await sleep(3000);
  await app.evaluate(({ BrowserWindow }) => { const w = BrowserWindow.getAllWindows().find((x) => x.webContents.getURL().startsWith('http://localhost:8899')); w?.webContents.forcefullyCrashRenderer(); });
  const crashOut = await waitFor(async () => (await acc())?.channel?.name !== 'Standup', 20);
  const crashBack = crashOut && await waitFor(async () => (await acc())?.channel?.name === 'Standup', 60);
  console.log('AFTER CRASH, BACK IN STANDUP:', crashBack);
  if (!crashBack) fail('after its page crashed the account should be reloaded and rejoin Standup');

  // Taken out of voice with the page still up (a kick or a moderator's disconnect):
  // it must not rejoin, and the dashboard's Auto switch must say it stopped.
  await sleep(3000);
  await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().find((x) => x.webContents.getURL().startsWith('http://localhost:8899'))?.webContents.executeJavaScript('window.__joined = false'));
  await sleep(12000);
  const kicked = await acc();
  console.log('AFTER KICK:', JSON.stringify(kicked?.channel?.name || null));
  if (!logs.join('').includes('stopped there')) fail('the controller should log that it stopped after the kick');
  if (kicked?.channel) fail('after being taken out of voice the account must not rejoin; it went to ' + kicked.channel.name);
  const dash = app.windows().find((x) => x.url().includes(`127.0.0.1:${PORT}`) || x.url().includes(`localhost:${PORT}`));
  if (!dash) fail('dashboard window missing');
  else {
    const label = await dash.textContent('#p-auto-t').catch(() => null);
    console.log('AUTO PILL:', JSON.stringify(label));
    if (label !== 'Auto · 1 stopped') fail('the Auto switch should say "Auto · 1 stopped"; got ' + label);
    // Turning Auto off and on starts it there again.
    await dash.click('#p-auto'); await sleep(800);
    const off = await dash.getAttribute('#p-auto', 'aria-checked');
    await dash.click('#p-auto'); await sleep(800);
    const on = await dash.getAttribute('#p-auto', 'aria-checked');
    if (off !== 'false' || on !== 'true') fail(`the Auto switch should go off then on; got ${off} then ${on}`);
    const again = await waitFor(async () => (await acc())?.channel?.name === 'Standup', 40);
    console.log('AFTER AUTO OFF/ON, BACK IN STANDUP:', again);
    if (!again) fail('after Auto is turned off and on the account should join Standup again');
  }
  clearInterval(seeder);

  await app.evaluate(({ app }) => app.quit()); await sleep(2500); site.close();
  if (failed) { console.log(`\n${failed} FAILED`); console.log(logs.join('').split('\n').slice(-25).join('\n')); process.exit(1); }
  console.log('\nPASS'); process.exit(0);
})().catch((e) => { console.error(e); site.close(); process.exit(1); });
