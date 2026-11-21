// End-to-end test of the desktop app: the real Electron app starts its own
// server, opens the fake Discord call (test/site) in its built-in Discord
// window and must transcribe each speaker separately.
//   node test/app-test.js <scratch-dir>      (env SHOW=1 shows the Discord window)
const { _electron: electron } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const SCRATCH = path.resolve(process.argv[2] || fs.mkdtempSync(path.join(require('os').tmpdir(), 'ct-app-')));
fs.mkdirSync(SCRATCH, { recursive: true });
const ROOT = path.join(__dirname, '..');
const DESKTOP = path.join(ROOT, 'app', 'desktop');
const OUT = path.join(SCRATCH, 'tx-app');
const PORT = 8768;
fs.rmSync(OUT, { recursive: true, force: true });

const site = http.createServer((req, res) => {
  const u = req.url.split('?')[0];
  const f = path.join(__dirname, 'site', u === '/' ? 'index.html' : u.slice(1));
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': f.endsWith('.html') ? 'text/html' : 'audio/wav' }); fs.createReadStream(f).pipe(res);
}).listen(8899);

const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fail = (msg) => { console.log('FAIL', msg); process.exitCode = 1; };

(async () => {
  const app = await electron.launch({
    executablePath: exe, args: [DESKTOP],
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_DISCORD_URL: 'http://localhost:8899/' + (process.env.SITE_QUERY || ''),
      CT_SERVER_ARGS: `--out ${OUT} ${process.env.SERVER_ARGS || ''}`, CT_USER_DATA: path.join(SCRATCH, 'userdata'),
      CT_SHOW: process.env.SHOW || '0', CT_SHARE_PICK: 'app:share' },
  });
  const logs = [];
  app.process().stdout.on('data', (d) => logs.push(String(d)));
  app.process().stderr.on('data', (d) => logs.push(String(d)));
  let discord = null, dash = null;
  for (let i = 0; i < 100 && !(discord && dash); i++) {
    for (const w of app.windows()) {
      const u = w.url();
      if (u.startsWith('http://localhost:8899')) discord = w;
      else if (u.startsWith(`http://127.0.0.1:${PORT}`) || u.startsWith('data:')) dash = w;
    }
    await sleep(300);
  }
  if (!discord) { fail('no Discord window'); return; }
  discord.on('console', (m) => { if (m.type() === 'error') console.log('  [discord error]', m.text()); });
  const early = await discord.evaluate(() => ({ loadedAt: window.__dtHook?.loadedAt, v: window.__dtHook?.version, ua: navigator.userAgent, native: typeof window.DiscordNative }));
  console.log('HOOK', JSON.stringify(early));
  if (early.loadedAt !== 'loading') fail('hook not injected at document start: ' + early.loadedAt);
  if (/Electron/i.test(early.ua)) fail('user agent mentions Electron');

  // wait for the app's own server
  for (let i = 0; i < 240; i++) {
    const up = await new Promise((r) => http.get(`http://127.0.0.1:${PORT}/api/state`, (res) => r(res.statusCode === 200)).on('error', () => r(false)));
    if (up) break; await sleep(500);
  }
  await sleep(4000);                                     // model ready + app's socket connected
  console.log('STEP server up; window visible:', await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map((w) => [w.getTitle(), w.isVisible()])));
  // Mute the call for the user (as the dashboard button does); transcription must still hear everyone.
  if (dash) {
    for (let i = 0; i < 40 && !(await dash.evaluate(() => !!window.ctApp && location.href.startsWith('http'))); i++) await sleep(250);
    await dash.evaluate(() => window.ctApp.setMuted(true)); await sleep(300);
  }
  const mutedNow = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map((w) => w.webContents).find((wc) => wc.getURL().startsWith('http://localhost:8899'))?.isAudioMuted());
  console.log('CALL AUDIO MUTED FOR USER:', mutedNow);
  if (!mutedNow) fail('mute toggle did not mute the Discord window');
  await discord.click('#join');
  const t0 = Date.now(); let c0 = null;
  await sleep(3000);
  c0 = await discord.evaluate(() => window.__dtHook.frames || 0);
  await sleep(10000);
  const c1 = await discord.evaluate(() => window.__dtHook.frames || 0);
  const rate = (c1 - c0) / 10;
  console.log('CHUNK RATE (all chunks incl. silent, expect ~15 for 3 tracks):', rate.toFixed(1), '/s');
  if (rate < 10) fail('capture throttled: ' + rate + ' chunks/s');
  const state = await new Promise((r) => http.get(`http://127.0.0.1:${PORT}/api/state`, (res) => { let b = ''; res.on('data', (d) => (b += d)); res.on('end', () => r(JSON.parse(b))); }));
  console.log('SERVER EXT', JSON.stringify(state.ext));
  if (!state.ext.connected || state.ext.client !== 'app') fail('server does not see the app as connected');
  console.log('PRESENT (from Discord voice states):', JSON.stringify(state.ext.present), 'accounts:', JSON.stringify(state.accounts), 'own:', JSON.stringify(state.own));
  if (!(state.ext.present || []).length) fail('server does not know who is in the call');
  await sleep(Math.max(0, 32000 - (Date.now() - t0)));
  // Screen share: Discord asks for a screen; the app hands it the shared compact view.
  await discord.click('#share');
  await sleep(4000);
  const share1 = await discord.evaluate(() => window.__share);
  await sleep(2000);
  const share2 = await discord.evaluate(() => window.__share);
  console.log('SCREEN SHARE', JSON.stringify(share2), 'frames in 2s:', (share2?.frames || 0) - (share1?.frames || 0));
  if (share2?.status !== 'sharing' || !(share2.w > 0)) fail('screen share of the compact view did not start');
  const shareView = app.windows().find((w) => w.url().includes('view=share'));
  if (!shareView) fail('shared view window missing');
  else {
    await shareView.screenshot({ path: path.join(SCRATCH, 'app-share-view.png') });
    const hidden = await shareView.evaluate(() => ['#export', '.c-res', '.pills'].map((q) => { const e = document.querySelector(q); return e ? getComputedStyle(e).display : 'none'; }).concat(getComputedStyle(document.querySelector('.side')).visibility === 'hidden' ? 'none' : 'shown', document.querySelector('.app').getBoundingClientRect().width > innerWidth * 0.9 ? 'none' : 'layout-broken'));
    console.log('SHARED VIEW hides private controls:', hidden.every((d) => d === 'none'), 'page scroll:', await shareView.evaluate(() => document.documentElement.scrollHeight - innerHeight));
    if (!hidden.every((d) => d === 'none')) fail('shared view shows private controls');
  }
  const hook = await discord.evaluate(() => ({ capture: window.__dtHook.capture, tracks: window.__dtHook.tracks, chunks: window.__dtHook.chunks, errors: window.__dtHook.errors, names: window.__dtHook.names, dupDropped: window.__dtHook.dupDropped || 0 }));
  console.log('PAGE', JSON.stringify(hook));
  if (dash) {
    const d = await dash.evaluate(() => ({ url: location.href, ctApp: typeof window.ctApp, keys: window.ctApp && Object.keys(window.ctApp) }));
    console.log('DASH', JSON.stringify(d));
    if (d.ctApp !== 'object') fail('dashboard bridge missing');
    await dash.screenshot({ path: path.join(SCRATCH, 'app-dash.png') });
    await dash.click('#p-ext'); await sleep(400);
    await dash.screenshot({ path: path.join(SCRATCH, 'app-dash-pop.png') });
    console.log('PILL', await dash.evaluate(() => document.querySelector('#p-ext').innerText + ' | discord button visible: ' + !document.querySelector('#p-discord').hidden));
    // Settings: open, turn personality reads off, check the server and the page follow
    await dash.evaluate(() => window.openSettings()); await sleep(800);
    await dash.screenshot({ path: path.join(SCRATCH, 'app-settings.png') });
    await dash.$eval('input[data-set="personality"]', (el) => el.click());   // (the Settings list scrolls; click it directly)
    for (let i = 0; i < 25 && !(await dash.evaluate(() => document.documentElement.classList.contains('no-persona'))); i++) await sleep(200);
    const sset = await new Promise((r) => http.get(`http://127.0.0.1:${PORT}/api/settings`, (res) => { let b = ''; res.on('data', (d) => (b += d)); res.on('end', () => r(JSON.parse(b))); }));
    const hidden = await dash.evaluate(() => document.documentElement.classList.contains('no-persona'));
    const appSet = await dash.evaluate(() => window.ctApp.appSettings());
    console.log('SETTINGS personality ->', sset.personality, '| page hides reads:', hidden, '| app settings:', JSON.stringify(appSet));
    if (sset.personality !== false || !hidden) fail('personality setting did not apply');
    await dash.$eval('input[data-set="personality"]', (el) => el.click());   // (the Settings list scrolls; click it directly) await sleep(500);
    await dash.evaluate(() => window.showView('calls'));   // Settings is a page now; back to the calls
    const leaked = await discord.evaluate(() => typeof window.ctApp + '/' + typeof window.require);
    if (leaked !== 'undefined/undefined') fail('Discord page can see app APIs: ' + leaked);
  }
  // quit through the app: must stop its server
  const pidState = state.pid;
  await app.evaluate(({ app }) => app.quit());
  await sleep(4000);
  let serverAlive = true; try { process.kill(pidState, 0); } catch { serverAlive = false; }
  console.log('SERVER STOPPED ON QUIT:', !serverAlive);
  if (serverAlive) fail('server still running after quit');
  site.close();
  const files = fs.existsSync(OUT) ? fs.readdirSync(OUT).filter((f) => f.endsWith('.jsonl')) : [];
  const lines = files.flatMap((f) => fs.readFileSync(path.join(OUT, f), 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l)));
  console.log('TRANSCRIPT');
  for (const l of lines) console.log(`  ${l.speaker.padEnd(8)} [${l.chName || l.ch}] ${l.text}`);
  const speakers = new Set(lines.map((l) => l.speaker));
  for (const s of ['Alice', 'Bob', 'Carol', 'Dave']) if (![...speakers].some((x) => x.toLowerCase().startsWith(s.toLowerCase()))) fail('missing speaker ' + s);
  const doubled = lines.filter((l) => /\b(\w+) \1\b/i.test(l.text) || l.text.split(' ').length > 25);
  if (doubled.length) fail('the same speech was mixed in twice: ' + doubled.map((l) => l.text).join(' | '));
  const bob = lines.filter((l) => l.speaker === 'Bob').length;
  if (bob > 3) fail('Bob was transcribed more than once per thing he said: ' + bob + ' lines');
  if (process.exitCode !== 1) console.log('PASS');
  else console.log(logs.join('').split('\n').slice(-30).join('\n'));
})().catch((e) => { console.error(e); process.exit(1); });
