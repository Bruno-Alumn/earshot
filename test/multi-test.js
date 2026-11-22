// Multi-account test: the real desktop app with two Discord accounts.
//   1. separate calls: main in the usual fake call, a listener in another
//      ("Side room", other people) -> both transcribed, kept apart, no duplicates;
//      the listener has a silent microphone and its sound is off.
//   2. same call: both accounts in one channel -> only one transcribes it.
//   node test/multi-test.js <scratch-dir>
const { _electron: electron } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const SCRATCH = path.resolve(process.argv[2]);
fs.mkdirSync(SCRATCH, { recursive: true });
const DESKTOP = path.join(__dirname, '..', 'app', 'desktop');
const PORT = 8771;
const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = false;
const fail = (m) => { console.log('FAIL', m); failed = true; };
const get = (p) => new Promise((r) => http.get({ host: '127.0.0.1', port: PORT, path: p, agent: false }, (res) => { let b = ''; res.on('data', (d) => (b += d)); res.on('end', () => { try { r(JSON.parse(b)); } catch { r(null); } }); }).on('error', () => r(null)));

const site = http.createServer((req, res) => {
  const u = req.url.split('?')[0];
  const f = path.join(__dirname, 'site', u === '/' ? 'index.html' : u.slice(1));
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': f.endsWith('.html') ? 'text/html' : 'audio/wav' }); fs.createReadStream(f).pipe(res);
}).listen(8899);

async function scenario(name, listenerQuery) {
  console.log(`\n== ${name}`);
  const dir = path.join(SCRATCH, name), out = path.join(dir, 'tx'), ud = path.join(dir, 'userdata');
  fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([
    { id: 'main', role: 'main', partition: 'persist:test-main', testQuery: 'me=1' },
    { id: 'l1', role: 'listener', partition: 'persist:test-l1', muted: true, testQuery: listenerQuery },
  ]));
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP],
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_DISCORD_URL: 'http://localhost:8899/', CT_USER_DATA: ud,
      CT_SERVER_ARGS: `--out ${out} ${process.env.SERVER_ARGS || ''}` } });
  const logs = []; app.process().stdout.on('data', (d) => logs.push(String(d))); app.process().stderr.on('data', (d) => logs.push(String(d)));
  let wins = [];
  for (let i = 0; i < 100 && wins.length < 2; i++) { wins = app.windows().filter((w) => w.url().startsWith('http://localhost:8899')); await sleep(300); }
  if (wins.length < 2) { fail('expected two Discord windows, got ' + wins.length); await app.close(); return; }
  const isListener = async (w) => w.evaluate(() => !!window.__ctListener);
  const listener = (await isListener(wins[0])) ? wins[0] : wins[1], main = listener === wins[0] ? wins[1] : wins[0];
  for (let i = 0; i < 240 && !(await get('/api/state'))?.modelReady; i++) await sleep(500);
  await sleep(3000);

  // listener: silent mic, no camera, real devices refused, sound off
  const mic = await listener.evaluate(async () => {
    const r = {};
    try { const st = await navigator.mediaDevices.getUserMedia({ audio: true }); r.audio = st.getAudioTracks().length; r.label = st.getAudioTracks()[0]?.label || ''; } catch (e) { r.audioErr = e.name; }
    try { await navigator.mediaDevices.getUserMedia({ video: true }); r.video = 'allowed'; } catch (e) { r.video = e.name; }
    try { r.perm = (await navigator.permissions.query({ name: 'microphone' })).state; } catch (e) { r.perm = 'n/a'; }
    return r;
  });
  const muted = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().filter((w) => w.webContents.getURL().startsWith('http://localhost:8899')).map((w) => [w.getTitle(), w.webContents.isAudioMuted()]));
  console.log('LISTENER MIC', JSON.stringify(mic), '| sound muted per window', JSON.stringify(muted));
  if (mic.audio !== 1 || mic.label !== 'MediaStreamAudioDestinationNode') fail('listener did not get the silent microphone');
  if (mic.video !== 'NotAllowedError') fail('listener camera not refused');
  // your own account: never the real microphone either (a silent one instead)
  const mainMic = await main.evaluate(async () => { try { const st = await navigator.mediaDevices.getUserMedia({ audio: true }); return st.getAudioTracks()[0]?.label; } catch (e) { return e.name; } });
  console.log('MAIN MIC', mainMic);
  if (mainMic !== 'MediaStreamAudioDestinationNode') fail('your own account got a real microphone: ' + mainMic);
  if (!muted.some(([t, m]) => /listener/.test(t) && m)) fail('listener window is not muted');

  await main.click('#join'); await listener.click('#join');
  await sleep(15000);
  const hk = async (w) => w.evaluate(() => ({ ctx: window.__dtHook?.ctxState, listener: !!window.__ctListener, en: window.__dtHook?.enabled, capture: window.__dtHook?.capture, tracks: window.__dtHook?.tracks, frames: window.__dtHook?.frames, chunks: window.__dtHook?.chunks, errors: window.__dtHook?.errors, harness: window.__harness?.status }));
  console.log('HOOK main', JSON.stringify(await hk(main)), '| listener', JSON.stringify(await hk(listener)));
  // the server settles who transcribes a shared channel within a few seconds; wait for it
  let st = await get('/api/state');
  for (let i = 0; i < 20 && name === 'same' && (st?.accounts || []).filter((x) => x.capturing && x.inCall).length !== 1; i++) { await sleep(500); st = await get('/api/state'); }
  console.log('ACCOUNTS', JSON.stringify((st?.accounts || []).map((a) => ({ role: a.role, ch: a.channel?.name, capturing: a.capturing, present: a.present.length }))), 'own', JSON.stringify(st?.own));
  await sleep(20000);
  const calls = await get('/api/calls');
  await app.evaluate(({ app }) => app.quit()); await sleep(3000);
  const lines = fs.existsSync(out) ? fs.readdirSync(out).filter((f) => f.endsWith('.jsonl')).flatMap((f) => fs.readFileSync(path.join(out, f), 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l))) : [];
  for (const l of lines) console.log(`  ${l.speaker.padEnd(15)} [${l.chName}] ${l.text.slice(0, 60)}`);
  console.log('CALLS', JSON.stringify((calls || []).map((c) => `${c.chName}: ${c.lines} lines`)));
  return { st, lines, calls, logs };
}

async function transcribeFlow() {
  console.log('\n== transcribe a channel');
  const dir = path.join(SCRATCH, 'pick'), out = path.join(dir, 'tx'), ud = path.join(dir, 'userdata');
  fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([
    { id: 'main', role: 'main', partition: 'persist:test-main', testQuery: 'me=1' },
    { id: 'l1', role: 'listener', partition: 'persist:test-l1', muted: true, testQuery: 'set=b&me=2' },
  ]));
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP],
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_DISCORD_URL: 'http://localhost:8899/', CT_USER_DATA: ud, CT_SERVER_ARGS: `--out ${out}` } });
  for (let i = 0; i < 240 && !(await get('/api/state'))?.modelReady; i++) await sleep(500);
  let dash;
  for (let i = 0; i < 60 && !dash; i++) { dash = app.windows().find((w) => w.url().startsWith(`http://127.0.0.1:${PORT}`)); await sleep(300); }
  await sleep(10000);                                  // accounts report their servers
  const info = await dash.evaluate(() => window.ctApp.accounts());
  console.log('DIRECTORY', JSON.stringify(info.guilds.map((g) => `${g.name}: ${g.channels.map((c) => c.name).join('/')} [${g.accounts.join(',')}]`)));
  if (!info.guilds.some((g) => g.channels.some((c) => c.name === 'Voice 4'))) fail('channel directory missing');
  await dash.click('#p-status'); await sleep(500);     // the one status button (it replaced the Server/Discord/Model pills)
  await dash.screenshot({ path: path.join(SCRATCH, 'accounts-popover.png') });
  await dash.evaluate(() => window.openTranscribe()); await sleep(600);
  await dash.screenshot({ path: path.join(SCRATCH, 'transcribe-picker.png') });
  const r = await dash.evaluate(() => window.ctApp.transcribe('900000000000000001', '700000000000000001'));
  console.log('ASSIGNED', JSON.stringify(r));
  if (r.account !== 'l1') fail('should pick the free listener');
  await sleep(4500);
  const lw = app.windows().find((w) => w.url().includes('set=b'));
  const title = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map((w) => w.getTitle()).find((t) => /Join Voice/.test(t)));
  const outline = await lw.evaluate(() => getComputedStyle(document.querySelector('#join')).outlineStyle);
  console.log('HINT title:', title, '| join button outline:', outline, '| url ch:', new URL(lw.url()).searchParams.get('ch'));
  if (!title || outline !== 'solid') fail('Join Voice not highlighted');
  await lw.click('#join'); await sleep(9000);
  const st = await get('/api/state');
  const l1 = (st.accounts || []).find((a) => a.account === 'l1');
  const title2 = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map((w) => w.getTitle()).filter((t) => /Discord ·/.test(t)));
  console.log('AFTER JOIN', JSON.stringify({ ch: l1?.channel?.name, capturing: l1?.capturing }), JSON.stringify(title2));
  if (l1?.channel?.name !== 'Voice 4') fail('listener did not end up in the chosen channel');
  const again = await dash.evaluate(() => window.ctApp.transcribe('900000000000000001', '700000000000000002'));
  console.log('SECOND PICK (listener busy, main free)', JSON.stringify(again));
  if (again.account !== 'main') fail('second channel should go to the next free account');
  await app.evaluate(({ app }) => app.quit()); await sleep(3000);
}

(async () => {
  // 1. two accounts, two separate calls
  const a = await scenario('separate', 'set=b&me=2');
  if (a) {
    const byCh = (n) => a.lines.filter((l) => l.chName === n);
    const sideNames = new Set(byCh('Side room').map((l) => l.speaker));
    if (byCh('Side room').length < 4) fail('second call (listener) was not transcribed: ' + byCh('Side room').length + ' lines');
    if (byCh('Voice 4').length < 3) fail('first call (main) was not transcribed');
    for (const n of ['Erin', 'Frank']) if (![...sideNames].some((x) => x.startsWith(n))) fail('missing ' + n + ' in Side room');
    if ([...sideNames].some((x) => /Alice|Bob|Carol/.test(x))) fail('people from call 1 leaked into call 2');
    const texts = a.lines.map((l) => l.chName + l.speaker + l.text);
    if (new Set(texts).size !== texts.length) fail('duplicate lines');
    const sep = (a.calls || []).filter((c) => c.chName === 'Side room').length;
    if (sep !== 1) fail('Side room should be one call, got ' + sep);
    const v4 = (a.calls || []).filter((c) => c.chName === 'Voice 4').length;
    if (v4 !== 1) fail('Voice 4 got split into ' + v4 + ' calls (calls must not fragment when two run at once)');
    if (!(a.st?.accounts || []).every((x) => x.capturing)) fail('both accounts should transcribe (different channels)');
  }
  // 2. two accounts in the same call
  const b = await scenario('same', 'me=2');
  if (b) {
    const cap = (b.st?.accounts || []).filter((x) => x.capturing && x.inCall);
    if (cap.length !== 1 || cap[0].role !== 'listener') fail('exactly the listener should transcribe a shared channel: ' + JSON.stringify(b.st?.accounts?.map((x) => [x.role, x.capturing])));
    const texts = b.lines.map((l) => l.speaker + l.text);
    if (new Set(texts).size !== texts.length) fail('same call transcribed twice');
    if (b.lines.length < 4 || b.lines.length > 8) fail('expected the call once (5 utterances), got ' + b.lines.length + ' lines');
    const doubled = b.lines.filter((l) => /\b(\w+) \1 (\w+) \2\b/i.test(l.text));
    if (doubled.length) fail('audio from two accounts mixed together: ' + doubled.map((l) => l.text).join(' | '));
  }
  // 3. "Transcribe a channel": the free listener opens the channel with Join highlighted; you click Join.
  await transcribeFlow();
  site.close();
  console.log(failed ? '\nFAILED' : '\nPASS');
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
