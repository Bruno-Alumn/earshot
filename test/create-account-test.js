// "Create account": a new listener account opens Discord's sign-up page in its own window.
//   node test/create-account-test.js <scratch-dir>
const { _electron: electron } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const SCRATCH = path.resolve(process.argv[2]);
const DESKTOP = path.join(__dirname, '..', 'app', 'desktop');
const PORT = 8772;
const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = false;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) failed = true; };
const site = http.createServer((req, res) => {
  const u = req.url.split('?')[0];
  const f = path.join(__dirname, 'site', u === '/' ? 'index.html' : u.slice(1));
  if (!fs.existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': f.endsWith('.html') ? 'text/html' : 'audio/wav' }); fs.createReadStream(f).pipe(res);
}).listen(8897);
(async () => {
  const dir = path.join(SCRATCH, 'create-account'), ud = path.join(dir, 'userdata');
  fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([{ id: 'main', role: 'main', partition: 'persist:test-main', testQuery: 'me=1' }]));
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP],
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_DISCORD_URL: 'http://localhost:8897/', CT_USER_DATA: ud, CT_SERVER_ARGS: `--out ${path.join(dir, 'tx')}` } });
  let dash;
  for (let i = 0; i < 200 && !dash; i++) { dash = app.windows().find((w) => w.url().startsWith(`http://127.0.0.1:${PORT}`)); if (!dash) await sleep(300); }
  ok(!!dash, 'dashboard opened');
  await dash.waitForFunction(() => !!window.ctApp, null, { timeout: 30000 });
  const before = app.windows().length;
  const res = await dash.evaluate(() => window.ctApp.addAccount('create'));
  ok(res && res.role === 'listener', `new account is a listener: ${JSON.stringify(res)}`);
  let w;
  for (let i = 0; i < 50 && !w; i++) { w = app.windows().find((x) => x.url().includes('page=register')); if (!w) await sleep(200); }
  ok(!!w && app.windows().length === before + 1, 'its window opens on the sign-up page');
  const info = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().filter((x) => x.webContents.getURL().includes('page=register')).map((x) => ({ title: x.getTitle(), visible: x.isVisible(), muted: x.webContents.isAudioMuted() }))[0]);
  ok(info && /sign up/.test(info.title) && info.muted, `window: ${JSON.stringify(info)}`);
  const saved = JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8'));
  ok(saved.length === 2 && saved[1].role === 'listener' && saved[1].muted && saved[1].partition !== saved[0].partition, 'saved with its own sign-in, sound off');
  // leaving the sign-up page puts the usual title back
  await w.evaluate(() => history.pushState({}, '', '/channels/@me'));
  await sleep(500);
  const t2 = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().map((x) => x.getTitle()).find((t) => /listener/.test(t)));
  ok(!!t2 && /^Discord · /.test(t2), `title after signing up: "${t2}"`);
  // Add account still opens the sign-in page
  await dash.evaluate(() => window.ctApp.addAccount());
  let w2; for (let i = 0; i < 50 && !w2; i++) { w2 = app.windows().find((x) => x.url().includes('page=login')); if (!w2) await sleep(200); }
  ok(!!w2, 'Add account opens the sign-in page');
  // the Settings buttons are there
  const btns = await dash.evaluate(() => { const d = document.createElement('div'); d.innerHTML = accountsHtml(true); return [...d.querySelectorAll('[data-desk]')].map((b) => b.textContent); });
  ok(btns.includes('Create account') && btns.includes('Add account'), `buttons: ${btns.join(', ')}`);
  await app.close(); site.close();
  console.log(failed ? 'FAIL' : 'PASS'); process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
