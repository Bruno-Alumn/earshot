// Settings → Sync in the real Electron app: the section shows, a pairing code
// appears and can be cancelled, a bad code gives a plain error, and pairing
// with a second sync link (as another computer) lists it and syncs.
//   node test/sync-app-test.js <scratch-dir>
const { _electron: electron } = require('playwright');
const fs = require('fs'), path = require('path');
const { createSync } = require('../app/desktop/sync.js');
const SCRATCH = path.resolve(process.argv[2] || fs.mkdtempSync(path.join(require('os').tmpdir(), 'ct-syncapp-')));
fs.mkdirSync(SCRATCH, { recursive: true });
const DESKTOP = path.join(__dirname, '..', 'app', 'desktop');
const PORT = 8782, SPORT = 8799;
const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const ok = (c, m) => { console.log((c ? 'ok   ' : 'FAIL ') + m); if (!c) failed++; };

(async () => {
  const ud = path.join(SCRATCH, 'userdata-sync');
  fs.rmSync(ud, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([{ id: 'main', role: 'main', partition: 'persist:syt-main' }]));
  const tx = path.join(SCRATCH, 'tx-sync', 'transcripts');
  fs.rmSync(path.dirname(tx), { recursive: true, force: true }); fs.mkdirSync(tx, { recursive: true });
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP], timeout: 120000,
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_USER_DATA: ud, CT_SYNC_PORT: String(SPORT),
      CT_DISCORD_URL: 'http://127.0.0.1:59998/', CT_SERVER_ARGS: `--out ${tx} ${process.env.SERVER_ARGS || ''}`.trim() } });
  let dash = null;
  for (let i = 0; i < 150 && !dash; i++) { dash = app.windows().find((w) => w.url().startsWith(`http://127.0.0.1:${PORT}`)) || null; if (!dash) await sleep(200); }
  if (!dash) { ok(false, 'dashboard window'); await app.close(); process.exit(1); }
  const errs = []; dash.on('pageerror', (e) => errs.push(e.message));
  for (let i = 0; i < 60 && !(await dash.evaluate(() => typeof window.ctApp?.sync === 'function').catch(() => false)); i++) await sleep(250);

  await dash.click('#rail [data-view="settings"]');
  await dash.waitForSelector('#sync-box .srow', { timeout: 30000 });
  ok(await dash.isVisible('[data-jump="sync"]'), 'Settings has a Sync section');
  const btn = await dash.textContent('#sync-box [data-s="pair"]');
  ok(btn === 'Show a pairing code', 'the button says what it does');

  await dash.fill('#sync-box [data-s="code"]', 'not a code');
  await dash.click('#sync-box [data-s="join"]');
  await dash.waitForFunction(() => /doesn't look right/.test(document.querySelector('#sync-msg')?.textContent || ''), null, { timeout: 5000 });
  ok(true, 'a bad code gives a plain error, and the button comes back');
  ok(!(await dash.isDisabled('#sync-box [data-s="join"]')), 'Pair is usable again');

  await dash.click('#sync-box [data-s="pair"]');
  await dash.waitForSelector('#sync-box .pcode', { timeout: 10000 });
  const code = await dash.textContent('#sync-box .pcode');
  ok(/^\S+ [A-Z0-9]{4}(-[A-Z0-9]{4}){3}$/.test(code), `a code shows (${code.replace(/ .*/, ' ****')})`);

  // another "computer" types it
  fs.rmSync(path.join(SCRATCH, 'other-sync.json'), { force: true });
  const other = createSync({ stateFile: path.join(SCRATCH, 'other-sync.json'), serverBase: `http://127.0.0.1:${PORT}`, port: SPORT + 1, name: 'Other PC', log: (m) => console.log('  [other] ' + m) });
  const j = await other.join(`127.0.0.1:${SPORT} ${code.split(' ')[1]}`);
  ok(j.ok, `the other computer pairs (${j.error || j.name})`);
  await dash.waitForFunction(() => [...document.querySelectorAll('#sync-box .srow b')].some((b) => b.textContent === 'Other PC'), null, { timeout: 10000 });
  ok(!(await dash.$('#sync-box .pcode')), 'the code is gone once used, and the computer is listed');
  ok(await dash.isVisible('#sync-box [data-s="now"]') && await dash.isVisible('#sync-box [data-s="unpair"]'), 'Sync now and Unpair are there');
  const r = await other.syncNow();
  const res = Object.values(r)[0];
  ok(res && !res.error, `a round runs against the app (${res?.error || 'ok'})`);
  await dash.waitForFunction(() => /synced with this computer/.test(document.querySelector('#sync-box')?.textContent || ''), null, { timeout: 5000 }).then(() => ok(true, 'a round the other computer started shows here')).catch(() => ok(false, 'a round the other computer started shows here'));
  ok(await dash.isVisible('#set-sync-h') && (await dash.textContent('#set-sync-h')) === 'Sync', 'the section has its heading');
  await dash.screenshot({ path: path.join(SCRATCH, 'sync-settings.png') });

  ok(!errs.length, `no page errors ${errs.join(' | ')}`);
  other.stop();
  await app.close();
  console.log(failed ? '\nFAILED' : '\nPASS');
  process.exit(failed ? 1 : 0);
})().catch((e) => { console.error(e); process.exit(1); });
