// Webshare proxies in the real Electron app, against a fake Webshare API:
// save the API key (checked, encrypted), list proxies without passwords, give an
// account one, and create an account with a proxy set before its window opens
// (through the bridge and through the dialog).
//   node test/webshare-app-test.js <scratch-dir>
const { _electron: electron } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const SCRATCH = path.resolve(process.argv[2] || fs.mkdtempSync(path.join(require('os').tmpdir(), 'ct-ws-')));
fs.mkdirSync(SCRATCH, { recursive: true });
const DESKTOP = path.join(__dirname, '..', 'app', 'desktop');
const PORT = 8781, API = 8797;
const exe = process.env.EXE || (process.platform === 'darwin'
  ? path.join(DESKTOP, 'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron')
  : path.join(DESKTOP, 'node_modules', 'electron', 'dist', 'electron.exe'));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failed = 0;
const fail = (m) => { console.log('FAIL', m); failed++; };
const ok = (c, m) => (c ? console.log('ok  ', m) : fail(m));

const KEY = 'goodkey-1234567890';
const PROXIES = [
  { id: 'd-1', username: 'wsuser1', password: 'wspass-ONE', proxy_address: '10.1.1.1', port: 6001, valid: true, country_code: 'US', city_name: 'New York' },
  { id: 'd-2', username: 'wsuser2', password: 'wspass-TWO', proxy_address: '10.2.2.2', port: 6002, valid: true, country_code: 'DE', city_name: 'Berlin' },
  { id: 'd-3', username: 'wsuser3', password: 'wspass-THREE', proxy_address: '10.3.3.3', port: 6003, valid: false, country_code: 'US', city_name: 'Dallas' },
  { id: 'd-4', username: 'wsuser4', password: 'wspass-FOUR', proxy_address: '10.4.4.4', port: 6004, valid: true, country_code: 'FR', city_name: 'Paris' },
];
let calls = 0;
const api = http.createServer((req, res) => {
  calls++;
  const u = new URL(req.url, `http://127.0.0.1:${API}`);
  if (req.headers.authorization !== `Token ${KEY}`) { res.writeHead(401, { 'content-type': 'application/json' }); res.end('{"detail":"Invalid token."}'); return; }
  if (u.pathname !== '/api/v2/proxy/list/' || u.searchParams.get('mode') !== 'direct') { res.writeHead(400); res.end('{}'); return; }
  const page = Number(u.searchParams.get('page')), size = 2;            // two pages, to check paging
  const results = PROXIES.slice((page - 1) * size, page * size);
  const next = page * size < PROXIES.length ? `http://127.0.0.1:${API}/api/v2/proxy/list/?mode=direct&page=${page + 1}&page_size=100` : null;
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ count: PROXIES.length, next, previous: null, results }));
}).listen(API);

(async () => {
  const ud = path.join(SCRATCH, 'userdata-ws');
  fs.rmSync(ud, { recursive: true, force: true }); fs.mkdirSync(ud, { recursive: true });
  fs.writeFileSync(path.join(ud, 'accounts.json'), JSON.stringify([
    { id: 'main', role: 'main', partition: 'persist:wst-main' },
    { id: 'l1', role: 'listener', partition: 'persist:wst-l1', muted: true },
  ]));
  const app = await electron.launch({ executablePath: exe, args: [DESKTOP], timeout: 120000,
    env: { ...process.env, CT_TEST: '1', CT_PORT: String(PORT), CT_USER_DATA: ud, CT_WEBSHARE_API: `http://127.0.0.1:${API}`,
      CT_DISCORD_URL: 'http://127.0.0.1:59998/', CT_SERVER_ARGS: `--out ${path.join(SCRATCH, 'tx-ws')}` } });
  const logs = []; app.process().stdout.on('data', (d) => logs.push(String(d))); app.process().stderr.on('data', (d) => logs.push(String(d)));
  let dash = null;
  for (let i = 0; i < 150 && !dash; i++) { dash = app.windows().find((w) => w.url().startsWith(`http://127.0.0.1:${PORT}`)) || null; if (!dash) await sleep(200); }
  if (!dash) { fail('dashboard window missing'); await app.close(); api.close(); return; }
  for (let i = 0; i < 60 && !(await dash.evaluate(() => typeof window.ctApp?.webshare === 'function').catch(() => false)); i++) await sleep(250);
  const resolve = (p) => app.evaluate(async ({ session }, part) => session.fromPartition(part).resolveProxy('https://discord.com'), p);
  const C = (fn, ...a) => dash.evaluate(fn, ...a);

  ok(!(await C(() => window.ctApp.webshare())).hasKey, 'no key at first');
  const bad = await C(() => window.ctApp.setWebshareKey('wrongkey-123456'));
  ok(/didn't accept/.test(bad?.error || ''), `a wrong key is refused: ${bad?.error}`);
  ok(!fs.existsSync(path.join(ud, 'webshare.json')), 'a wrong key is not saved');
  const good = await C((k) => window.ctApp.setWebshareKey(k), KEY);
  ok(good?.ok && good.count === 4, `the key is checked and saved (${good?.count} proxies, over two pages)`);
  const enc = await app.evaluate(({ safeStorage }) => { try { return safeStorage.isEncryptionAvailable(); } catch { return false; } });
  const wsFile = fs.readFileSync(path.join(ud, 'webshare.json'), 'utf8');
  ok(!enc || !wsFile.includes(KEY), enc ? 'the key is stored encrypted' : 'no OS encryption here: stored as is');

  const list = await C(() => window.ctApp.webshareList(false));
  const flat = JSON.stringify(list);
  ok(list?.proxies?.length === 4, 'the list has every proxy');
  ok(!/wspass|wsuser/.test(flat) && !flat.includes(KEY), 'the dashboard never sees usernames, passwords or the key');
  ok(list.proxies.find((p) => p.id === 'd-3')?.valid === false, 'a broken proxy is marked');

  // give the listener one
  const use = await C(() => window.ctApp.webshareUse('l1', 'd-2'));
  ok(!use?.error, 'an account takes a Webshare proxy ' + (use?.error || ''));
  ok(/PROXY 10\.2\.2\.2:6002/.test(await resolve('persist:wst-l1')), 'its session uses that proxy');
  const l2 = await C(() => window.ctApp.webshareList(false));
  ok(l2.proxies.find((p) => p.id === 'd-2')?.usedBy?.some((u) => u.id === 'l1'), 'the list says which account uses it');
  const shown = (await C(() => window.ctApp.accounts())).accounts.find((a) => a.id === 'l1');
  ok(shown.proxy === 'http://wsuser2:***@10.2.2.2:6002', `the account shows it without the password: ${shown.proxy}`);

  // create an account with a proxy, through the bridge: the proxy is on its session before the window loads
  const n0 = JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8')).length;
  const badNew = await C(() => window.ctApp.addAccount('create', { value: 'socks5://' }));
  ok(badNew?.error && JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8')).length === n0, 'a bad proxy stops the new account before anything opens');
  const made = await C(() => window.ctApp.addAccount('create', { webshare: 'd-1' }));
  ok(made?.id && !made.error, 'an account is created with a Webshare proxy');
  const part = JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8')).find((a) => a.id === made.id)?.partition;
  ok(/PROXY 10\.1\.1\.1:6001/.test(await resolve(part)), 'the new account\'s session has the proxy');

  // the same through the dialog: pick the country, pick a free one, open
  await C(() => openNewAccount('create'));
  await dash.waitForSelector('.wsrow', { timeout: 15000 });
  const rows = await C(() => [...document.querySelectorAll('.wsrow')].map((r) => r.innerText.replace(/\s+/g, ' ')));
  ok(rows.length === 4 && rows.some((r) => /used by/.test(r)) && rows.some((r) => /not working/.test(r)), `the dialog lists the proxies: ${rows.join(' | ')}`);
  await dash.click('[data-ws="random"]'); if (process.env.SHOT) await dash.screenshot({ path: process.env.SHOT });
  const pick = await C(() => document.querySelector('.wsrow.on')?.dataset.wsid);
  ok(pick === 'd-4', `"Pick a free one" skips broken and used proxies (${pick})`);
  // account details: a strong password is suggested, a bad email is caught, everything is saved encrypted
  const pw1 = await C(() => document.querySelector('input[data-k="password"]').value);
  ok(pw1.length === 20 && /[A-Z]/.test(pw1) && /[a-z]/.test(pw1) && /\d/.test(pw1) && /[^A-Za-z0-9]/.test(pw1), 'a strong 20-character password is suggested');
  await dash.click('[data-d="gen"]');
  const pw = await C(() => document.querySelector('input[data-k="password"]').value);
  ok(pw !== pw1 && pw.length === 20, 'New password suggests a different one');
  await dash.fill('input[data-k="email"]', 'not-an-email');
  await dash.click('[data-a="go"]'); await sleep(300);
  ok(/email/.test(await C(() => document.querySelector('#pxymsg').textContent)) && JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8')).length === n0 + 1, 'a bad email is caught before anything opens');
  await dash.fill('input[data-k="email"]', 'tester@example.test');
  await dash.fill('input[data-k="username"]', 'tester_01');
  await dash.fill('input[data-k="displayName"]', 'Tester One');
  await dash.click('[data-a="go"]'); await sleep(1500);
  const accRaw = fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8');
  const newest = JSON.parse(accRaw).slice(-1)[0];
  ok(!enc || (!accRaw.includes(pw) && !accRaw.includes('tester@example.test') && newest.credsEnc), 'the details are stored encrypted');
  const listJson = JSON.stringify(await C(() => window.ctApp.accounts()));
  ok(!listJson.includes(pw) && !listJson.includes('tester@example.test') && /"hasDetails":true/.test(listJson), 'the account list never carries them');
  const det = await C((id) => window.ctApp.accountDetails(id), newest.id);
  ok(det?.creds?.password === pw && det.creds.email === 'tester@example.test' && det.creds.username === 'tester_01' && det.creds.displayName === 'Tester One', 'Details… gets them back');
  await C((id) => window.ctApp.setAccountDetails(id, { email: 'new@example.test', username: 'tester_01', displayName: 'Tester One', password: 'changed-PW-123' }), newest.id);
  ok((await C((id) => window.ctApp.accountDetails(id), newest.id)).creds.email === 'new@example.test', 'and they can be changed');
  const acc = JSON.parse(fs.readFileSync(path.join(ud, 'accounts.json'), 'utf8'));
  ok(acc.length === n0 + 2 && !(await C(() => !!document.querySelector('.modal.pxmodal'))), 'the dialog creates the account and closes');

  // the menu item asks first
  await app.evaluate(({ Menu }) => Menu.getApplicationMenu()?.items.flatMap((i) => i.submenu?.items || []).find((i) => /Create a New Discord Account/.test(i.label))?.click());
  await sleep(800);
  ok(await C(() => /Create a Discord account/.test(document.querySelector('#modal-h')?.textContent || '')), 'the menu item opens the same dialog');
  await C(() => document.querySelector('[data-a="close"]')?.click());

  // remove the key
  const rm = await C(() => window.ctApp.setWebshareKey(''));
  ok(rm?.ok && !fs.existsSync(path.join(ud, 'webshare.json')), 'Remove key deletes it');
  ok(!logs.join('').includes(KEY), 'the key never appears in the app log');

  await app.evaluate(({ app }) => app.quit()); await sleep(2500); api.close();
  if (failed) { console.log(`\n${failed} FAILED`); console.log(logs.join('').split('\n').slice(-25).join('\n')); process.exitCode = 1; }
  else console.log('\nPASS');
})().catch((e) => { console.error(e); api.close(); process.exit(1); });
