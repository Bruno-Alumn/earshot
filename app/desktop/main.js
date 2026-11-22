// Earshot desktop app (macOS + Windows, one codebase).
//
//  - Owns the local transcription server: starts it on launch, stops it on
//    quit (the server also watches our pid via --parent-pid).
//  - Dashboard window: the server's live dashboard.
//  - Discord window: discord.com in a built-in Chromium. page-hook.js is
//    injected before Discord's scripts run and each person's audio is relayed
//    to the server here.
const { app, BrowserWindow, Menu, Tray, dialog, ipcMain, nativeImage, powerSaveBlocker,
  session, shell, desktopCapturer, systemPreferences, safeStorage } = require('electron');
const { spawn, execFile } = require('child_process');
const fs = require('fs');
const path = require('path');
const http = require('http');
const { parseProxy, redactProxy, proxyLogin } = require('./proxy');
const { candidates, isIdle, connectedChannelId, isAlone, busiestFirst, classifyLeave, guildOf } = require('./autojoin');

// Auto-join: a free account hops into an open, flagged voice channel and moves
// on after a stretch of silence. Timings are overridable for the test harness.
const AUTOJOIN_IDLE = Number(process.env.CT_AUTOJOIN_IDLE) || 180;          // seconds of silence before moving on
const AUTOJOIN_COOLDOWN = Number(process.env.CT_AUTOJOIN_COOLDOWN) || 120000; // ms to avoid a channel we just left or failed to join
const AUTOJOIN_ALONE = Number(process.env.CT_AUTOJOIN_ALONE) || 60;         // seconds alone in a channel before moving on
const AUTOJOIN_TICK = Number(process.env.CT_AUTOJOIN_TICK) || 8000;         // how often the controller re-evaluates

const ROOT = path.resolve(__dirname, '..', '..');
const SERVER_DIR = path.join(ROOT, 'server');
const IS_WIN = process.platform === 'win32';
const IS_MAC = process.platform === 'darwin';
const TEST = process.env.CT_TEST === '1';                        // test harness only
const PORT = Number(process.env.CT_PORT || 8765);
const BASE = `http://127.0.0.1:${PORT}/`;
const LOG = path.join(ROOT, TEST ? `server-test-${PORT}.log` : 'server.log');
const ICON_PNG = path.join(ROOT, 'app', 'windows', 'icon.png');
const ICON_ICO = path.join(ROOT, 'app', 'windows', 'icon.ico');
const BG = { dark: '#111110', light: '#f4f4f2' };

app.setName('Earshot');
// Keep using the data folder from before the rename (sign-ins, accounts, settings).
{ const old = path.join(app.getPath('appData'), 'Call Transcriber');
  if (!TEST && fs.existsSync(old)) app.setPath('userData', old); }
if (TEST && process.env.CT_USER_DATA) app.setPath('userData', process.env.CT_USER_DATA);
if (IS_WIN) app.setAppUserModelId('local.calltranscriber');
if (!app.requestSingleInstanceLock()) { app.quit(); process.exit(0); }

let dash = null, tray = null;
let server = null, stoppedByUser = false, quitting = false, paused = false;

// ----------------------------------------------------------------- helpers
const isDiscordUrl = (u) => {
  try {
    const x = new URL(u);
    if (TEST && process.env.CT_DISCORD_URL && x.origin === new URL(process.env.CT_DISCORD_URL).origin) return true;
    return x.protocol === 'https:' && /(^|\.)(discord\.com|discordapp\.com)$/.test(x.hostname);
  } catch { return false; }
};
const isDashUrl = (u) => typeof u === 'string' && (u.startsWith(BASE) || u.startsWith('data:text/html'));

function request(pathname, { method = 'GET', timeout = 1000, app: appHeader = false } = {}) {
  return new Promise((resolve) => {
    const req = http.request(BASE + pathname.replace(/^\//, ''), { method, timeout, agent: false, headers: { ...(appHeader ? { 'X-CT': '1' } : {}), ...(method === 'POST' ? { 'Content-Length': 0 } : {}) } }, (res) => {
      let body = ''; res.on('data', (d) => (body += d)); res.on('end', () => resolve({ status: res.statusCode, body }));
    });
    req.on('error', () => resolve(null)); req.on('timeout', () => { req.destroy(); resolve(null); });
    req.end();
  });
}
const serverUp = async () => { const r = await request('api/state'); return !!r && r.status === 200 && r.body.startsWith('{'); };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const alive = (pid) => { try { process.kill(pid, 0); return true; } catch { return false; } };

function run(cmd, args, opts = {}) {
  return new Promise((resolve) => {
    const p = spawn(cmd, args, { windowsHide: true, stdio: 'ignore', ...opts });
    p.on('error', () => resolve(-1)); p.on('exit', (code) => resolve(code ?? -1));
  });
}

function killTree(pid) {
  if (!pid || !alive(pid)) return Promise.resolve();
  if (IS_WIN) return run('taskkill', ['/PID', String(pid), '/T', '/F']);   // venv python.exe is a stub + child
  return (async () => {
    try { process.kill(pid, 'SIGINT'); } catch {}                    // lets the server flush pending lines
    for (let i = 0; i < 40 && alive(pid); i++) await sleep(100);
    if (alive(pid)) try { process.kill(pid, 'SIGKILL'); } catch {}
  })();
}

function listeningPids() {
  return new Promise((resolve) => {
    if (IS_WIN) {
      execFile('netstat', ['-ano', '-p', 'tcp'], { windowsHide: true }, (_e, out = '') => resolve([...new Set(out.split('\n')
        .map((l) => l.trim().split(/\s+/)).filter((c) => c[1]?.endsWith(':' + PORT) && c[3] === 'LISTENING').map((c) => Number(c[4])))]));
    } else {
      execFile('/usr/sbin/lsof', ['-nP', '-t', `-iTCP:${PORT}`, '-sTCP:LISTEN'], (_e, out = '') =>
        resolve(out.split('\n').map(Number).filter(Boolean)));
    }
  });
}

// ----------------------------------------------------------------- server
function serverPython() {
  return IS_WIN ? path.join(SERVER_DIR, '.venv', 'Scripts', 'python.exe') : path.join(SERVER_DIR, '.venv', 'bin', 'python');
}

async function stopStrays() {
  // Our own server already on the port (started from a terminal, or left over): stop it.
  if (!(await serverUp())) return;
  await request('api/shutdown', { method: 'POST', app: true });
  for (let i = 0; i < 50 && (await serverUp()); i++) await sleep(100);
  if (await serverUp()) for (const pid of await listeningPids()) if (pid !== server?.pid) await killTree(pid);
}

async function startServer() {
  stoppedByUser = false;
  if (server && server.exitCode === null) { showDashboard(); return; }
  splash('Starting the transcription server…');
  await stopStrays();
  const py = serverPython();
  if (!fs.existsSync(py) || !fs.existsSync(path.join(SERVER_DIR, 'server.py'))) {
    splash(IS_WIN ? "Couldn't find the server. Run install.cmd again." : "Couldn't find the server. See README.md (Setup).", { error: true }); return;
  }
  const log = fs.openSync(LOG, 'w');
  const extra = TEST && process.env.CT_SERVER_ARGS ? process.env.CT_SERVER_ARGS.split(' ').filter(Boolean) : [];
  const p = spawn(py, ['-u', 'server.py', '--port', String(PORT), '--parent-pid', String(process.pid), ...extra], {
    cwd: SERVER_DIR, stdio: ['ignore', log, log], windowsHide: true,
    env: { ...process.env, PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' },
  });
  server = p;
  p.on('exit', (code) => {
    fs.closeSync(log);
    if (server !== p) return;
    server = null;
    if (quitting) return;
    if (stoppedByUser) splash('The server is stopped. Nothing is being transcribed.', { offerStart: true });
    else splash(`The server stopped unexpectedly (exit ${code}). See the server log.`, { error: true, offerStart: true });
    updateTray();
  });
  for (let i = 0; i < 480; i++) {                         // model loading can take a while the first time
    if (server !== p || p.exitCode !== null) return;
    if (await serverUp()) { showDashboard(); updateTray(); return; }
    await sleep(500);
  }
  splash('The server is taking too long to start. See the server log.', { error: true });
}

async function stopServer({ quiet = false } = {}) {
  const p = server;
  if (!p || p.exitCode !== null) { if (!quiet) splash('The server is stopped. Nothing is being transcribed.', { offerStart: true }); return; }
  stoppedByUser = true;
  if (!quiet) splash('Stopping the server…');
  await request('api/shutdown', { method: 'POST', app: true, timeout: 1500 });   // clean stop: flushes pending lines
  for (let i = 0; i < 50 && p.exitCode === null; i++) await sleep(100);
  if (p.exitCode === null) await killTree(p.pid);
}

async function restartServer() {
  if (server && server.exitCode === null) { splash('Restarting the server…'); await stopServer({ quiet: true }); server = null; }
  return startServer();
}

// ----------------------------------------------------------------- accounts
// Each Discord account has its own window, sign-in (session partition) and
// connection to the server. "main" uses your real microphone and speakers;
// "listener" accounts get a silent microphone and start with sound off.
const accountsFile = () => path.join(app.getPath('userData'), 'accounts.json');
let accounts = [];                 // persisted: { id, role, partition, label, userId, muted, directory, directoryAt }
const rt = new Map();              // runtime per account id: { win, ws, wsOpen, capturing, status, statusAt, hint }

function loadAccounts() {
  let list = [];
  try { list = JSON.parse(fs.readFileSync(accountsFile(), 'utf8')); } catch {}
  if (!Array.isArray(list) || !list.length) list = [{ id: 'main', role: 'main', partition: 'persist:discord', muted: !!loadSettings().muted }];
  if (!list.some((a) => a.role === 'main')) list[0].role = 'main';
  for (const a of list) {                 // proxy credentials are encrypted at rest when the OS can
    if (!a.proxyEnc) continue;
    try { const v = safeStorage.decryptString(Buffer.from(a.proxyEnc, 'base64')); if (v) { a.proxy = v; delete a.proxyEnc; } }
    catch {}                              // keep it: the account shows a proxy that needs attention
  }
  return list;
}
function saveAccounts() {
  try {
    const enc = (() => { try { return safeStorage.isEncryptionAvailable(); } catch { return false; } })();
    const out = accounts.map(({ proxy, proxyEnc, ...rest }) => {
      if (!proxy) return proxyEnc ? { ...rest, proxyEnc } : rest;
      if (!enc) return { ...rest, proxy };
      try { return { ...rest, proxyEnc: safeStorage.encryptString(proxy).toString('base64') }; }
      catch { return { ...rest, proxy }; }
    });
    fs.writeFileSync(accountsFile(), JSON.stringify(out, null, 1));
  } catch {}
}
const acct = (id) => accounts.find((a) => a.id === id);
const mainAccount = () => accounts.find((a) => a.role === 'main') || accounts[0];

// Per-channel flags for auto-join, keyed by channel id: { auto, fav }.
//   auto — a free account may automatically join this channel while it is open
//          (someone is in it) and not full (the blue icon).
//   fav  — a favourite; it sorts to the top of the channel list (the yellow star).
let channelFlags = {};
const channelsFile = () => path.join(app.getPath('userData'), 'channels.json');
function loadChannelFlags() {
  try {
    const raw = JSON.parse(fs.readFileSync(channelsFile(), 'utf8'));
    channelFlags = {};
    for (const [id, f] of Object.entries(raw || {})) if (f && (f.auto || f.fav)) channelFlags[String(id)] = { auto: !!f.auto, fav: !!f.fav };
  } catch { channelFlags = {}; }
}
function saveChannelFlags() {
  try { fs.writeFileSync(channelsFile(), JSON.stringify(channelFlags)); } catch {}
}
function setChannelFlag(id, key, on) {
  if (key !== 'auto' && key !== 'fav') return channelFlags;
  const f = channelFlags[id] || { auto: false, fav: false };
  f[key] = !!on;
  if (!f.auto && !f.fav) delete channelFlags[id]; else channelFlags[id] = f;
  saveChannelFlags();
  for (const a of accounts) sendWatchList(a);   // the hook only reports occupancy for flagged channels
  broadcast();
  return channelFlags;
}
const R = (id) => { if (!rt.has(id)) rt.set(id, { win: null, ws: null, wsOpen: false, capturing: true, status: null, statusAt: 0 }); return rt.get(id); };
const winOf = (id) => { const w = rt.get(id)?.win; return w && !w.isDestroyed() ? w : null; };
const nameOf = (a) => a.label || (a.role === 'main' ? 'Main account' : 'Listener (not signed in)');

function accountOfSender(e) {
  for (const a of accounts) {
    const w = winOf(a.id);
    if (w && e.sender === w.webContents && e.senderFrame === w.webContents.mainFrame && isDiscordUrl(e.senderFrame.url)) return a;
  }
  return null;
}

function accountOfWebContents(wc) {
  if (!wc) return null;
  for (const a of accounts) {
    const w = winOf(a.id);
    if (w && w.webContents === wc) return a;
  }
  return null;
}

function connect(a) {
  const r = R(a.id);
  if (quitting || r.ws || !winOf(a.id)) return;
  let sock;
  try { sock = new WebSocket(`ws://127.0.0.1:${PORT}`); } catch { setTimeout(() => connect(a), 2000); return; }
  r.ws = sock;
  sock.onopen = () => { r.wsOpen = true; hello(a); broadcast(); };
  sock.onclose = () => {
    if (r.ws === sock) { r.ws = null; r.wsOpen = false; broadcast(); }
    if (winOf(a.id) && acct(a.id)) setTimeout(() => connect(a), 2000);
  };
  sock.onerror = () => {};
  sock.onmessage = (m) => {
    try { const msg = JSON.parse(m.data); if (msg.type === 'capture') { r.capturing = !!msg.on; broadcast(); } } catch {}
  };
}
const sendTo = (a, obj) => { const r = rt.get(a.id); if (r?.wsOpen && r.ws?.readyState === 1) try { r.ws.send(JSON.stringify(obj)); } catch {} };
const hello = (a) => sendTo(a, { type: 'hello', account: a.id, role: a.role, userId: a.userId || null, label: a.label || null });

function broadcast() {
  for (const a of accounts) {
    const r = rt.get(a.id), w = winOf(a.id);
    if (w) w.webContents.send('ct:status', !!(r?.wsOpen && !paused && r.capturing !== false));
  }
  if (dash && !dash.isDestroyed()) dash.webContents.send('ct:app-state', appState());
  updateTray();
}
function accountSummary(a) {
  const r = rt.get(a.id) || {}, s = r.status || {}, fresh = Date.now() - (r.statusAt || 0) < 15000;
  const self = fresh ? s.me?.self : null;
  const px = a.proxyEnc ? { ok: false } : parseProxy(a.proxy);
  return { id: a.id, role: a.role, label: nameOf(a), signedIn: !!a.userId, userId: a.userId || null,
    open: !!winOf(a.id), visible: !!winOf(a.id)?.isVisible(), connected: !!r.wsOpen, muted: !!a.muted,
    inCall: fresh && !!s.pcs && !!s.tracks, channel: fresh ? s.channel || null : null, capturing: r.capturing !== false,
    deafened: !!(self && (self.deaf || self.selfDeaf)), hint: r.hint || null,
    autoStopped: Object.values(r.autoStop || {}),
    guilds: (a.directory || []).length,
    proxy: px.ok && (px.proxy || px.pac) ? redactProxy(a.proxy) : '',
    proxyMode: !px.ok ? 'invalid' : px.proxy ? 'proxy' : px.pac ? 'pac' : px.direct ? 'direct' : 'system' };
}
const appState = () => {
  const m = mainAccount();
  return { discordOpen: !!winOf(m.id)?.isVisible(), paused, muted: !!m.muted, autoJoin: appSettings().autoJoin,
    connected: accounts.some((a) => rt.get(a.id)?.wsOpen), accounts: accounts.map(accountSummary) };
};

// App settings that persist (userData/settings.json).
const APP_DEFAULTS = { openAtLogin: false, showCallWindowOnStart: true, autoJoin: false };
function loadSettings() { try { return JSON.parse(fs.readFileSync(path.join(app.getPath('userData'), 'settings.json'), 'utf8')); } catch { return {}; } }
function appSettings() { const s = loadSettings(); return Object.fromEntries(Object.entries(APP_DEFAULTS).map(([k, v]) => [k, k in s ? !!s[k] : v])); }
function setAppSetting(key, value) {
  if (!(key in APP_DEFAULTS)) return appSettings();
  try { fs.writeFileSync(path.join(app.getPath('userData'), 'settings.json'), JSON.stringify({ ...loadSettings(), [key]: !!value })); } catch {}
  if (key === 'autoJoin') { for (const r of rt.values()) { r.autoStop = {}; r.lastIn = null; } setTimeout(broadcast, 0); }
  if (key === 'openAtLogin' && !TEST) {
    // Windows runs the app as electron.exe <app folder>; the Mac bundle starts itself.
    app.setLoginItemSettings(IS_WIN ? { openAtLogin: !!value, path: process.execPath, args: [__dirname] } : { openAtLogin: !!value });
  }
  return appSettings();
}

// Sound on/off for you, per account. The hook reads each person's audio before
// it reaches the speakers, so a muted account is still transcribed.
function setMuted(on, id = mainAccount().id) {
  const a = acct(id); if (!a) return;
  a.muted = !!on; saveAccounts();
  winOf(a.id)?.webContents.setAudioMuted(a.muted);
  broadcast(); buildMenu();
}

// Names learned in earlier calls (kept by the app between sessions).
const namesFile = () => path.join(app.getPath('userData'), 'names.json');
const loadNames = () => { try { return JSON.parse(fs.readFileSync(namesFile(), 'utf8')); } catch { return {}; } };

const fromDash = (e) => dash && !dash.isDestroyed() && e.sender === dash.webContents &&
  e.senderFrame === dash.webContents.mainFrame && isDashUrl(e.senderFrame.url);

// Listener accounts: replace the page's camera/microphone access with a silent
// microphone, before Discord's own code runs. The real devices are never opened
// (their session also refuses media permission, so a bypass fails closed).
const LISTENER_PRELUDE = `(() => { try {
  const deny = () => Promise.reject(new DOMException('Listener accounts have no camera or microphone', 'NotAllowedError'));
  let ctx; const silent = () => { const C = window.AudioContext || window.webkitAudioContext; ctx ||= new C(); return ctx.createMediaStreamDestination().stream; };   // one context, reused
  const md = MediaDevices.prototype;
  md.getUserMedia = function (c) { return c && c.audio ? Promise.resolve(silent()) : deny(); };
  md.getDisplayMedia = function () { return deny(); };
  window.__ctListener = true;
} catch (e) {} })();`;
// No account ever gets your microphone: Discord gets a silent one instead of the
// real one, before its own code runs, so nothing you say can reach a call even
// if Discord shows you unmuted. Camera and screen sharing still work on your own
// account. (Every session also refuses real-microphone requests, so a bypass fails closed.)
const MIC_OFF_PRELUDE = `(() => { try {
  let ctx; const silent = () => { const C = window.AudioContext || window.webkitAudioContext; ctx ||= new C(); return ctx.createMediaStreamDestination().stream; };   // one context, reused
  const md = MediaDevices.prototype, real = md.getUserMedia;
  md.getUserMedia = function (c) {
    if (!c || !c.audio) return real.call(this, c);
    if (!c.video) return Promise.resolve(silent());
    return real.call(this, { video: c.video }).then((v) => { const s = silent(); v.getVideoTracks().forEach((t) => s.addTrack(t)); return s; });
  };
  window.__ctMicOff = true;
} catch (e) {} })();`;
let hookSource = '';
ipcMain.on('ct:hook-source', (e) => {
  const a = accounts.find((x) => winOf(x.id) && e.sender === winOf(x.id).webContents);
  if (!a) { e.returnValue = {}; return; }
  hookSource ||= fs.readFileSync(path.join(__dirname, 'page-hook.js'), 'utf8');
  const prelude = a.role === 'listener' ? LISTENER_PRELUDE : MIC_OFF_PRELUDE;
  e.returnValue = { role: a.role, src: (prelude ? prelude + '\n' : '') + hookSource };
});
ipcMain.on('ct:hello', (e) => {
  const a = accountOfSender(e); if (!a) return;
  const r = R(a.id);
  e.sender.send('ct:status', !!(r.wsOpen && !paused && r.capturing !== false));
  e.sender.send('ct:names-init', loadNames());
  setTimeout(() => winOf(a.id)?.webContents.send('ct:want-directory'), 8000);
});
ipcMain.on('ct:audio', (e, m) => {
  const a = accountOfSender(e);
  if (!a || paused || !m?.pcm || R(a.id).capturing === false) return;
  const pcm = Buffer.from(m.pcm instanceof ArrayBuffer ? new Uint8Array(m.pcm) : m.pcm);
  sendTo(a, { type: 'track-audio', user: String(m.user), name: m.name ?? null, ch: m.ch ?? null, t: m.t, pcm: pcm.toString('base64') });
});
ipcMain.on('ct:hook-status', (e, s) => {
  const a = accountOfSender(e); if (!a || !s || typeof s !== 'object') return;
  const r = R(a.id);
  const wasIn = !!(r.status?.pcs && r.status?.tracks);
  r.status = s; r.statusAt = Date.now();
  const me = s.me;
  if (me?.id && (me.id !== a.userId || (me.name && me.name !== a.label))) {
    a.userId = me.id; a.label = me.name || a.label; saveAccounts(); hello(a); buildMenu(); broadcast();
    if (!r.hint) winOf(a.id)?.setTitle(`Discord · ${nameOf(a)}${a.role === 'listener' ? ' (listener)' : ''}`);
  }
  const inCall = !!(s.pcs && s.tracks);
  if (inCall && r.hint) { r.hint = null; clearJoinHint(a); }
  const deaf = !!(me?.self && (me.self.deaf || me.self.selfDeaf));
  if (inCall !== wasIn || deaf !== !!r.deaf) { r.deaf = deaf; broadcast(); }
  sendTo(a, { type: 'hook-status', s: { ...s, client: 'app', account: a.id, role: a.role }, paused });
});
ipcMain.on('ct:directory', (e, dir) => {
  const a = accountOfSender(e); if (!a || !Array.isArray(dir)) return;
  a.directory = dir.slice(0, 200).map((g) => ({ id: String(g.id), name: String(g.name || ''),
    channels: (g.channels || []).slice(0, 100).map((c) => ({ id: String(c.id), name: String(c.name || '') })) }));
  a.directoryAt = Date.now(); saveAccounts();
});
ipcMain.on('ct:names', (e, names) => {
  if (!accountOfSender(e) || !names || typeof names !== 'object') return;
  try { fs.writeFileSync(namesFile(), JSON.stringify({ ...loadNames(), ...names })); } catch {}
});
setInterval(() => { for (const a of accounts) winOf(a.id)?.webContents.send('ct:want-directory'); }, 10 * 60000);
setInterval(autoJoinTick, AUTOJOIN_TICK);

// ----------------------------------------------------------------- dashboard bridge
ipcMain.on('ct:server', (e, action) => {
  if (!fromDash(e)) return;
  ({ start: startServer, stop: stopServer, restart: restartServer })[action]?.();
});
ipcMain.handle('ct:save', async (e, name, content) => {
  if (!fromDash(e)) return false;
  const { canceled, filePath } = await dialog.showSaveDialog(dash, { defaultPath: path.join(app.getPath('downloads'), path.basename(name)) });
  if (canceled || !filePath) return false;
  fs.writeFileSync(filePath, content, 'utf8');
  return true;
});
ipcMain.on('ct:relaunch', (e, opts) => { if (fromDash(e)) relaunchAfterUpdate(opts || {}); });
ipcMain.on('ct:open-discord', (e, id) => { if (fromDash(e)) showDiscord(typeof id === 'string' ? id : undefined); });
ipcMain.on('ct:pause', (e, on) => { if (fromDash(e)) setPaused(!!on); });
ipcMain.on('ct:mute', (e, on, id) => { if (fromDash(e)) setMuted(!!on, typeof id === 'string' ? id : undefined); });
ipcMain.handle('ct:accounts', (e) => {
  if (!fromDash(e)) return null;
  // Every server/voice channel your accounts can see, with which accounts are in that server.
  const guilds = new Map();
  for (const a of accounts) for (const g of a.directory || []) {
    const cur = guilds.get(g.id) || { id: g.id, name: g.name, channels: new Map(), accounts: [] };
    cur.accounts.push(a.id);
    for (const c of g.channels) cur.channels.set(c.id, c);
    guilds.set(g.id, cur);
  }
  return { accounts: accounts.map(accountSummary),
    guilds: [...guilds.values()].map((g) => ({ ...g, channels: [...g.channels.values()] })).sort((x, y) => x.name.localeCompare(y.name)) };
});
ipcMain.on('ct:open-plugins', (e) => { if (fromDash(e)) { const d = path.join(ROOT, 'plugins'); fs.mkdirSync(d, { recursive: true }); shell.openPath(d); } });
ipcMain.handle('ct:app-settings', (e) => (fromDash(e) ? appSettings() : null));
ipcMain.handle('ct:set-app-setting', (e, key, value) => (fromDash(e) ? setAppSetting(String(key), !!value) : null));
ipcMain.handle('ct:transcribe', (e, guildId, channelId) => (fromDash(e) ? transcribeChannel(String(guildId), String(channelId)) : null));
ipcMain.handle('ct:add-account', (e) => (fromDash(e) ? addAccount() : null));
ipcMain.handle('ct:remove-account', (e, id) => (fromDash(e) ? removeAccount(String(id)) : null));
ipcMain.handle('ct:set-role', (e, id, role) => (fromDash(e) ? setRole(String(id), role === 'main' ? 'main' : 'listener') : null));
ipcMain.handle('ct:set-proxy', (e, id, value) => (fromDash(e) ? setAccountProxy(String(id), String(value == null ? '' : value)) : null));
ipcMain.handle('ct:channel-flags', (e) => (fromDash(e) ? channelFlags : null));
ipcMain.handle('ct:set-channel-flag', (e, id, key, on) => (fromDash(e) ? setChannelFlag(String(id), String(key), !!on) : null));

function setPaused(on) { paused = on; broadcast(); buildMenu(); }

// ----------------------------------------------------------------- windows
const SPLASH = ({ text, spin, error, offerStart }) => `<!doctype html><html><head><meta charset="utf-8"><meta name="color-scheme" content="light dark"><style>
html,body{margin:0;height:100%;overflow:hidden}
body{display:grid;place-items:center;font:15px -apple-system,"Segoe UI",system-ui,sans-serif;background:${BG.light};color:#52514e}
@media (prefers-color-scheme:dark){body{background:${BG.dark};color:#c3c2b7}}
.s{width:28px;height:28px;border-radius:50%;border:3px solid #3987e544;border-top-color:#3987e5;animation:r 1s linear infinite;margin:0 auto 16px}
@keyframes r{to{transform:rotate(360deg)}} .e{color:#e66767}
button{margin-top:16px;font:inherit;font-size:14px;padding:8px 18px;border-radius:10px;border:0;background:#3987e5;color:#fff;cursor:pointer}
</style></head><body><div style="text-align:center">${spin ? '<div class="s"></div>' : ''}<div class="${error ? 'e' : ''}">${text}</div>
${offerStart ? '<button onclick="ctApp.server(\'start\')">Start server</button>' : ''}</div></body></html>`;

function splash(text, { error = false, offerStart = false } = {}) {
  if (!dash || dash.isDestroyed()) return;
  const html = SPLASH({ text, error, offerStart, spin: !error && !offerStart });
  dash.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html)).catch(() => {});
}
function showDashboard() { if (dash && !dash.isDestroyed()) dash.loadURL(BASE).catch(() => {}); }

function windowBg() { return require('electron').nativeTheme.shouldUseDarkColors ? BG.dark : BG.light; }
const icon = () => (IS_WIN && fs.existsSync(ICON_ICO) ? ICON_ICO : fs.existsSync(ICON_PNG) ? ICON_PNG : undefined);

function createDashboard() {
  dash = new BrowserWindow({
    width: 1320, height: 860, minWidth: 760, minHeight: 540, title: 'Earshot', show: false,   // the dashboard adapts down to this
    backgroundColor: windowBg(), icon: icon(), autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'dashboard-preload.js'), contextIsolation: true, sandbox: true, backgroundThrottling: true },
  });
  dash.once('ready-to-show', () => dash.show());
  dash.webContents.on('did-finish-load', () => dash.webContents.send('ct:app-state', appState()));
  dash.on('close', (e) => { if (!quitting) { e.preventDefault(); dash.hide(); } });   // keeps transcribing
  dash.webContents.on('will-navigate', (e, url) => { if (!isDashUrl(url)) { e.preventDefault(); openExternal(url); } });
  dash.webContents.setWindowOpenHandler(({ url }) => { openExternal(url); return { action: 'deny' }; });
  dash.webContents.on('did-fail-load', (_e, code, _d, url, isMain) => {
    if (!isMain || code === -3 || !url.startsWith(BASE)) return;      // -3: aborted by our own navigation
    if (!server) splash(stoppedByUser ? 'The server is stopped. Nothing is being transcribed.' : 'Server not running.', { offerStart: true, error: !stoppedByUser });
    else setTimeout(async () => { if (await serverUp()) showDashboard(); }, 1000);
  });
  splash('Starting…');
}

function chromeUserAgent() {
  // Look like plain Chrome (Discord shows an "unsupported browser" notice otherwise).
  return app.userAgentFallback.split(' ').filter((t) => !/^(Electron|CallTranscriber|call-transcriber)\//i.test(t)).join(' ');
}

// ----------------------------------------------------------------- proxy
// Each account's Discord window has its own session, so a proxy set here
// applies to that account only (sign-in, pages, API, gateway). Chromium's
// proxy rules cannot carry credentials, so app.on('login') answers the
// proxy's challenge with the credentials from the same account.
const proxyApplied = new Set();    // partitions we changed (so clearing one goes back to the system)
async function applyProxy(a) {
  const p = a.proxyEnc ? { ok: false, error: 'The saved proxy could not be unlocked.' } : parseProxy(a.proxy);
  if (!p.ok) return p;
  // Accounts that never had a proxy keep Electron's own default (the system
  // network, or a command-line --proxy-server), so we don't override it.
  if (!p.proxy && !p.direct && !p.pac && !proxyApplied.has(a.partition)) return { ok: true, direct: false };
  const ses = session.fromPartition(a.partition);
  try {
    await ses.setProxy(p.config);
    proxyApplied.add(a.partition);
    try { await ses.closeAllConnections(); } catch {}   // don't reuse sockets from the old proxy
    return { ok: true, proxy: p.proxy || null, direct: !!p.direct };
  } catch {
    return { ok: false, error: 'Chromium could not use that proxy.' };
  }
}

async function setAccountProxy(id, value) {
  const a = acct(id); if (!a) return { error: 'That account no longer exists.' };
  const raw = String(value == null ? '' : value).trim();
  const p = parseProxy(raw);
  if (!p.ok) return { error: p.error };
  const before = a.proxy, beforeEnc = a.proxyEnc;
  if (raw) { a.proxy = raw; delete a.proxyEnc; } else { delete a.proxy; delete a.proxyEnc; }
  saveAccounts();
  const r = await applyProxy(a);
  if (!r.ok) {                        // keep the old, working setting
    if (before) a.proxy = before; else delete a.proxy;
    if (beforeEnc) a.proxyEnc = beforeEnc; else delete a.proxyEnc;
    saveAccounts();
    await applyProxy(a);
    return { error: r.error };
  }
  const w = winOf(id);
  if (w) w.webContents.reload();      // every new connection uses the new proxy
  broadcast(); buildMenu();
  return { ok: true, account: accountSummary(a) };
}

// The credentials Chromium asks for when a proxy wants a username and
// password. Matched to the account by its window, or failing that by the
// proxy host and port (some requests start outside the page).
function proxyDetailsFor(wc, authInfo) {
  if (!authInfo || !authInfo.isProxy) return null;
  const a = accountOfWebContents(wc) || accounts.find((x) => proxyLogin(x.proxy, authInfo.host, authInfo.port));
  return a ? proxyLogin(a.proxy, authInfo.host, authInfo.port) : null;
}

function setupDiscordSession(a) {
  const ses = session.fromPartition(a.partition);
  if (ses.__ctReady) return ses;
  ses.__ctReady = true;
  ses.setUserAgent(chromeUserAgent());
  const listener = () => acct(a.id)?.role === 'listener';
  const allowed = new Set(['media', 'notifications', 'clipboard-sanitized-write', 'clipboard-read', 'fullscreen', 'speaker-selection', 'display-capture']);
  // No account gets the real microphone; listener accounts get no camera either.
  const input = (types) => types.some((t) => t === 'audio' || t === 'video');
  const ok = (perm, url, types = []) => allowed.has(perm) && isDiscordUrl(url || '')
    && !(listener() && (perm === 'display-capture' || (perm === 'media' && input(types))))
    && !(perm === 'media' && types.includes('audio'));                   // never the microphone
  const log = TEST && process.env.CT_PERMLOG ? (...x) => console.log('PERM', a.role, ...x) : () => {};
  ses.setPermissionRequestHandler((_wc, perm, cb, details) => { const r = ok(perm, details.requestingUrl, details.mediaTypes || []); log('request', perm, JSON.stringify(details.mediaTypes || []), r); cb(r); });
  // Checks only report state (and Chromium also checks before playing audio on a machine
  // without speakers), so they stay allowed; actual camera/microphone requests are refused.
  ses.setPermissionCheckHandler((_wc, perm, origin, details) => { const r = allowed.has(perm) && isDiscordUrl(origin || ''); log('check', perm, details?.mediaType, r); return r; });
  // Screen share ("Share Your Screen" / Go Live): our own picker, which also
  // offers the transcriber itself — a clean compact view of this account's call.
  ses.setDisplayMediaRequestHandler(async (_req, cb) => {
    if (listener()) { cb({}); return; }
    try { cb((await pickShareSource(a)) || {}); } catch { cb({}); }
  });
  return ses;
}

// ----------------------------------------------------------------- screen share
const shareWins = new Map();     // channel id ('' = whatever is live) -> window
let picker = null, pickerSources = new Map(), pickerDone = null;

function openShareView(ch = '') {
  return new Promise((resolve) => {
    const have = shareWins.get(ch);
    if (have && !have.isDestroyed()) { have.showInactive(); resolve(have); return; }
    const w = new BrowserWindow({
      width: 1280, height: 720, minWidth: 800, minHeight: 450, title: 'Earshot shared view',
      backgroundColor: windowBg(), icon: icon(), autoHideMenuBar: true, show: false,
      webPreferences: { contextIsolation: true, sandbox: true, backgroundThrottling: false },
    });
    shareWins.set(ch, w);
    w.webContents.on('will-navigate', (e) => e.preventDefault());
    w.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
    w.on('closed', () => { if (shareWins.get(ch) === w) shareWins.delete(ch); });
    w.webContents.once('did-finish-load', () => { w.showInactive(); setTimeout(() => resolve(w), 400); });
    w.loadURL(BASE + '?view=share' + (ch ? '&ch=' + encodeURIComponent(ch) : '')).catch(() => resolve(w));
  });
}

async function sourceFor(id, ch = '') {
  if (id === 'app:share') { const w = await openShareView(ch); return { video: w.webContents.mainFrame }; }
  if (id === 'app:dash') { if (!dash || dash.isDestroyed()) return null; return { video: dash.webContents.mainFrame }; }
  const src = pickerSources.get(id);
  if (!src) return null;
  return { video: src, ...(IS_WIN && id.startsWith('screen:') ? { audio: 'loopback' } : {}) };
}

function pickShareSource(a) {
  const ch = rt.get(a?.id)?.status?.channel?.id || '';
  if (TEST && process.env.CT_SHARE_PICK) return sourceFor(process.env.CT_SHARE_PICK, ch);   // harness: no dialog
  if (picker && !picker.isDestroyed()) { picker.focus(); return Promise.resolve(null); }
  return new Promise((resolve) => {
    picker = new BrowserWindow({
      width: 760, height: 560, minWidth: 520, minHeight: 380, title: 'Share in Discord', parent: (a && winOf(a.id)) || undefined, modal: IS_MAC,
      backgroundColor: windowBg(), autoHideMenuBar: true, show: false,
      webPreferences: { preload: path.join(__dirname, 'picker-preload.js'), contextIsolation: true, sandbox: true },
    });
    picker.webContents.on('will-navigate', (e) => e.preventDefault());
    picker.once('ready-to-show', () => picker.show());
    pickerDone = async (id) => {
      pickerDone = null;
      const w = picker; picker = null;
      if (w && !w.isDestroyed()) w.destroy();
      resolve(id ? await sourceFor(id, ch).catch(() => null) : null);
    };
    picker.on('closed', () => { if (pickerDone) pickerDone(null); });
    picker.loadFile(path.join(__dirname, 'picker.html'));
  });
}

const fromPicker = (e) => picker && !picker.isDestroyed() && e.sender === picker.webContents;
ipcMain.handle('ct:picker-sources', async (e) => {
  if (!fromPicker(e)) return null;
  const app_ = [
    { id: 'app:share', name: 'Earshot compact view', note: 'A 1280×720 window for sharing, with the live transcript and stats', placeholder: 'Compact view' },
    { id: 'app:dash', name: 'Earshot dashboard', note: 'The dashboard as you see it', placeholder: 'Dashboard' },
  ];
  // live previews of our own views
  try {
    const img = dash && !dash.isDestroyed() ? await dash.webContents.capturePage() : null;
    if (img && !img.isEmpty()) app_[1].thumb = img.resize({ width: 320 }).toDataURL();
    const sw = [...shareWins.values()].find((w) => !w.isDestroyed());
    const simg = sw ? await sw.webContents.capturePage() : img;
    if (simg && !simg.isEmpty()) app_[0].thumb = simg.resize({ width: 320 }).toDataURL();
  } catch {}
  let screens = [], windows = [], note = '';
  try {
    const all = await desktopCapturer.getSources({ types: ['screen', 'window'], thumbnailSize: { width: 320, height: 180 }, fetchWindowIcons: false });
    pickerSources = new Map(all.map((x) => [x.id, x]));
    const own = new Set([picker, dash, ...shareWins.values(), ...accounts.map((x) => winOf(x.id))].filter((w) => w && !w.isDestroyed()).map((w) => w.getMediaSourceId()));
    const item = (x) => ({ id: x.id, name: x.name, thumb: x.thumbnail.isEmpty() ? null : x.thumbnail.toDataURL() });
    screens = all.filter((x) => x.id.startsWith('screen:')).map((x, i) => ({ ...item(x), name: all.filter((y) => y.id.startsWith('screen:')).length > 1 ? `Screen ${i + 1}` : 'Entire screen' }));
    windows = all.filter((x) => x.id.startsWith('window:') && !own.has(x.id) && x.name).map(item);
  } catch {}
  if (IS_MAC && systemPreferences.getMediaAccessStatus('screen') !== 'granted') {
    note = 'To share a screen or another app, allow Earshot in System Settings › Privacy & Security › Screen & System Audio Recording, then reopen it. The two Earshot views work without that.';
  }
  return { app: app_, screens, windows, note };
});
ipcMain.on('ct:picker-choose', (e, id) => { if (fromPicker(e) && pickerDone) pickerDone(id); });

function discordUrl(a, path_ = '/app') {
  if (TEST && process.env.CT_DISCORD_URL) {      // harness: fake Discord, account and channel in the query
    const u = new URL(process.env.CT_DISCORD_URL);
    if (a.testQuery) for (const [k, v] of new URLSearchParams(a.testQuery)) u.searchParams.set(k, v);
    const m = /^\/channels\/(\d+)\/(\d+)/.exec(path_); if (m) u.searchParams.set('ch', m[2]);
    return u.href;
  }
  return 'https://discord.com' + path_;
}

function createDiscord(a, { show = true, url } = {}) {
  const r = R(a.id);
  if (winOf(a.id)) return winOf(a.id);
  const w = new BrowserWindow({
    width: 1240, height: 820, minWidth: 940, minHeight: 500, show: false,
    title: `Discord · ${nameOf(a)}${a.role === 'listener' ? ' (listener)' : ''}`,
    backgroundColor: '#313338', icon: icon(), autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'discord-preload.js'), contextIsolation: true, sandbox: true, nodeIntegration: false,
      session: setupDiscordSession(a),
      backgroundThrottling: false,          // keep capturing at full rate behind other windows
      spellcheck: a.role === 'main',
    },
  });
  r.win = w; r.capturing = true;
  const wc = w.webContents;
  wc.setAudioMuted(!!a.muted);
  wc.on('will-navigate', (e, u) => { if (!isDiscordUrl(u)) { e.preventDefault(); openExternal(u); } });
  wc.on('will-redirect', (e, u) => { if (!isDiscordUrl(u)) { e.preventDefault(); openExternal(u); } });
  wc.setWindowOpenHandler(({ url: u }) => { openExternal(u); return { action: 'deny' }; });
  // Auto-join: note every reload, so a voice drop it caused reads as ours and not as
  // a kick; and bring a crashed page back (auto-join then rejoins its channel).
  wc.on('did-start-navigation', (d) => { if (d?.isMainFrame !== false && !d?.isSameDocument) R(a.id).navAt = Date.now(); });
  wc.on('render-process-gone', (_e, det) => {
    R(a.id).navAt = Date.now();
    console.log(`page for ${nameOf(a)} went away (${det?.reason}); reloading it`);
    if (det?.reason !== 'clean-exit' && !quitting) setTimeout(() => {
      if (w.isDestroyed()) return;
      R(a.id).navAt = Date.now();
      w.loadURL(wc.getURL() || discordUrl(a)).catch(() => {});
    }, 3000);
  });
  w.on('page-title-updated', (e) => e.preventDefault());      // keep "Discord · <account>"
  if (a.role === 'listener') wc.on('did-finish-load', () => wc.insertCSS('*,*::before,*::after{animation-duration:0s!important;transition-duration:0s!important}').catch(() => {}));
  w.on('close', (e) => { if (!quitting && acct(a.id)) { e.preventDefault(); w.hide(); broadcast(); } });   // stay in the call
  w.on('closed', () => { const rr = rt.get(a.id); if (rr && rr.win === w) { rr.win = null; try { rr.ws?.close(); } catch {} } });
  w.on('show', broadcast); w.on('hide', broadcast);
  w.once('ready-to-show', () => { if (show && (!TEST || process.env.CT_SHOW === '1')) w.show(); });
  w.loadURL(url || discordUrl(a)).catch(() => {});
  if (server || TEST) connect(a);
  return w;
}

async function showDiscord(id) {
  const a = (id && acct(id)) || mainAccount();
  const w = winOf(a.id);
  if (!w) { createDiscord(a); return; }
  w.show(); w.focus();
}

// ---- adding, removing and assigning accounts
function addAccount() {
  const id = 'acct-' + Date.now().toString(36);
  const a = { id, role: accounts.some((x) => x.role === 'main') ? 'listener' : 'main', partition: 'persist:discord-' + id, muted: true };
  accounts.push(a); saveAccounts(); buildMenu();
  createDiscord(a, { show: true, url: discordUrl(a, '/login') });
  broadcast();
  return accountSummary(a);
}

async function removeAccount(id) {
  const a = acct(id); if (!a || a.role === 'main') return { error: 'The main account can\'t be removed.' };
  const { response } = await dialog.showMessageBox(dash, { type: 'warning', buttons: ['Remove', 'Cancel'], defaultId: 1, cancelId: 1,
    message: `Remove ${nameOf(a)}?`, detail: 'Its Discord window closes and it is signed out on this computer. Your transcripts are kept.' });
  if (response !== 0) return { cancelled: true };
  accounts = accounts.filter((x) => x.id !== id); saveAccounts();
  const w = winOf(id); const r = rt.get(id);
  try { r?.ws?.close(); } catch {}
  if (w) w.destroy();
  rt.delete(id);
  await session.fromPartition(a.partition).clearStorageData().catch(() => {});
  buildMenu(); broadcast();
  return { ok: true };
}

function setRole(id, role) {
  const a = acct(id); if (!a || a.role === role) return accountSummary(a || {});
  if (role === 'main') { const m = mainAccount(); if (m && m !== a) { m.role = 'listener'; reloadAccount(m); } }
  a.role = role; if (role === 'listener') a.muted = true;
  saveAccounts(); reloadAccount(a); buildMenu(); broadcast();
  return accountSummary(a);
}
function reloadAccount(a) {       // a role change takes effect in a fresh page (mic access is set up before Discord loads)
  const w = winOf(a.id); if (!w) return;
  w.setTitle(`Discord · ${nameOf(a)}${a.role === 'listener' ? ' (listener)' : ''}`);
  w.webContents.setAudioMuted(!!a.muted);
  w.webContents.reload();
  hello(a);
}

// "Transcribe a channel": the next free account in that server opens the
// channel with Join Voice highlighted and you click Join — this manual flow
// never joins, leaves or mutes for you. (Auto-join, off by default, is the one
// place the app joins on its own; see autoJoinTick.)
function transcribeChannel(guildId, channelId) {
  const inServer = accounts.filter((a) => (a.directory || []).some((g) => g.id === guildId));
  if (!inServer.length) return { error: 'None of your accounts is in that server.' };
  const busy = (a) => { const s = accountSummary(a); return s.inCall; };
  const already = inServer.find((a) => { const s = accountSummary(a); return s.inCall && s.channel?.id === channelId; });
  if (already) { showDiscord(already.id); return { account: already.id, label: nameOf(already), already: true }; }
  const free = inServer.filter((a) => !busy(a)).sort((x, y) => (x.role === 'listener' ? 0 : 1) - (y.role === 'listener' ? 0 : 1));
  const a = free[0];
  if (!a) return { error: 'Every account in that server is already in a call. Add another account, or leave a call first.' };
  const r = R(a.id);
  const url = discordUrl(a, `/channels/${guildId}/${channelId}`);
  r.hint = { guildId, channelId, at: Date.now() };
  r.lastIn = null;                               // you picked a channel: don't rejoin the old one
  const w = winOf(a.id);
  if (w) w.loadURL(url).catch(() => {}); else createDiscord(a, { url });
  const ww = winOf(a.id);
  ww.webContents.once('did-finish-load', () => setTimeout(() => showJoinHint(a), 2500));
  ww.show(); ww.focus();
  broadcast();
  return { account: a.id, label: nameOf(a) };
}
const JOIN_CSS = `button[aria-label*="Join Voice" i], button[aria-label*="Join Call" i], [class*="joinButton"], #join {
  outline: 3px solid #f0b232 !important; outline-offset: 3px !important; box-shadow: 0 0 0 7px rgba(240,178,50,.35) !important; }`;
async function showJoinHint(a) {
  const w = winOf(a.id); if (!w) return;
  try { R(a.id).hintCss = await w.webContents.insertCSS(JOIN_CSS); } catch {}
  w.setTitle(`${nameOf(a)}: click Join Voice to start transcribing`);
}
function clearJoinHint(a) {
  const w = winOf(a.id), r = R(a.id); if (!w) return;
  if (r.hintCss) w.webContents.removeInsertedCSS(r.hintCss).catch(() => {});
  r.hintCss = null;
  w.setTitle(`Discord · ${nameOf(a)}${a.role === 'listener' ? ' (listener)' : ''}`);
}

// ----------------------------------------------------------------- auto-join
// Tell an account's page which channels to report occupancy for (the flagged
// ones in its own servers), so the controller can see which are open and full.
function flaggedChannelIds(a) {
  const ids = [];
  for (const g of a.directory || []) for (const c of g.channels || []) if (channelFlags[c.id]?.auto) ids.push(c.id);
  return ids;
}
function sendWatchList(a, connected = null) {      // plus the channel it sits in, to tell when it is left alone
  const ids = flaggedChannelIds(a);
  winOf(a.id)?.webContents.send('ct:watch-channels', connected ? [connected, ...ids.filter((id) => id !== connected)] : ids);
}
function occOf(status) {
  const m = {};
  for (const o of status?.occ || []) if (o && o.id) m[o.id] = { count: o.count | 0, limit: o.limit | 0 };
  return m;
}
// Click the voice-channel's Join button in the page (the same control
// "Transcribe a channel" highlights). true = a user gesture, so the browser
// lets the audio context start; without it the capture stays suspended and the
// channel would look silent for ever.
const CLICK_JOIN_JS = `(() => { const el = document.querySelector('button[aria-label*="Join Voice" i], button[aria-label*="Join Call" i], [class*="joinButton"], #join'); if (el) { el.click(); return true; } return false; })()`;
function autoJoinChannel(a, cand) {
  const r = R(a.id), w = winOf(a.id); if (!w) return;
  r.autoTarget = cand.channelId; r.autoTargetAt = Date.now(); r.autoJoined = null;
  const url = discordUrl(a, `/channels/${cand.guildId}/${cand.channelId}`);
  w.webContents.once('did-finish-load', () => setTimeout(() => {
    sendWatchList(a);
    winOf(a.id)?.webContents.executeJavaScript(CLICK_JOIN_JS, true).catch(() => {});
  }, 2500));
  w.loadURL(url).catch(() => {});
  broadcast();
}
// Runs on a timer. For each free account, join the best open flagged channel;
// for a channel that has gone quiet, move on to the next one.
// An account left alone in any channel (nobody there but your own accounts) for
// AUTOJOIN_ALONE seconds, or in a channel silent for AUTOJOIN_IDLE, moves to the
// best open flagged channel none of your other accounts is in. One that drops out
// because its page reloaded or crashed goes back; one taken out of voice any other
// way (a kick, a moderator's disconnect, your own Leave) stops auto-joining in that
// server until Auto is turned off and on again.
function autoJoinTick() {
  if (!appSettings().autoJoin || paused) return;
  const now = Date.now();
  const connectedNow = new Map();                 // channel id -> which account holds it
  const ownCount = new Map();                     // channel id -> how many of your accounts are in it
  for (const a of accounts) {
    const r = rt.get(a.id);
    if (!r?.status || now - (r.statusAt || 0) > 15000) continue;
    const ch = connectedChannelId(r.status);
    if (ch) { connectedNow.set(ch, a.id); ownCount.set(ch, (ownCount.get(ch) || 0) + 1); }
  }
  for (const a of accounts) {
    const r = rt.get(a.id), w = winOf(a.id);
    if (!w || !r || !r.wsOpen || r.capturing === false || r.hint) continue;
    if (now - (r.statusAt || 0) > 15000 || !r.status) continue;
    const status = r.status;
    const connected = connectedChannelId(status);
    sendWatchList(a, connected || r.lastIn?.channelId || null);                  // keep the page's watch list current across reloads
    r.cooldowns = r.cooldowns || {};
    for (const id of Object.keys(r.cooldowns)) if (r.cooldowns[id] <= now) delete r.cooldowns[id];
    // Resolve a join we started: confirm it, or give up and cool the channel down.
    if (r.autoTarget) {
      if (connected === r.autoTarget) { r.autoJoined = r.autoTarget; r.autoTarget = null; r.autoJoinAt = now; }
      else if (now - (r.autoTargetAt || 0) > 20000) { r.cooldowns[r.autoTarget] = now + AUTOJOIN_COOLDOWN; r.autoTarget = null; }
      else continue;                              // still connecting; give it a moment
    }
    if (!connected) r.autoJoined = null;
    r.autoStop = r.autoStop || {};
    // Left voice without the app moving it: a reload or crash goes back; a kick,
    // a moderator's disconnect or your own Leave stops auto-join in that server.
    if (!connected && r.lastIn) {
      const why = classifyLeave({ lastIn: r.lastIn, navAt: r.navAt || 0 }), back = r.lastIn;
      r.lastIn = null;
      if (why === 'removed') {
        const g = guildOf(a.directory || [], back.channelId);
        const gid = back.guildId || g?.guildId;
        if (gid) r.autoStop[gid] = { guildId: gid, guildName: g?.guildName || '', channelName: g?.name || '', at: now };
        console.log(`auto-join: ${nameOf(a)} left voice in ${g?.guildName || gid} without the app moving it; stopped there`);
        broadcast();
      } else {
        const o = occOf(status)[back.channelId];
        const held = connectedNow.has(back.channelId) && connectedNow.get(back.channelId) !== a.id;
        const open = !o || (o.count >= 1 && !(o.limit > 0 && o.count >= o.limit));
        if (!held && open && !((r.cooldowns[back.channelId] || 0) > now)) {
          const g = guildOf(a.directory || [], back.channelId);
          console.log(`auto-join: ${nameOf(a)} dropped out of ${g?.name || back.channelId} when its page reloaded; going back`);
          autoJoinChannel(a, { guildId: back.guildId || g?.guildId, channelId: back.channelId, name: g?.name || '' });
          connectedNow.set(back.channelId, a.id);
          continue;
        }
      }
    }
    if (connected !== r.inCh) { r.inCh = connected; r.inSince = now; }   // when it got to this channel, however it got there
    if (connected) {
      const g = guildOf(a.directory || [], connected);
      // at = when the page last said so (not now): a status sent before a reload must
      // not make the reload look older than the last sighting.
      r.lastIn = { channelId: connected, guildId: g?.guildId || r.lastIn?.guildId || null, at: r.statusAt || now };
    }
    const occ = occOf(status);
    // How long this account has been alone where it is (restarts when it changes channel).
    const alone = isAlone({ occ, channelId: connected, ownHere: Math.max(0, (ownCount.get(connected) || 0) - 1) });
    if (!alone || r.aloneIn !== connected) { r.aloneIn = alone ? connected : null; r.aloneSince = alone ? now : 0; }
    const ownChannels = new Set([...connectedNow.keys()].filter((ch) => connectedNow.get(ch) !== a.id));
    const cands = candidates({ directory: (a.directory || []).filter((g) => !r.autoStop[g.id]), flags: channelFlags, occ,
      ownChannels, cooldowns: r.cooldowns, currentChannelId: connected, now });
    if (connected) {
      if (alone && cands.length && now - r.aloneSince >= AUTOJOIN_ALONE * 1000) {
        const to = busiestFirst(cands)[0];
        console.log(`auto-join: ${nameOf(a)} was alone in ${connected}, moving to ${to.name || to.channelId}`);
        r.cooldowns[connected] = now + AUTOJOIN_COOLDOWN;
        r.aloneIn = null; r.aloneSince = 0;
        autoJoinChannel(a, to);
        if (connectedNow.get(connected) === a.id) connectedNow.delete(connected);
        connectedNow.set(to.channelId, a.id);
      } else if (cands.length
          && isIdle({ status, joinedAt: Math.max(r.autoJoinAt || 0, r.inSince || 0), now, idleSec: AUTOJOIN_IDLE })) {
        r.cooldowns[connected] = now + AUTOJOIN_COOLDOWN;   // don't bounce straight back to the quiet one
        r.autoJoined = null;
        autoJoinChannel(a, cands[0]);
        connectedNow.delete(connected); connectedNow.set(cands[0].channelId, a.id);
      }
    } else if (cands.length) {
      autoJoinChannel(a, cands[0]);
      connectedNow.set(cands[0].channelId, a.id);
    }
  }
}

function openExternal(url) {
  try { const u = new URL(url); if (u.protocol === 'https:' || u.protocol === 'http:') shell.openExternal(u.href); } catch {}
}

function showMain() { if (!dash || dash.isDestroyed()) createDashboard(); dash.show(); dash.focus(); }

// ----------------------------------------------------------------- update relaunch
async function relaunchAfterUpdate({ rebuild = false, deps = false, app: appDeps = false }) {
  splash('Installing the update…');
  await stopServer({ quiet: true });
  if (deps) {
    const uv = IS_WIN ? 'uv' : '/bin/zsh';
    const args = IS_WIN ? ['sync', '--inexact'] : ['-lc', 'uv sync --inexact'];
    await run(uv, args, { cwd: SERVER_DIR, shell: IS_WIN });
  }
  const desktopDir = __dirname;
  if (IS_MAC && (rebuild || appDeps)) {
    // The app bundle embeds Electron: rebuild it, then open the new copy.
    const code = await run('/bin/zsh', ['-lc', `"${path.join(ROOT, 'app', 'mac', 'build.sh')}"`]);
    if (code !== 0) { splash('The app rebuild failed. Run app/mac/build.sh in a terminal to see why.', { error: true, offerStart: true }); return; }
    const bundle = path.join(process.env.CT_INSTALL_DIR || path.join(app.getPath('home'), 'Applications'), 'Earshot.app');
    const envArgs = TEST ? Object.entries(process.env).filter(([k]) => k.startsWith('CT_')).flatMap(([k, v]) => ['--env', `${k}=${v}`]) : [];
    spawn('/usr/bin/open', ['-n', ...envArgs, bundle], { detached: true, stdio: 'ignore' }).unref();
  } else if (IS_WIN && appDeps) {
    // electron.exe can't be replaced while it runs: refresh it after we exit, then start again.
    const exe = path.join(desktopDir, 'node_modules', 'electron', 'dist', 'electron.exe');
    const script = path.join(app.getPath('temp'), 'ct-update.cmd');
    fs.writeFileSync(script, `@echo off\r\nping -n 3 127.0.0.1 >nul\r\ncd /d "${desktopDir}"\r\ncall npm ci --no-audit --no-fund\r\nnode node_modules\\electron\\install.js\r\nstart "" "${exe}" "${desktopDir}"\r\n`);
    spawn('cmd.exe', ['/c', script], { detached: true, stdio: 'ignore', windowsHide: true }).unref();
  } else {
    app.relaunch();
  }
  quitting = true;
  app.exit(0);
}

// ----------------------------------------------------------------- menu / tray
function buildMenu() {
  const t = [];
  if (IS_MAC) t.push({ label: 'Earshot', submenu: [
    { role: 'about' }, { type: 'separator' },
    { label: 'Settings…', accelerator: 'Cmd+,', click: () => { showMain(); dash?.webContents.executeJavaScript('window.openSettings?.()').catch(() => {}); } },
    { type: 'separator' }, { role: 'hide' }, { role: 'hideOthers' }, { type: 'separator' },
    { label: 'Quit Earshot (stops transcribing)', accelerator: 'Cmd+Q', click: quit },
  ] });
  t.push({ label: 'Discord', submenu: [
    { label: 'Show Discord', accelerator: 'CmdOrCtrl+D', click: () => showDiscord() },
    ...accounts.filter((a) => a.role === 'listener').map((a) => ({ label: `Show ${nameOf(a)} (listener)`, click: () => showDiscord(a.id) })),
    { label: 'Show Dashboard', accelerator: 'CmdOrCtrl+1', click: showMain },
    { type: 'separator' },
    { label: 'Transcribe a Channel…', accelerator: 'CmdOrCtrl+Shift+T', click: () => { showMain(); dash?.webContents.executeJavaScript('window.openTranscribe?.()').catch(() => {}); } },
    { label: 'Add Account…', click: () => addAccount() },
    { label: 'Open Shared View (for screen sharing)', click: () => openShareView() },
    { type: 'separator' },
    { label: 'Mute Call Audio (keeps transcribing)', type: 'checkbox', checked: !!mainAccount()?.muted, accelerator: 'CmdOrCtrl+Shift+M', click: (i) => setMuted(i.checked) },
    { label: 'Pause Transcribing', type: 'checkbox', checked: paused, click: (i) => setPaused(i.checked) },
    { label: 'Reload Discord', click: () => winOf(mainAccount().id)?.webContents.reload() },
  ] });
  t.push({ label: 'Server', submenu: [
    { label: 'Start Server', click: startServer },
    { label: 'Restart Server', accelerator: 'CmdOrCtrl+Shift+R', click: restartServer },
    { label: 'Stop Server', click: () => stopServer() },
    { type: 'separator' },
    { label: 'Show Server Log', accelerator: 'CmdOrCtrl+L', click: () => shell.openPath(LOG) },
  ] });
  const js = (code) => () => dash && !dash.isDestroyed() && dash.webContents.executeJavaScript(code).catch(() => {});
  t.push({ label: 'View', submenu: [
    { label: 'Reload Dashboard', accelerator: 'CmdOrCtrl+R', click: () => (server ? showDashboard() : splash('The server is stopped. Nothing is being transcribed.', { offerStart: true })) },
    { label: 'Show/Hide Calls List', accelerator: 'CmdOrCtrl+\\', click: js("window.showView?.('calls'); document.getElementById('toggle-side')?.click()") },
    { label: 'Compact View (everything on one screen)', accelerator: 'CmdOrCtrl+Shift+K', click: js("window.showView?.('calls'); document.getElementById('toggle-compact')?.click()") },
    { label: 'Search All Calls…', accelerator: 'CmdOrCtrl+Shift+F', click: js("(()=>{window.showView?.('calls');const s=document.getElementById('shell');s?.classList.remove('collapsed');requestAnimationFrame(()=>document.getElementById('hq')?.focus())})()") },
    { label: 'Find in Transcript…', accelerator: 'CmdOrCtrl+F', click: js("(()=>{window.showView?.('calls');requestAnimationFrame(()=>document.getElementById('q')?.focus())})()") },
    { type: 'separator' },
    { label: 'Open Transcripts Folder', accelerator: 'CmdOrCtrl+O', click: () => shell.openPath(path.join(ROOT, 'transcripts')) },
    { label: 'Open Dashboard in Browser', accelerator: 'CmdOrCtrl+B', click: () => shell.openExternal(BASE) },
    { type: 'separator' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'resetZoom' }, { role: 'toggleDevTools' },
  ] });
  t.push({ role: 'editMenu' });
  t.push({ label: 'Window', submenu: [{ role: 'minimize' }, { label: 'Close Window (keeps transcribing)', accelerator: 'CmdOrCtrl+W', role: 'close' },
    ...(IS_WIN ? [{ type: 'separator' }, { label: 'Settings…', accelerator: 'Ctrl+,', click: () => { showMain(); dash?.webContents.executeJavaScript('window.openSettings?.()').catch(() => {}); } },
      { label: 'Quit (stops transcribing)', click: quit }] : [])] });
  Menu.setApplicationMenu(Menu.buildFromTemplate(t));
}

function updateTray() {
  if (!tray) return;
  const calls = accounts.filter((a) => accountSummary(a).inCall).length;
  const state = !server ? 'server stopped' : paused ? 'paused' : calls ? `${calls} call${calls > 1 ? 's' : ''}` : 'ready';
  tray.setToolTip(`Earshot: ${state}`);
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: 'Open Earshot', click: showMain },
    { label: 'Open Discord', click: () => showDiscord() },
    { type: 'separator' },
    { label: 'Mute call audio (keeps transcribing)', type: 'checkbox', checked: !!mainAccount()?.muted, click: (i) => setMuted(i.checked) },
    { label: 'Pause transcribing', type: 'checkbox', checked: paused, click: (i) => setPaused(i.checked) },
    { label: 'Restart server', click: restartServer },
    { label: server ? 'Stop server' : 'Start server', click: () => (server ? stopServer() : startServer()) },
    { label: 'Open transcripts folder', click: () => shell.openPath(path.join(ROOT, 'transcripts')) },
    { label: 'Settings…', click: () => { showMain(); dash?.webContents.executeJavaScript('window.openSettings?.()').catch(() => {}); } },
    { type: 'separator' },
    { label: 'Quit (stops transcribing)', click: quit },
  ]));
}

async function quit() {
  if (quitting) return;
  quitting = true;
  await stopServer({ quiet: true });
  app.quit();
}

// ----------------------------------------------------------------- lifecycle
app.on('second-instance', showMain);
app.on('activate', showMain);
app.on('window-all-closed', () => {});                    // closing windows keeps transcribing
app.on('before-quit', (e) => {
  if (quitting) return;
  e.preventDefault(); quit();
});
app.on('web-contents-created', (_e, wc) => {
  wc.on('will-attach-webview', (ev) => ev.preventDefault());
});
// A proxy that asks for a username and password gets the ones saved with the
// account whose window made the request. Anything else keeps Electron's
// default (cancel), as before.
app.on('login', (event, wc, _details, authInfo, callback) => {
  const cred = proxyDetailsFor(wc, authInfo);
  if (!cred) return;
  event.preventDefault();
  callback(cred.username, cred.password);
});

app.whenReady().then(async () => {
  accounts = loadAccounts(); saveAccounts();
  loadChannelFlags();
  for (const a of accounts) await applyProxy(a);    // before any Discord window loads
  buildMenu();
  powerSaveBlocker.start('prevent-app-suspension');        // keep transcribing at full speed in the background
  if (IS_WIN && !TEST) {      // shortcuts from before the rename: "Call Transcriber" -> "Earshot"
    for (const dir of [path.join(app.getPath('appData'), 'Microsoft', 'Windows', 'Start Menu', 'Programs'), app.getPath('desktop')]) {
      const old = path.join(dir, 'Call Transcriber.lnk');
      try {
        if (fs.existsSync(old)) {
          shell.writeShortcutLink(path.join(dir, 'Earshot.lnk'), { target: process.execPath, args: `"${__dirname}"`, cwd: ROOT, icon: ICON_ICO, iconIndex: 0, description: 'Live transcripts of your voice calls' });
          fs.unlinkSync(old);
        }
      } catch {}
    }
  }
  if (IS_WIN && fs.existsSync(ICON_PNG)) {
    tray = new Tray(nativeImage.createFromPath(ICON_PNG).resize({ width: 16, height: 16 }));
    tray.on('click', showMain);
    updateTray();
  }
  createDashboard();
  createDiscord(mainAccount(), { show: appSettings().showCallWindowOnStart });
  // Listener accounts load when you use them; the test harness loads every account.
  if (TEST) for (const a of accounts) if (a.role !== 'main') createDiscord(a);
  startServer().then(() => { for (const a of accounts) if (winOf(a.id)) connect(a); });
});
