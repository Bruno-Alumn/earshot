// Sync between two Earshot installs (say a Mac and a PC), straight from one to the
// other over your own network (Tailscale or home Wi-Fi). No cloud.
//
// Pairing: one computer shows a code (its address and a one-time secret); you type
// it on the other. The secret itself never crosses the network: both sides derive a
// key from it, use that to swap fresh X25519 keys, and the long-term key comes from
// that exchange (so someone who recorded the pairing and later sees the code still
// can't work it out). From then on every request and answer is encrypted and
// signed with it (AES-256-GCM).
//
// A sync round (every few minutes, or Sync now), started by either side:
//   1. both sides' manifests: a hash per transcript day and per kind of data,
//   2. for days that differ, the line keys each side has,
//   3. the lines each side is missing go across, both ways, plus any kind of data
//      that differs (people, names, Q&A, merges, archive, reads, tests, voice history,
//      deletions),
//   4. each server merges what it got (server/sync.py has the rules).
// One round moves data both ways, so it is enough that one computer can reach the
// other (a firewall may block one direction).
//
// Only data syncs. Accounts, Discord sign-ins, keys and settings stay on each computer.
// No Electron in here (main.js passes in what it needs), so test/sync-test.js can run
// two of these against two servers.
'use strict';
const http = require('http');
const crypto = require('crypto');
const zlib = require('zlib');
const os = require('os');
const fs = require('fs');
const path = require('path');

const PROTO = 1;
const PAIR_MS = 10 * 60 * 1000;                 // a pairing code works for 10 minutes
const SKEW_MS = 3 * 60 * 1000;                  // requests older (or newer) than this are refused
const ROUND_MS = 3 * 60 * 1000;                 // automatic sync every 3 minutes
const MAX_BODY = 256 * 1024 * 1024;            // from a paired computer
const PAIR_BODY = 16 * 1024;                    // anyone else: a pairing request at most
const CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789';  // no 0/O, 1/I/L
const PUSH_CHUNK = 15000;                       // lines per push request

const b64 = (b) => Buffer.from(b).toString('base64');
const unb64 = (s) => Buffer.from(String(s || ''), 'base64');

function seal(key, obj, aad) {
  const iv = crypto.randomBytes(12);
  const c = crypto.createCipheriv('aes-256-gcm', key, iv);
  c.setAAD(Buffer.from(aad));
  const body = Buffer.concat([c.update(zlib.gzipSync(Buffer.from(JSON.stringify(obj)))), c.final(), c.getAuthTag()]);
  return { n: b64(iv), c: b64(body) };
}
function open(key, env, aad) {
  const iv = unb64(env.n), all = unb64(env.c);
  if (iv.length !== 12 || all.length < 17) throw new Error('bad envelope');
  const d = crypto.createDecipheriv('aes-256-gcm', key, iv);
  d.setAAD(Buffer.from(aad));
  d.setAuthTag(all.subarray(all.length - 16));
  const plain = Buffer.concat([d.update(all.subarray(0, all.length - 16)), d.final()]);
  return JSON.parse(zlib.gunzipSync(plain, { maxOutputLength: MAX_BODY * 4 }).toString('utf8'));
}
const pairKey = (secret) => crypto.scryptSync(secret, 'earshot-pair-v1', 32, { N: 16384, r: 8, p: 1 });
// Fresh X25519 keys for each pairing; the long-term key comes from the shared secret they make.
function ephemeral() {
  const { publicKey, privateKey } = crypto.generateKeyPairSync('x25519');
  return { privateKey, pub: publicKey.export({ type: 'spki', format: 'der' }).toString('base64') };
}
function longTermKey(privateKey, peerPub, codeKey, idA, idB) {
  const shared = crypto.diffieHellman({ privateKey, publicKey: crypto.createPublicKey({ key: unb64(peerPub), format: 'der', type: 'spki' }) });
  const ids = [idA, idB].sort().join('|');
  return Buffer.from(crypto.hkdfSync('sha256', Buffer.concat([shared, codeKey]), Buffer.from('earshot-sync-v1'), Buffer.from(ids), 32));
}
const ID_RE = /^[a-f0-9]{16}$/;

function newSecret() {
  let s = '';
  while (s.length < 16) { const b = crypto.randomBytes(1)[0]; if (b < 248) s += CODE_ALPHABET[b % 31]; }
  return s;
}
const fmtSecret = (s) => s.match(/.{4}/g).join('-');
function parseCode(text) {
  const m = String(text || '').trim().match(/^(\S+?)\s+([A-Za-z0-9-]{16,24})$/);
  if (!m) return null;
  let host = m[1], port = null;
  const hp = host.match(/^\[?([^\]]+?)\]?(?::(\d{2,5}))?$/);
  if (hp) { host = hp[1]; port = hp[2] ? Number(hp[2]) : null; }
  const secret = m[2].toUpperCase().replace(/-/g, '');
  if (secret.length !== 16 || [...secret].some((ch) => !CODE_ALPHABET.includes(ch))) return null;
  return { host, port, secret };
}

// This computer's addresses: Tailscale first (it works from anywhere), then the home network.
// Virtual adapters (WSL, Hyper-V, Docker, VMs) come last: the other computer can't reach them.
function localAddrs() {
  const out = [];
  const virt = /vethernet|wsl|hyper-v|docker|virtualbox|vmware|vmnet|vbox|utun|bridge|loopback/i;
  for (const [name, list] of Object.entries(os.networkInterfaces())) {
    for (const a of list || []) {
      if (a.family !== 'IPv4' && a.family !== 4) continue;
      if (a.internal || a.address.startsWith('169.254.')) continue;
      out.push({ ip: a.address, name });
    }
  }
  const ts = (ip) => { const [x, y] = ip.split('.').map(Number); return x === 100 && y >= 64 && y <= 127; };
  const lan = (ip) => /^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(ip);
  const rank = (a) => (ts(a.ip) ? 0 : virt.test(a.name) ? 3 : lan(a.ip) ? 1 : 2);
  return [...new Map(out.sort((a, b) => rank(a) - rank(b)).map((a) => [a.ip, a])).keys()];
}

function readBody(req, limit) {
  return new Promise((resolve, reject) => {
    const chunks = []; let n = 0;
    req.on('data', (c) => { n += c.length; if (n > limit) { reject(new Error('too big')); req.destroy(); } else chunks.push(c); });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

function postJson(url, obj, timeoutMs, headers = {}) {
  return new Promise((resolve, reject) => {
    const body = Buffer.from(JSON.stringify(obj));
    const req = http.request(url, { method: 'POST', agent: false, headers: { 'Content-Type': 'application/json', 'Content-Length': body.length, Connection: 'close', ...headers }, timeout: timeoutMs },
      (res) => {
        const chunks = []; let n = 0;
        res.on('data', (c) => { n += c.length; if (n > MAX_BODY) req.destroy(new Error('answer too big')); else chunks.push(c); });
        res.on('end', () => { try { resolve({ status: res.statusCode, body: JSON.parse(Buffer.concat(chunks).toString('utf8') || '{}') }); } catch (e) { reject(e); } });
      });
    req.on('timeout', () => req.destroy(new Error('timed out')));
    req.on('error', reject);
    req.end(body);
  });
}

function createSync({ stateFile, serverBase, sealKey = (s) => ({ key: s }), unsealKey = (o) => o.key, port = 8767,
                      log = () => {}, onChange = () => {}, name = os.hostname(), hooks = {} } = {}) {
  let state = { installId: null, name, peers: [], last: {} };
  try { state = { ...state, ...JSON.parse(fs.readFileSync(stateFile, 'utf8')) }; }
  catch (e) { if (e.code !== 'ENOENT') { try { fs.copyFileSync(stateFile, stateFile + '.damaged'); } catch {} } }   // keep it for a look; start fresh
  if (!state.installId) state.installId = crypto.randomBytes(8).toString('hex');
  const keys = new Map();                              // peer id -> key Buffer
  for (const p of state.peers) { try { const k = unsealKey(p.keyStore); if (k) keys.set(p.id, unb64(k)); } catch {} }
  const save = () => {
    try {
      fs.mkdirSync(path.dirname(stateFile), { recursive: true });
      fs.writeFileSync(stateFile, JSON.stringify(state, null, 1), { mode: 0o600 });
    } catch {}
  };
  save();
  let pairing = null;                                  // { secret, key, until, fails: Map(ip -> n) }
  let server = null, listening = false, listenError = null;
  let busy = false, timer = null, startTimer = null;
  const seen = new Map();                              // request ids seen recently (replays are refused)

  const local = async (p, params = {}) => {
    const q = new URLSearchParams(params).toString();
    let r;
    try { r = await fetch(`${serverBase}/api/sync/${p}${q ? '?' + q : ''}`, { headers: { 'X-CT': '1' } }); }
    catch { const e = new Error('Earshot\'s transcription server isn\'t running there. Start it from the tray menu, then sync again.'); e.code = 'SERVER_DOWN'; throw e; }
    const j = await r.json();
    if (!r.ok || j?.error) throw new Error(j?.error || `server answered ${r.status}`);
    return j;
  };
  const localImport = async (bundle, peerId, mode) => {
    const m = await local('manifest');
    const name = crypto.randomBytes(8).toString('hex');
    fs.mkdirSync(m.inbox, { recursive: true });
    fs.writeFileSync(path.join(m.inbox, `${name}.json`), JSON.stringify(bundle));
    return local('import', { name, peer: peerId, mode });
  };

  // ---------------------------------------------------------------- serving the other computer
  async function serve(op, args, peerId) {
    if (op === 'hello') return { installId: state.installId, name: state.name, addrs: localAddrs(), port };
    if (op === 'manifest') {
      if (busy === peerId) throw new Error('busy');      // both started a round at once: theirs waits, before it does any work
      return local('manifest');
    }
    if (op === 'keys') return local('keys', { days: (args.days || []).join(',') });
    if (op === 'records') {
      const want = new Set(args.keys || []);
      const recs = await local('records', { day: String(args.day || '') });
      return recs.filter(([k]) => want.has(k));
    }
    if (op === 'parts') return local('parts', { names: (args.names || []).join(',') });
    if (op === 'push') {
      if (busy === peerId) throw new Error('busy');      // both started a round at once: theirs waits for the next one
      const r = await localImport(args.bundle || {}, peerId, 'push');
      noteGot(peerId, r);
      return r;
    }
    if (op === 'settle') return local('settle', { peer: peerId, names: (args.names || []).join(',') });
    throw new Error('unknown request');
  }
  function noteGot(peerId, r) {
    const l = state.last[peerId] = state.last[peerId] || {};
    l.gotAt = l.inAt = Date.now(); l.gotLines = (l.gotLines || 0) + (r?.added || 0); save(); onChange();
  }

  async function handle(req, res) {
    const reply = (status, obj) => { const b = Buffer.from(JSON.stringify(obj)); res.writeHead(status, { 'Content-Type': 'application/json', 'Content-Length': b.length }); res.end(b); };
    if (req.method !== 'POST' || req.url !== '/earshot-sync/v1') return reply(404, { error: 'not found' });
    const ip = req.socket.remoteAddress?.replace(/^::ffff:/, '') || '';
    const isPair = req.headers['x-earshot-pair'] === '1', from = String(req.headers['x-earshot-from'] || '');
    // Decide how much to read before reading anything: only a paired computer may send a lot.
    const limit = isPair ? (pairing && Date.now() < pairing.until ? PAIR_BODY : 0) : keys.has(from) ? MAX_BODY : 0;
    if (!limit) { req.resume(); return reply(403, { error: isPair ? 'no-pairing' : 'unpaired' }); }
    if (Number(req.headers['content-length'] || 0) > limit) { req.resume(); return reply(413, { error: 'too big' }); }
    let env;
    try { env = JSON.parse((await readBody(req, limit)).toString('utf8')); } catch { return reply(400, { error: 'bad request' }); }
    if (isPair) {                                       // a computer typing this one's code
      if (!pairing || Date.now() > pairing.until) return reply(403, { error: 'no-pairing' });
      const tries = pairing.fails.get(ip) || 0;
      if (tries >= 5) return reply(429, { error: 'too-many' });      // that address only: others can still pair
      let msg;
      try { msg = open(pairing.key, env, 'pair'); } catch { pairing.fails.set(ip, tries + 1); return reply(403, { error: 'bad-code' }); }
      if (Math.abs(Date.now() - (msg.ts || 0)) > SKEW_MS || !ID_RE.test(msg.installId || '') || msg.installId === state.installId || typeof msg.epk !== 'string') return reply(400, { error: 'bad request' });
      const eph = ephemeral();
      let key;
      try { key = longTermKey(eph.privateKey, msg.epk, pairing.key, state.installId, msg.installId); } catch { return reply(400, { error: 'bad request' }); }
      addPeer({ id: msg.installId, name: String(msg.name || 'Other computer').slice(0, 60), addrs: [ip, ...(Array.isArray(msg.addrs) ? msg.addrs : [])].filter((a) => typeof a === 'string' && a.length < 64), port: Number(msg.port) || 8767 }, key);
      const out = seal(pairing.key, { ts: Date.now(), installId: state.installId, name: state.name, addrs: localAddrs(), port, epk: eph.pub }, 'pair');
      pairing = null; onChange();
      log(`sync: paired with ${msg.name}`);
      return reply(200, out);
    }
    if (env.from !== from) return reply(400, { error: 'bad request' });
    const key = keys.get(env.from);
    if (!key) return reply(403, { error: 'unpaired' });
    let msg;
    try { msg = open(key, env, `req:${env.from}`); } catch { return reply(403, { error: 'bad-key' }); }
    if (Math.abs(Date.now() - (msg.ts || 0)) > SKEW_MS) return reply(400, { error: 'clock' });
    if (typeof msg.rid !== 'string' || msg.rid.length < 16 || seen.has(msg.rid)) return reply(400, { error: 'replay' });
    seen.set(msg.rid, Date.now());
    for (const [k, t] of seen) if (Date.now() - t > 2 * SKEW_MS) seen.delete(k);
    const peer = state.peers.find((p) => p.id === env.from);
    if (peer && ip && !peer.addrs.includes(ip)) { peer.addrs = [ip, ...peer.addrs].slice(0, 8); save(); }
    if (msg.op === 'manifest') { const l = state.last[env.from] = state.last[env.from] || {}; l.inAt = Date.now(); l.gotLines = 0; save(); onChange(); }   // it started a round
    try {
      const out = await serve(msg.op, msg.args || {}, env.from);
      return reply(200, seal(key, { ok: true, rid: msg.rid, out }, `res:${env.from}`));
    } catch (e) {
      return reply(200, seal(key, { ok: false, rid: msg.rid, error: String(e.message || e).slice(0, 200) }, `res:${env.from}`));
    }
  }

  function addPeer(p, key) {
    // the same computer paired again (say after a reinstall, with a new id): the old entry goes
    const stale = state.peers.filter((x) => x.id !== p.id && x.name === p.name && x.addrs.some((a) => p.addrs.includes(a)));
    for (const x of stale) { keys.delete(x.id); delete state.last[x.id]; }
    state.peers = state.peers.filter((x) => x.id !== p.id && !stale.includes(x));
    state.peers.push({ ...p, addrs: [...new Set(p.addrs)].slice(0, 8), pairedAt: Date.now(), keyStore: sealKey(b64(key)) });
    keys.set(p.id, key);
    save(); onChange();
  }

  let listening$ = null;
  function listen() {
    if (server && listening) return Promise.resolve(true);
    if (listening$) return listening$;                // already starting: wait for that one
    if (server) { try { server.close(); } catch {} server = null; }       // a failed listen: try again
    return (listening$ = new Promise((resolve) => {
      server = http.createServer((req, res) => { handle(req, res).catch(() => { try { res.writeHead(500); res.end(); } catch {} }); });
      server.maxConnections = 16;
      server.keepAliveTimeout = 1000;
      server.requestTimeout = 5 * 60 * 1000;
      server.headersTimeout = 20 * 1000;
      server.on('error', (e) => { listening = false; listenError = e.code === 'EADDRINUSE' ? `Port ${port} is in use by another program.` : String(e.message); log('sync: ' + listenError); onChange(); listening$ = null; resolve(false); });
      server.listen(port, '0.0.0.0', () => { listening = true; listenError = null; onChange(); listening$ = null; resolve(true); });
    }));
  }
  function closeIfIdle() {                             // nothing to answer: no paired computer, no code showing
    if (server && !state.peers.length && !(pairing && Date.now() < pairing.until)) { server.close(); server = null; listening = false; onChange(); }
  }

  // ---------------------------------------------------------------- talking to the other computer
  async function rpc(peer, addr, op, args = {}, timeoutMs = 60000) {
    const key = keys.get(peer.id);
    if (!key) throw new Error('This computer has no key for it. Pair again.');
    const rid = crypto.randomBytes(12).toString('hex');
    const env = { v: PROTO, from: state.installId, ...seal(key, { ts: Date.now(), rid, op, args }, `req:${state.installId}`) };
    const host = addr.includes(':') ? `[${addr}]` : addr;
    const r = await postJson(`http://${host}:${peer.port || 8767}/earshot-sync/v1`, env, timeoutMs, { 'X-Earshot-From': state.installId });
    if (r.status === 403 && r.body?.error === 'unpaired') throw new Error(`${peer.name} doesn't know this computer any more. Pair them again.`);
    if (r.status !== 200) throw new Error(`${peer.name} refused the request (${r.body?.error || r.status}).`);
    if (!r.body?.n) throw new Error(`${peer.name} sent back something unexpected.`);
    const msg = open(key, r.body, `res:${state.installId}`);
    if (msg.rid !== rid) throw new Error('mixed-up answer');
    if (!msg.ok) throw new Error(msg.error === 'busy' ? `${peer.name} was syncing with this computer at the same moment. It catches up at the next round.` : `${peer.name}: ${msg.error}`);
    return msg.out;
  }

  async function reach(peer) {
    const errs = [];
    for (const addr of peer.addrs) {
      try { await rpc(peer, addr, 'hello', {}, 6000); return addr; } catch (e) { errs.push(e.message); if (/Pair them again|no key/.test(e.message)) throw e; }
    }
    log(`sync: couldn't reach ${peer.name}: ${errs.join(' / ')}`);
    throw new Error(`Couldn't reach ${peer.name}. Is Earshot running there, on the same network or Tailscale?`);
  }

  async function round(peer) {
    const addr = await reach(peer);
    if (peer.addrs[0] !== addr) { peer.addrs = [addr, ...peer.addrs.filter((a) => a !== addr)]; save(); }
    const [mL, mR] = await Promise.all([local('manifest'), rpc(peer, addr, 'manifest')]);
    const days = [...new Set([...Object.keys(mL.days), ...Object.keys(mR.days)])].filter((d) => mL.days[d] !== mR.days[d]).sort();
    const pulled = [], pushed = [];
    for (let i = 0; i < days.length; i += 20) {
      const batch = days.slice(i, i + 20);
      const [kL, kR] = await Promise.all([local('keys', { days: batch.join(',') }), rpc(peer, addr, 'keys', { days: batch })]);
      for (const d of batch) {
        const l = new Set(kL[d] || []), r = new Set(kR[d] || []);
        const need = [...r].filter((k) => !l.has(k)), give = [...l].filter((k) => !r.has(k));
        if (need.length) pulled.push(...await rpc(peer, addr, 'records', { day: d, keys: need }, 120000));
        if (give.length) { const g = new Set(give); pushed.push(...(await local('records', { day: d })).filter(([k]) => g.has(k))); }
      }
    }
    const parts = Object.keys({ ...mL.parts, ...mR.parts }).filter((k) => mL.parts[k] !== mR.parts[k]);
    const pR = parts.length ? await rpc(peer, addr, 'parts', { names: parts }, 120000) : {};
    const split = (p) => { const { deleted, ...rest } = p || {}; return { deleted: deleted || [], parts: rest }; };
    const fromR = split(pR);
    // 1. take in theirs (merged against what both had last time; kept as pending until they have ours)
    let got = { added: 0 };
    if (pulled.length || fromR.deleted.length || Object.keys(fromR.parts).length) got = await localImport({ deleted: fromR.deleted, records: pulled, parts: fromR.parts }, peer.id, 'pull');
    if (hooks.afterPull) await hooks.afterPull();       // tests only: something happens mid-round
    // 2. send ours, as merged just now, so both end up with the same; the parts go with the last lines
    const fromL = split(parts.length ? await local('parts', { names: parts.join(',') }) : {});
    let sent = 0;
    if (pushed.length || fromL.deleted.length || Object.keys(fromL.parts).length) {
      for (let i = 0; i === 0 || i < pushed.length; i += PUSH_CHUNK) {
        const last = i + PUSH_CHUNK >= pushed.length;
        const r = await rpc(peer, addr, 'push', { bundle: { deleted: i === 0 ? fromL.deleted : [], records: pushed.slice(i, i + PUSH_CHUNK), parts: last ? fromL.parts : {} } }, 300000);
        sent += r?.added || 0;
      }
    }
    // 3. they have it: what both hold is the base for next time
    await local('commit', { peer: peer.id });
    const three = ['merges', 'archive', 'study'];
    const [aL, aR] = await Promise.all([local('manifest'), rpc(peer, addr, 'manifest')]);
    const same = three.filter((k) => aL.parts[k] === aR.parts[k]);
    if (same.length) await Promise.all([local('settle', { peer: peer.id, names: same.join(',') }), rpc(peer, addr, 'settle', { names: same })]);
    return { addr, got: got.added || 0, sent, dupes: got.dupes || 0, days: days.length, parts };
  }

  async function syncNow(only, auto = false) {
    if (busy) return { busy: true };
    busy = true; onChange();                           // true, then the id of the computer it is syncing with
    const results = {};
    try {
      for (const peer of state.peers.filter((p) => !only || p.id === only)) {
        const l = state.last[peer.id] = state.last[peer.id] || {};
        // it ran a round with this computer just now (moving data both ways): nothing to do. This also
        // keeps a computer whose firewall blocks one direction from showing an error every round.
        if (auto && l.inAt && Date.now() - l.inAt < ROUND_MS) continue;
        l.tryAt = Date.now();
        busy = peer.id;
        try {
          const r = await round(peer);
          Object.assign(l, { okAt: Date.now(), error: null, got: r.got, sent: r.sent, addr: r.addr });
          if (r.got || r.sent) log(`sync with ${peer.name}: got ${r.got} lines, sent ${r.sent}`);
          l.dupes = r.dupes;
          results[peer.id] = r;
        } catch (e) {
          l.error = e.code === 'SERVER_DOWN' ? 'Earshot\'s transcription server isn\'t running on this computer. Start it from the tray menu.' : String(e.message || e).slice(0, 240);
          results[peer.id] = { error: l.error };
        }
        save();
      }
    } finally { busy = false; onChange(); }
    return results;
  }

  function start() {
    if (state.peers.length) listen();             // nothing to answer until paired (pairing opens it too)
    clearTimeout(timer); clearTimeout(startTimer);
    startTimer = setTimeout(() => syncNow().catch(() => {}), 20000);
    const next = () => {                           // a little jitter, so two computers rarely start at the same moment
      timer = setTimeout(async () => { if (state.peers.length) await syncNow(null, true).catch(() => {}); next(); }, ROUND_MS + Math.floor(Math.random() * 30000));
    };
    next();
  }
  function stop() {
    clearTimeout(timer); clearTimeout(startTimer);
    if (server) { server.close(); server = null; listening = false; }
  }

  return {
    start, stop, syncNow,
    get installId() { return state.installId; },
    view() {
      return {
        name: state.name, installId: state.installId, addrs: localAddrs(), port, listening, listenError, busy: !!busy,
        pairing: pairing && Date.now() < pairing.until ? { code: `${localAddrs()[0] || '?'}${port !== 8767 ? ':' + port : ''} ${fmtSecret(pairing.secret)}`, until: pairing.until } : null,
        peers: state.peers.map((p) => ({ id: p.id, name: p.name, addrs: p.addrs, pairedAt: p.pairedAt, last: state.last[p.id] || {} })),
      };
    },
    async startPairing() {
      if (!(await listen())) return { error: `The other computer couldn't reach this one: ${listenError} Pair from this computer instead: show the code on the other one and type it here.` };
      const secret = newSecret();
      pairing = { secret, key: pairKey(secret), until: Date.now() + PAIR_MS, fails: new Map() };
      setTimeout(() => { closeIfIdle(); onChange(); }, PAIR_MS + 1000);
      onChange();
      return this.view().pairing;
    },
    cancelPairing() { pairing = null; closeIfIdle(); onChange(); },
    async join(code) {
      const c = parseCode(code);
      if (!c) return { error: 'That code doesn\'t look right. It is an address and 16 letters and numbers, like 100.64.1.2 ABCD-EFGH-JKLM-NPQR.' };
      const key = pairKey(c.secret);
      const p = c.port || 8767;
      const host = c.host.includes(':') ? `[${c.host}]` : c.host;
      const eph = ephemeral();
      let r;
      try { r = await postJson(`http://${host}:${p}/earshot-sync/v1`, seal(key, { ts: Date.now(), installId: state.installId, name: state.name, addrs: localAddrs(), port, epk: eph.pub }, 'pair'), 15000, { 'X-Earshot-Pair': '1' }); }
      catch { return { error: `Couldn't reach ${c.host}. Check that Earshot is open there and both computers are on Tailscale or the same network.` }; }
      if (r.status !== 200) return { error: r.body?.error === 'no-pairing' ? 'That code has run out or was cancelled. Show a new one on the other computer.' : r.body?.error === 'too-many' ? 'Too many wrong tries from this computer. Show a new code on the other computer.' : 'The other computer didn\'t accept that code. Check it and try again.' };
      let msg, longKey;
      try {
        msg = open(key, r.body, 'pair');
        if (!ID_RE.test(msg.installId || '') || msg.installId === state.installId) throw new Error('bad id');
        longKey = longTermKey(eph.privateKey, msg.epk, key, state.installId, msg.installId);
      } catch { return { error: 'The answer didn\'t match the code. Try again with a new code.' }; }
      addPeer({ id: msg.installId, name: String(msg.name || 'Other computer').slice(0, 60), addrs: [c.host, ...(Array.isArray(msg.addrs) ? msg.addrs : [])].filter((a) => typeof a === 'string' && a.length < 64), port: Number(msg.port) || p }, longKey);
      listen();
      setTimeout(() => syncNow(msg.installId).catch(() => {}), 1000);
      return { ok: true, name: msg.name };
    },
    unpair(id) {
      state.peers = state.peers.filter((p) => p.id !== id); delete state.last[id]; keys.delete(id);
      save(); closeIfIdle(); onChange(); return { ok: true };
    },
    rename(n) { state.name = String(n || '').trim().slice(0, 60) || os.hostname(); save(); onChange(); return { ok: true }; },
  };
}

module.exports = { createSync, parseCode, seal, open, pairKey, localAddrs };
