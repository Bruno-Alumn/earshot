// Runs in Discord's page (MAIN world) at document_start, before Discord opens
// its voice connection.
//
// Discord's web client receives every participant as a separate WebRTC audio
// track; the remote stream id is "<userId>-<ssrc>". We read a clone of each
// track (playback is untouched), downsample to 16 kHz mono, and post 200 ms
// chunks tagged with the user id + display name to content.js.
(() => {
  if (window.__dtHook) return;
  window.__dtHook = { version: 13, tracks: 0, chunks: 0, errors: [], loadedAt: document.readyState, pcs: 0, badStreamIds: [] };
  const H = window.__dtHook;
  // Discord's end-to-end voice encryption logs this when it can't decrypt a
  // frame. A burst of them while every track is silent means the call audio is
  // lost on this side (rejoining the channel fixes it): count them so the
  // dashboard can say so.
  H.decryptFails = 0; H.lastHeard = 0;
  try {
    const err = console.error;
    console.error = function (...a) {
      try { if (a.some((x) => typeof x === 'string' && x.includes('Failed to finalize decryption'))) H.decryptFails++; } catch {}
      return err.apply(this, a);
    };
  } catch {}
  const pcs = new Set();
  const ssrcUser = new Map();           // voice gateway op 5: ssrc -> user id
  const readers = new Map();            // track.id -> entry
  const idName = new Map();             // user id -> display name
  const lastActive = new Map();         // user id -> ms of last audible chunk
  let enabled = false;

  // ---------------------------------------------------------------- names
  const ROW = '[class*="voiceUser"]';
  const clean = (n) => (typeof n === 'string' ? n : '').replace(/, (Muted|Deafened|Server Muted|Server Deafened)$/i, '').trim().slice(0, 60);
  function rowName(row) {
    const el = row.querySelector('[class*="usernameFont"], [class*="username"], [class*="displayName"]');
    return clean(el?.textContent) ||
      clean(row.querySelector('[aria-label]')?.getAttribute('aria-label')) ||
      clean(row.innerText?.split('\n').map((s) => s.trim()).find(Boolean));
  }
  // Identify the user id behind a sidebar row.
  //  1. Avatar URL embeds the id (authoritative; seen in real Discord data).
  //  2. React props on the row or its ancestors. A shared ancestor would give
  //     the same id to many rows, so a React id is only trusted when exactly
  //     one row yields it and no avatar already belongs to someone else.
  function fiberOf(el) { const k = Object.keys(el).find((x) => x.startsWith('__reactFiber$')); return k ? el[k] : null; }
  function propId(p) {
    if (!p || typeof p !== 'object') return null;
    const c = [p.user?.id, p.userId, p.voiceState?.userId, p.participant?.user?.id, p.member?.userId, p.id && p.username && p.id];
    return c.find((x) => typeof x === 'string' && /^\d{15,22}$/.test(x)) || null;
  }
  function rowUserId(row) {
    const m = (row.querySelector('[style*="background-image"]')?.getAttribute('style') || '').match(/\/avatars\/(\d{15,22})\//);
    if (m) return [m[1], 'avatar'];
    for (let f = fiberOf(row), i = 0; f && i < 15; f = f.return, i++) {
      const id = propId(f.memoizedProps);
      if (id) return [id, 'react'];
    }
    return [null, null];
  }
  function fiberKeys(row) {      // diagnostics only: prop names up the tree, no values
    const out = [];
    for (let f = fiberOf(row), i = 0; f && i < 15; f = f.return, i++) {
      const p = f.memoizedProps;
      if (p && typeof p === 'object') out.push(Object.keys(p).filter((k) => k !== 'children').slice(0, 8).join(','));
    }
    return out;
  }

  // Discord's own in-memory stores. Discord's page runs several rspack
  // runtimes sharing webpackChunkdiscord_app; pushing a chunk calls back once
  // per runtime. Only the main one (thousands of cached modules) holds the
  // live Flux stores, so collect every runtime and use the largest cache.
  // Stores identify themselves via getName(). (Verified on live discord.com.)
  let stores = null, lastTry = 0;
  H.storeDiag = {};
  function findStores() {
    if (stores || Date.now() - lastTry < 15000) return stores;
    lastTry = Date.now();
    const D = H.storeDiag = {};
    try {
      const chunk = window.webpackChunkdiscord_app;
      if (!chunk?.push) { D.result = 'no webpack chunk'; return null; }
      const reqs = [];
      chunk.push([[Symbol('dt')], {}, (r) => { reqs.push(r); }]);
      try { chunk.pop(); } catch {}
      const size = (r) => { try { return Object.keys(r.c || {}).length; } catch { return 0; } };
      D.runtimes = reqs.map(size);
      const main = reqs.reduce((x, y) => (size(y) > size(x) ? y : x), reqs[0]);
      if (!main) { D.result = 'no runtime'; return null; }
      const want = { UserStore: 'user', GuildMemberStore: 'member', VoiceStateStore: 'voice',
        SelectedChannelStore: 'selected', ChannelStore: 'channels', GuildStore: 'guilds', GuildChannelStore: 'guildChannels' };
      const found = {};
      for (const mod of Object.values(main.c)) {
        const ex = mod?.exports;
        if (!ex || (typeof ex !== 'object' && typeof ex !== 'function')) continue;
        let vals; try { vals = [ex, ...Object.values(ex)]; } catch { continue; }
        for (const v of vals) {
          let n; try { n = v && typeof v.getName === 'function' ? v.getName() : null; } catch { continue; }
          if (typeof n === 'string' && want[n] && !found[want[n]]) found[want[n]] = v;
        }
        if (Object.keys(found).length === Object.keys(want).length) break;
      }
      D.found = Object.keys(found);
      if (found.user) { stores = found; D.result = 'ok'; } else D.result = 'UserStore not in main runtime';
      // Shapes only (never values): lets us check Discord's API from status.log.
      try {
        const g = Object.values(found.guilds?.getGuilds?.() || {})[0];
        const gc = g && found.guildChannels?.getChannels?.(g.id);
        const vch = gc?.VOCAL?.[0]?.channel || gc?.VOCAL?.[0];
        const vs = vch && found.voice?.getVoiceStatesForChannel?.(vch.id);
        const me = found.user.getCurrentUser?.();
        const mine = me && found.voice?.getVoiceStateForUser?.(me.id);
        D.probe = { guilds: Object.keys(found.guilds?.getGuilds?.() || {}).length, gcKeys: gc ? Object.keys(gc).join(',') : null,
          vocalItem: vch ? Object.keys(gc.VOCAL[0]).join(',') + '|type=' + vch.type : null,
          vsType: vs == null ? String(vs) : vs instanceof Map ? 'Map' : Array.isArray(vs) ? 'Array' : typeof vs,
          myState: mine ? Object.keys(mine).filter((k) => /deaf|mute|channel/i.test(k)).join(',') : 'none',
          hasMutableGuildChannels: typeof found.channels?.getMutableGuildChannelsForGuild };
      } catch (e) { D.probe = 'error ' + String(e).slice(0, 80); }
    } catch (e) { D.result = 'error ' + String(e).slice(0, 80); }
    return stores;
  }
  // The server a user is in voice in (for their server nickname).
  function voiceGuild(st, id) {
    try {
      const all = st.voice?.getAllVoiceStates?.() || {};
      for (const g of Object.keys(all)) if (all[g]?.[id]) return g;
    } catch {}
    return location.pathname.match(/^\/channels\/(\d+)/)?.[1] || null;
  }
  // The voice channel this tab is connected to: {id, name, guild, guildName}.
  let channel = null, channelAt = 0;
  // Who is in the voice channel right now (Discord's own voice states), not counting you.
  function inChannelIds() {
    const st = findStores(), ch = currentChannel()?.id;
    if (!st || !ch) return null;
    try {
      const me = st.user.getCurrentUser()?.id;
      let vs = st.voice?.getVoiceStatesForChannel?.(ch);
      if (!vs) return null;
      const ids = vs instanceof Map ? [...vs.keys()] : Array.isArray(vs) ? vs.map((v) => v?.userId) : Object.keys(vs);
      return ids.filter((id) => typeof id === 'string' && /^\d{15,22}$/.test(id) && id !== me);
    } catch (e) { H.storeDiag.presErr = String(e).slice(0, 80); return null; }
  }
  // Channels the app wants occupancy for (the ones flagged for auto-join).
  let watchIds = [];
  // How many people are in a channel right now, and its user limit (0 = none).
  // Count real people only — not this account itself. Used to decide whether an
  // open channel is worth auto-joining and whether it still has a free slot.
  function channelOcc(id) {
    const st = findStores();
    let count = 0, limit = 0;
    try {
      const me = st?.user?.getCurrentUser?.()?.id;
      const vs = st?.voice?.getVoiceStatesForChannel?.(id);
      if (vs) {
        const ids = vs instanceof Map ? [...vs.keys()] : Array.isArray(vs) ? vs.map((v) => v?.userId) : Object.keys(vs);
        count = ids.filter((u) => typeof u === 'string' && /^\d{15,22}$/.test(u) && u !== me).length;
      }
      const ch = st?.channels?.getChannel?.(id);
      const l = ch?.userLimit ?? ch?.user_limit;
      if (typeof l === 'number' && l > 0) limit = l;
    } catch (e) { H.storeDiag.occErr = String(e).slice(0, 80); }
    return { id, count, limit };
  }
  function meInfo() {
    const st = findStores(); if (!st) return null;
    try {
      const u = st.user.getCurrentUser?.(); if (!u?.id) return null;
      const v = st.voice?.getVoiceStateForUser?.(u.id);
      return { id: u.id, name: clean(u.globalName || u.global_name || u.username) || null,
               self: v ? { channelId: v.channelId || null, deaf: !!v.deaf, selfDeaf: !!v.selfDeaf, mute: !!v.mute, selfMute: !!v.selfMute } : null };
    } catch { return null; }
  }
  // Servers this account is in, with their voice channels (for "Transcribe a channel").
  function directory() {
    const st = findStores(); if (!st?.guilds) return null;
    try {
      return Object.values(st.guilds.getGuilds?.() || {}).map((g) => {
        let vocal = [];
        try { vocal = (st.guildChannels?.getChannels?.(g.id)?.VOCAL || []).map((x) => x.channel || x); } catch {}
        if (!vocal.length) { try { vocal = Object.values(st.channels?.getMutableGuildChannelsForGuild?.(g.id) || {}); } catch {} }
        const channels = vocal.filter((c) => c && (c.type === 2 || c.type === 13)).map((c) => ({ id: c.id, name: clean(c.name) || 'Voice' }));
        return { id: g.id, name: clean(g.name) || 'Server', channels };
      }).filter((g) => g.channels.length).slice(0, 200);
    } catch { return null; }
  }
  function currentChannel() {
    if (Date.now() - channelAt < 2000) return channel;
    channelAt = Date.now();
    const st = findStores();
    if (!st) return channel;
    try {
      const me = st.user.getCurrentUser()?.id;
      const id = st.selected?.getVoiceChannelId?.() || (me && st.voice?.getVoiceStateForUser?.(me)?.channelId) || null;
      if (!id) { channel = null; return channel; }
      const ch = st.channels?.getChannel?.(id);
      const g = ch?.guild_id ? st.guilds?.getGuild?.(ch.guild_id) : null;
      channel = { id, name: clean(ch?.name) || (ch?.guild_id ? 'Voice channel' : 'Private call'), guild: ch?.guild_id || null, guildName: clean(g?.name) || null };
    } catch (e) { H.storeDiag.chanErr = String(e).slice(0, 80); }
    return channel;
  }
  H.currentChannel = currentChannel;

  function storeName(id) {
    const st = findStores();
    if (!st) return null;
    try {
      const g = voiceGuild(st, id);
      let nick = null;
      try { nick = g ? st.member?.getNick?.(g, id) : null; } catch {}
      const u = st.user.getUser(id);
      if (!u) H.storeDiag.misses = (H.storeDiag.misses || 0) + 1;
      const pick = [nick, u?.globalName, u?.username].find((x) => typeof x === 'string' && x.trim());
      return clean(pick) || null;
    } catch (e) { H.storeDiag.lastErr = String(e).slice(0, 80); return null; }
  }

  function scanNames() {
    const avatarIds = new Map();   // id -> Set(names)
    const reactIds = new Map();    // id -> Set(names)
    const via = {}, unnamedRows = [];
    for (const row of document.querySelectorAll(ROW)) {
      const [id, how] = rowUserId(row), name = rowName(row);
      if (!name) continue;
      if (!id) { if (unnamedRows.length < 3) unnamedRows.push({ name, keys: fiberKeys(row) }); continue; }
      const m = how === 'avatar' ? avatarIds : reactIds;
      if (!m.has(id)) m.set(id, new Set());
      m.get(id).add(name);
    }
    for (const [id, names] of avatarIds) if (names.size === 1) { idName.set(id, [...names][0]); via[id] = 'avatar'; }
    for (const [id, names] of reactIds) {
      if (names.size !== 1 || avatarIds.has(id)) continue;     // shared/conflicting: never guess
      idName.set(id, [...names][0]); via[id] = 'react';
    }
    for (const r of readers.values()) {
      if (r.userId && !idName.has(r.userId)) {
        const n = storeName(r.userId);
        if (n) { idName.set(r.userId, n); via[r.userId] = 'store'; }
      }
    }
    H.nameVia = Object.assign(H.nameVia || {}, via);
    H.unnamedRows = unnamedRows;
  }
  // Who each person is on Discord: username (the unique handle), display name and
  // server nickname, for everyone in the call or heard. The server keeps a record of
  // every name a person has used, tied to their account id, which never changes.
  const accounts = new Map();          // id -> { u, g, n, at }
  function accountOf(id) {
    const st = findStores(); if (!st) return null;
    try {
      const u = st.user.getUser(id); if (!u?.username) return null;
      const g = voiceGuild(st, id);
      let nick = null; try { nick = g ? st.member?.getNick?.(g, id) : null; } catch {}
      return { u: String(u.username), g: clean(u.globalName || u.global_name) || null, n: clean(nick) || null };
    } catch { return null; }
  }
  setInterval(() => {
    const ids = new Set([...(inChannelIds() || []), ...[...readers.values()].map((r) => r.userId).filter(Boolean)]);
    for (const id of ids) {
      const old = accounts.get(id);
      if (old && Date.now() - old.at < 60000) continue;           // look each person up again once a minute
      const a = accountOf(id);
      if (a) accounts.set(id, { ...a, at: Date.now() });
    }
    H.accounts = Object.fromEntries([...accounts].map(([id, a]) => [id, { u: a.u, g: a.g, n: a.n }]));
  }, 3000);
  let lastSaved = '';
  setInterval(() => {
    const js = JSON.stringify(Object.fromEntries(idName));
    if (js !== lastSaved) { lastSaved = js; window.postMessage({ __dtNames: JSON.parse(js) }, location.origin); }
  }, 10000);
  window.addEventListener('message', (e) => {
    if (e.source !== window || !e.data?.__dtNamesInit) return;
    for (const [id, n] of Object.entries(e.data.__dtNamesInit)) {
      if (!idName.has(id) && typeof n === 'string') { idName.set(id, n); (H.nameVia = H.nameVia || {})[id] = 'memory'; }
    }
  });
  setInterval(() => { try { scanNames(); } catch (e) { H.errors.push('names: ' + e); } }, 500);

  // ---------------------------------------------------------------- audio
  // Capture runs on the AUDIO thread (AudioWorklet loaded from a blob: URL,
  // which Discord's CSP allows — verified on discord.com). The page's main
  // thread is busy with Discord; a main-thread ScriptProcessor dropped ~60% of
  // the audio once many people had joined, chopping speech into fragments.
  // Chunks are stamped with the audio clock, so late delivery can't distort
  // timing. Silence is dropped inside the worklet (only speech + a short tail
  // is sent). Falls back to ScriptProcessor only if the worklet can't load.
  const WORKLET = `
  class DtCapture extends AudioWorkletProcessor {
    constructor() { super(); this.buf = new Float32Array(3200); this.n = 0; this.t0 = 0; this.tail = 0; }
    process(inputs) {
      const ch = inputs[0] && inputs[0][0];
      if (!ch) return true;
      for (let i = 0; i < ch.length; i++) {
        if (this.n === 0) this.t0 = currentTime + i / sampleRate;
        this.buf[this.n++] = ch[i];
        if (this.n === this.buf.length) {
          let sq = 0; for (let k = 0; k < this.n; k++) sq += this.buf[k] * this.buf[k];
          const rms = Math.sqrt(sq / this.n);
          if (rms >= 0.002 || this.tail > 0) {
            this.tail = rms >= 0.002 ? 4 : this.tail - 1;
            const out = new Int16Array(this.n);
            for (let k = 0; k < this.n; k++) { const v = Math.max(-1, Math.min(1, this.buf[k])); out[k] = v * 32767; }
            this.port.postMessage({ t0: this.t0, rms, pcm: out.buffer }, [out.buffer]);
          } else {
            this.port.postMessage({ t0: this.t0, rms });        // silent: counted, not sent
          }
          this.n = 0;
        }
      }
      return true;
    }
  }
  registerProcessor('dt-capture', DtCapture);`;
  let ctx = null, workletReady = null, wallAtZero = 0;
  function audioCtx() {
    if (!ctx) {
      ctx = new AudioContext({ sampleRate: 16000 });   // Chrome resamples for us
      workletReady = ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })))
        .then(() => (H.capture = 'worklet', true))
        .catch((e) => { H.errors.push('worklet: ' + String(e).slice(0, 60)); H.capture = 'scriptprocessor'; return false; });
    }
    if (ctx.state !== 'running') ctx.resume().catch(() => {});
    return ctx;
  }
  // Audio-clock time -> wall-clock ms (re-synced periodically).
  function syncClock() {
    if (!ctx) return;
    const ts = ctx.getOutputTimestamp();
    if (ts.contextTime) wallAtZero = performance.timeOrigin + ts.performanceTime - ts.contextTime * 1000;
  }
  setInterval(syncClock, 5000);
  const wallOf = (ctxTime) => (wallAtZero || (syncClock(), wallAtZero) || Date.now() - ctx.currentTime * 1000) + ctxTime * 1000;
  document.addEventListener('pointerdown', () => ctx && ctx.resume().catch(() => {}), true);

  // One audio source per person: Discord sometimes keeps two tracks for someone
  // (a reconnect, or a stream with sound). Two readers feeding one person mixes
  // the same speech twice; keep the reader that has been talking, until it has
  // been quiet for 1.5 s.
  const owner = new Map();             // user id -> { entry, at }
  function emit(entry, t, pcm, rms) {
    const own = owner.get(entry.userId), now = Date.now();
    if (own && own.entry !== entry && now - own.at < 1500 && readers.has(own.entry.trackId)) { H.dupDropped = (H.dupDropped || 0) + 1; return; }
    owner.set(entry.userId, { entry, at: now });
    if (rms > 0.01) lastActive.set(entry.userId, Date.now());
    window.postMessage({ __dtAudio: true, user: entry.userId, name: idName.get(entry.userId) || null, ch: currentChannel(), t, pcm }, location.origin, [pcm]);
    H.chunks++;
  }

  async function readTrack(track, userId) {
    if (readers.has(track.id)) { if (userId) readers.get(track.id).userId = userId; return; }
    const c = audioCtx();
    const entry = { userId, trackId: track.id };
    readers.set(track.id, entry);
    H.tracks++;
    const stream = new MediaStream([track]);
    // Chrome only pulls remote WebRTC audio into Web Audio if a media element
    // also consumes the stream; a muted one is enough and plays nothing.
    const sink = new Audio(); sink.muted = true; sink.srcObject = stream; sink.play().catch(() => {});
    const src = c.createMediaStreamSource(stream);
    const silent = c.createGain(); silent.gain.value = 0; silent.connect(c.destination);
    let node;
    if (await workletReady) {
      node = new AudioWorkletNode(c, 'dt-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
      node.port.onmessage = ({ data }) => {
        H.frames = (H.frames || 0) + 1;
        // Audio-clock seconds covered, per this track: a shortfall vs wall time = dropped audio.
        entry.audioSec = (entry.audioSec || 0) + 0.2;
        if (!data.pcm) { H.silent = (H.silent || 0) + 1; return; }
        H.loudMax = Math.max(H.loudMax || 0, data.rms); H.lastHeard = Date.now();
        if (enabled && entry.userId) emit(entry, wallOf(data.t0), data.pcm, data.rms);
      };
    } else {
      node = c.createScriptProcessor(4096, 1, 1);
      let tail = 0;
      node.onaudioprocess = (ev) => {
        H.frames = (H.frames || 0) + 1;
        if (!enabled || !entry.userId) return;
        const x = ev.inputBuffer.getChannelData(0), n = x.length, out = new Int16Array(n);
        let sq = 0;
        for (let i = 0; i < n; i++) { const v = Math.max(-1, Math.min(1, x[i])); out[i] = v * 32767; sq += v * v; }
        const rms = Math.sqrt(sq / n);
        if (rms >= 0.002) H.lastHeard = Date.now();
        if (rms < 0.002) { if (!tail) return; tail--; } else tail = 3;
        emit(entry, wallOf(ev.playbackTime ?? c.currentTime) - (n / c.sampleRate) * 1000, out.buffer, rms);
      };
    }
    node.connect(silent);
    // Only process while the person is actually sending audio: Discord mutes a
    // track when someone leaves or goes quiet, and old tracks pile up otherwise.
    let connected = false;
    const attach = () => { if (!connected) { src.connect(node); connected = true; } };
    const detach = () => { if (connected) { try { src.disconnect(node); } catch {} connected = false; } };
    entry.startedAt = performance.now();
    if (!track.muted) attach();
    track.addEventListener('mute', detach);
    track.addEventListener('unmute', attach);
    entry.stop = () => { detach(); try { node.disconnect(); silent.disconnect(); } catch {} sink.srcObject = null; };
    entry.isActive = () => connected;
    track.addEventListener('ended', () => { entry.stop(); readers.delete(track.id); });
  }

  function onTrack(e) {
    if (e.track.kind !== 'audio') return;
    const sid = e.streams[0]?.id || '';
    const m = sid.match(/^(\d{15,22})-(\d+)$/);
    if (!m && H.badStreamIds.length < 5) H.badStreamIds.push(sid.slice(0, 40));
    readTrack(e.track, m ? m[1] : null);
  }

  const OrigPC = window.RTCPeerConnection;
  if (OrigPC) {
    const Hooked = function (...args) {
      const pc = new OrigPC(...args);
      pcs.add(pc); H.pcs++;
      pc.addEventListener('track', onTrack);
      return pc;
    };
    Hooked.prototype = OrigPC.prototype;
    Object.setPrototypeOf(Hooked, OrigPC);
    window.RTCPeerConnection = Hooked;
  }

  const OrigWS = window.WebSocket;
  const HookedWS = function (url, protocols) {
    const ws = protocols === undefined ? new OrigWS(url) : new OrigWS(url, protocols);
    if (/discord\.media/.test(String(url))) {
      ws.addEventListener('message', (e) => {
        if (typeof e.data !== 'string') return;
        try { const m = JSON.parse(e.data); if (m.op === 5 && m.d?.ssrc && m.d.user_id) ssrcUser.set(m.d.ssrc, m.d.user_id); } catch {}
      });
    }
    return ws;
  };
  HookedWS.prototype = OrigWS.prototype;
  Object.setPrototypeOf(HookedWS, OrigWS);
  window.WebSocket = HookedWS;

  // Re-resolve track -> user via ssrc in case Discord reuses a transceiver.
  setInterval(async () => {
    for (const pc of pcs) {
      if (pc.connectionState === 'closed') { pcs.delete(pc); continue; }
      const st = await pc.getStats().catch(() => null);
      st?.forEach((r) => {
        if (r.type !== 'inbound-rtp' || r.kind !== 'audio') return;
        const u = ssrcUser.get(r.ssrc), entry = readers.get(r.trackIdentifier);
        if (u && entry) entry.userId = u;
      });
    }
    H.live = [...readers.values()].map((r) => idName.get(r.userId) || r.userId);
    H.names = Object.fromEntries(idName);
  }, 2000);
  // Lost call audio (see console.error above): decryption failures in the last 30 s,
  // and how long since anyone was audible (or since this call started, if nobody
  // has been yet). This only notices when ALL audio is lost, not one person's.
  const failsAt = []; let lastFails = 0, callSince = 0;
  setInterval(() => {
    const now = Date.now();
    failsAt.push([now, H.decryptFails - lastFails]); lastFails = H.decryptFails;
    while (failsAt.length && now - failsAt[0][0] > 30000) failsAt.shift();
    if (H.pcs && H.tracks) callSince ||= now; else callSince = 0;
    const heardRef = Math.max(H.lastHeard || 0, callSince);
    window.postMessage({ __dtStatus: true, s: {
      decryptFails: failsAt.reduce((t, x) => t + x[1], 0), heardAgo: heardRef ? Math.round((now - heardRef) / 1000) : null,
      v: H.version, loadedAt: H.loadedAt, pcs: H.pcs, tracks: H.tracks, capture: H.capture,
      activeTracks: [...readers.values()].filter((r) => r.isActive?.()).length,
      silent: H.silent || 0, loudMax: +(H.loudMax || 0).toFixed(3), ctxRate: ctx?.sampleRate, ctxBase: ctx ? +(ctx.baseLatency || 0).toFixed(3) : null, frames: H.frames || 0, chunks: H.chunks,
      ctx: ctx ? ctx.state : 'none', enabled, live: H.live, liveIds: inChannelIds() ?? [...new Set([...readers.values()].map((r) => r.userId).filter(Boolean))], names: H.names, nameVia: H.nameVia, accounts: H.accounts || {},
      badStreamIds: H.badStreamIds, dupDropped: H.dupDropped || 0, errors: [...new Set(H.errors)].slice(-5),
      storeDiag: H.storeDiag, unnamedRows: H.unnamedRows, channel: currentChannel(), me: meInfo(),
      occ: watchIds.length ? watchIds.map(channelOcc) : undefined,
      unnamedLive: [...readers.values()].filter((r) => r.userId && !idName.has(r.userId)).map((r) => r.userId),
    } }, location.origin);
  }, 5000);

  window.addEventListener('message', (e) => {
    if (e.source !== window || !e.data?.__dtControl) return;
    if ('enabled' in e.data) { enabled = !!e.data.enabled; H.enabled = enabled; }
    if (e.data.directory) window.postMessage({ __dtDirectory: directory() }, location.origin);
    if (Array.isArray(e.data.watch)) watchIds = e.data.watch.filter((x) => typeof x === 'string').slice(0, 50);
  });
})();
