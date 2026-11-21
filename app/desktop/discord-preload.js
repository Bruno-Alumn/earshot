// Preload for the built-in Discord window (isolated world, sandboxed).
// Injects page-hook.js into Discord's own page
// before any of Discord's scripts run, and relays the per-person audio it
// posts to the main process (which holds the connection to the local server).
const { ipcRenderer, webFrame } = require('electron');

// 1. The hook, in the page's main world, at document start. (Run as privileged
//    injection, so Discord's Content-Security-Policy doesn't apply to it.)
//    Listener accounts first get a silent microphone and no camera (see main.js).
let hook = ipcRenderer.sendSync('ct:hook-source') || {};
if (typeof hook === 'string') hook = { src: hook };   // an older app (before restarting after an update) sends just the code
if (hook.src) webFrame.executeJavaScript(hook.src).catch(() => {});

// 2. The bridge (what content.js + background.js do in Chrome).
let enabled = false;
const setEnabled = (on) => { enabled = on; window.postMessage({ __dtControl: true, enabled: on }, location.origin); };
ipcRenderer.on('ct:status', (_e, on) => setEnabled(!!on));
ipcRenderer.on('ct:names-init', (_e, names) => { if (names) window.postMessage({ __dtNamesInit: names }, location.origin); });
ipcRenderer.on('ct:want-directory', () => window.postMessage({ __dtControl: true, directory: true }, location.origin));
ipcRenderer.on('ct:watch-channels', (_e, ids) => window.postMessage({ __dtControl: true, watch: Array.isArray(ids) ? ids : [] }, location.origin));
setInterval(() => setEnabled(enabled), 2000);          // a hook that loaded late still gets the state
ipcRenderer.send('ct:hello');

window.addEventListener('message', (e) => {
  if (e.source !== window || !e.data) return;
  const d = e.data;
  if (d.__dtAudio) {
    if (enabled) ipcRenderer.send('ct:audio', { user: d.user, name: d.name, ch: d.ch, t: d.t, pcm: d.pcm });
  } else if (d.__dtStatus) {
    ipcRenderer.send('ct:hook-status', d.s);
  } else if (d.__dtDirectory !== undefined) {
    ipcRenderer.send('ct:directory', d.__dtDirectory);
  } else if (d.__dtNames) {
    ipcRenderer.send('ct:names', d.__dtNames);
  }
});
