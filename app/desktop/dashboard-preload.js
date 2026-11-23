// Preload for the dashboard window: the dashboard's only way to talk to the app.
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('ctApp', {
  platform: process.platform,
  server: (action) => ipcRenderer.send('ct:server', String(action)),
  save: (name, content) => ipcRenderer.invoke('ct:save', String(name), String(content)),
  relaunch: (opts) => ipcRenderer.send('ct:relaunch', { rebuild: !!opts?.rebuild, deps: !!opts?.deps, app: !!opts?.app }),
  openDiscord: (id) => ipcRenderer.send('ct:open-discord', id == null ? undefined : String(id)),
  setPaused: (on) => ipcRenderer.send('ct:pause', !!on),
  setMuted: (on, id) => ipcRenderer.send('ct:mute', !!on, id == null ? undefined : String(id)),
  // accounts (several Discord accounts, one window each)
  accounts: () => ipcRenderer.invoke('ct:accounts'),
  transcribe: (guildId, channelId) => ipcRenderer.invoke('ct:transcribe', String(guildId), String(channelId)),
  channelFlags: () => ipcRenderer.invoke('ct:channel-flags'),
  setChannelFlag: (id, key, on) => ipcRenderer.invoke('ct:set-channel-flag', String(id), String(key), !!on),
  addAccount: (mode, proxy, creds) => ipcRenderer.invoke('ct:add-account', mode === 'create' ? 'create' : 'signin',
    proxy && typeof proxy === 'object' ? { webshare: proxy.webshare ? String(proxy.webshare) : undefined, value: proxy.value ? String(proxy.value) : undefined } : null,
    creds && typeof creds === 'object' ? Object.fromEntries(['email', 'username', 'displayName', 'password'].map((k) => [k, String(creds[k] || '')])) : null),
  // an account's saved sign-in details: fetched only when you open them
  accountDetails: (id) => ipcRenderer.invoke('ct:account-details', String(id)),
  setAccountDetails: (id, c) => ipcRenderer.invoke('ct:set-account-details', String(id), c && typeof c === 'object' ? Object.fromEntries(['email', 'username', 'displayName', 'password'].map((k) => [k, String(c[k] || '')])) : null),
  // Webshare: the key and proxy passwords stay in the app; this page sees addresses and places
  webshare: () => ipcRenderer.invoke('ct:webshare'),
  setWebshareKey: (key) => ipcRenderer.invoke('ct:webshare-key', String(key == null ? '' : key)),
  webshareList: (fresh) => ipcRenderer.invoke('ct:webshare-list', !!fresh),
  webshareUse: (id, proxyId) => ipcRenderer.invoke('ct:webshare-use', String(id), String(proxyId)),
  onCommand: (fn) => ipcRenderer.on('ct:command', (_e, c) => fn(String(c))),
  removeAccount: (id) => ipcRenderer.invoke('ct:remove-account', String(id)),
  setRole: (id, role) => ipcRenderer.invoke('ct:set-role', String(id), String(role)),
  setProxy: (id, value) => ipcRenderer.invoke('ct:set-proxy', String(id), String(value == null ? '' : value)),
  appSettings: () => ipcRenderer.invoke('ct:app-settings'),
  openPluginsFolder: () => ipcRenderer.send('ct:open-plugins'),
  setAppSetting: (key, value) => ipcRenderer.invoke('ct:set-app-setting', String(key), !!value),
  onState: (fn) => ipcRenderer.on('ct:app-state', (_e, s) => fn(s)),
  // sync with another computer (data only; pairing code typed by hand)
  sync: () => ipcRenderer.invoke('ct:sync'),
  syncPairStart: () => ipcRenderer.invoke('ct:sync-pair-start'),
  syncPairCancel: () => ipcRenderer.invoke('ct:sync-pair-cancel'),
  syncJoin: (code) => ipcRenderer.invoke('ct:sync-join', String(code || '')),
  syncNow: () => ipcRenderer.invoke('ct:sync-now'),
  syncUnpair: (id) => ipcRenderer.invoke('ct:sync-unpair', String(id)),
  syncRename: (name) => ipcRenderer.invoke('ct:sync-rename', String(name || '')),
  onSync: (fn) => ipcRenderer.on('ct:sync-changed', (_e, v) => fn(v)),
});
