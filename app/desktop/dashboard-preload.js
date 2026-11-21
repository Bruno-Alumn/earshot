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
  addAccount: () => ipcRenderer.invoke('ct:add-account'),
  removeAccount: (id) => ipcRenderer.invoke('ct:remove-account', String(id)),
  setRole: (id, role) => ipcRenderer.invoke('ct:set-role', String(id), String(role)),
  setProxy: (id, value) => ipcRenderer.invoke('ct:set-proxy', String(id), String(value == null ? '' : value)),
  appSettings: () => ipcRenderer.invoke('ct:app-settings'),
  openPluginsFolder: () => ipcRenderer.send('ct:open-plugins'),
  setAppSetting: (key, value) => ipcRenderer.invoke('ct:set-app-setting', String(key), !!value),
  onState: (fn) => ipcRenderer.on('ct:app-state', (_e, s) => fn(s)),
});
