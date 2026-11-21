// Auto-join decision logic, kept pure (no Electron) so test/autojoin-test.js can
// exercise every branch without launching the app. main.js gathers the live
// state (each account's server directory, who is in each channel, cool-downs)
// and asks these functions what to do.
'use strict';

// The voice channels an account could auto-join right now, best first.
// A channel qualifies when ALL of these hold:
//   - it is flagged for auto-join (the blue icon),
//   - it is one of the account's own servers' voice channels,
//   - at least one real person is in it — people, not your own accounts,
//   - it is not full (its user limit still has a free slot for us),
//   - no other of your accounts is already connected to it (don't pile in),
//   - it is not on cool-down (a join that just failed or a channel we just left),
//   - it is not the channel this account is already sitting in.
// Favourites (the yellow star) come first, then by name — so the list, and the
// order the controller tries channels in, both put favourites up top.
function candidates({ directory = [], flags = {}, occ = {}, ownChannels = new Set(),
                      cooldowns = {}, currentChannelId = null, now = 0 } = {}) {
  const out = [];
  const seen = new Set();
  for (const g of directory) {
    for (const c of (g.channels || [])) {
      if (seen.has(c.id)) continue;
      seen.add(c.id);
      const f = flags[c.id];
      if (!f || !f.auto) continue;                        // not marked for auto-join
      if (currentChannelId && c.id === currentChannelId) continue;  // already here
      if (ownChannels.has(c.id)) continue;                // another of your accounts holds it
      if ((cooldowns[c.id] || 0) > now) continue;         // recently failed or just left
      const o = occ[c.id];
      if (!o || !(o.count >= 1)) continue;                // nobody (real) is in it
      if (o.limit > 0 && o.count >= o.limit) continue;    // full: no slot left for us
      out.push({ guildId: g.id, channelId: c.id, name: c.name || '', guildName: g.name || '',
                 fav: !!f.fav, count: o.count, limit: o.limit || 0 });
    }
  }
  out.sort((a, b) => (Number(b.fav) - Number(a.fav))
    || a.name.localeCompare(b.name) || a.channelId.localeCompare(b.channelId));
  return out;
}

// Has an auto-joined account gone quiet long enough to move on?
// Only when capture is actually running (a suspended audio context would read as
// silence and make the account hop for ever) and we have been here at least the
// idle window, so a freshly joined channel is given its full chance to speak.
function isIdle({ status, joinedAt = 0, now = 0, idleSec = 180 } = {}) {
  if (!status) return false;
  if (status.ctx !== 'running') return false;             // audio not flowing: not real silence
  if (now - joinedAt < idleSec * 1000) return false;      // give the channel its full window first
  const ago = status.heardAgo;                            // seconds since the last speech (or since we joined)
  return typeof ago === 'number' && ago >= idleSec;
}

// The voice channel an account is connected to right now, or null. This is NOT
// the same as "in a call": the app counts a call only once someone else's audio
// track arrives, so an account alone in a channel, or in a silent one, still has
// a connected channel here even though it is not "in a call".
function connectedChannelId(status) {
  return (status && status.me && status.me.self && status.me.self.channelId)
      || (status && status.channel && status.channel.id) || null;
}

module.exports = { candidates, isIdle, connectedChannelId };
