// Unit test for the pure auto-join decision logic (app/desktop/autojoin.js).
//   node test/autojoin-test.js
const assert = require('assert');
const { candidates, isIdle, connectedChannelId, isAlone, busiestFirst, classifyLeave, guildOf } = require('../app/desktop/autojoin.js');

let failed = false;
const ok = (name, cond) => { if (cond) console.log('  ok  ' + name); else { console.log('  FAIL ' + name); failed = true; } };

// One server, four voice channels.
const directory = [{ id: 'g1', name: 'Stone Harbor', channels: [
  { id: 'cHot',  name: 'General' },     // flagged, busy, room -> the obvious pick
  { id: 'cFav',  name: 'Lounge' },      // flagged + favourite, busy -> must win over General
  { id: 'cFull', name: 'Standup' },     // flagged but full
  { id: 'cEmpty',name: 'AFK' },         // flagged but empty
  { id: 'cPlain',name: 'Music' },       // busy but NOT flagged
] }];
const flags = {
  cHot:  { auto: true },
  cFav:  { auto: true, fav: true },
  cFull: { auto: true },
  cEmpty:{ auto: true },
  cPlain:{ fav: true },                 // starred but not auto -> never auto-joined
};
const occ = {
  cHot:  { count: 3, limit: 0 },
  cFav:  { count: 1, limit: 10 },
  cFull: { count: 5, limit: 5 },
  cEmpty:{ count: 0, limit: 0 },
  cPlain:{ count: 4, limit: 0 },
};

// 1. Picks flagged, occupied, non-full channels; favourite first; skips full/empty/unflagged.
const list = candidates({ directory, flags, occ, now: 1000 });
ok('favourite sorts first', list[0] && list[0].channelId === 'cFav');
ok('then the other busy flagged channel', list[1] && list[1].channelId === 'cHot');
ok('exactly two candidates', list.length === 2);
ok('full channel excluded', !list.some((c) => c.channelId === 'cFull'));
ok('empty channel excluded', !list.some((c) => c.channelId === 'cEmpty'));
ok('unflagged channel excluded', !list.some((c) => c.channelId === 'cPlain'));

// 2. A channel another of your accounts already holds is skipped (no piling in).
const own = candidates({ directory, flags, occ, now: 1000, ownChannels: new Set(['cFav']) });
ok('own-held channel skipped', own[0] && own[0].channelId === 'cHot' && !own.some((c) => c.channelId === 'cFav'));

// 3. The channel we are already in is not re-offered.
const here = candidates({ directory, flags, occ, now: 1000, currentChannelId: 'cFav' });
ok('current channel not re-offered', !here.some((c) => c.channelId === 'cFav') && here[0].channelId === 'cHot');

// 4. Cool-down hides a channel until its time passes.
ok('cooldown hides channel', !candidates({ directory, flags, occ, now: 1000, cooldowns: { cFav: 5000 } }).some((c) => c.channelId === 'cFav'));
ok('cooldown expires', candidates({ directory, flags, occ, now: 6000, cooldowns: { cFav: 5000 } }).some((c) => c.channelId === 'cFav'));

// 5. Just-under-limit still has room for us; exactly-at-limit is full.
ok('room for us (4/5)', candidates({ directory: [{ id: 'g', name: 'g', channels: [{ id: 'x', name: 'x' }] }], flags: { x: { auto: true } }, occ: { x: { count: 4, limit: 5 } }, now: 0 }).length === 1);
ok('full (5/5)', candidates({ directory: [{ id: 'g', name: 'g', channels: [{ id: 'x', name: 'x' }] }], flags: { x: { auto: true } }, occ: { x: { count: 5, limit: 5 } }, now: 0 }).length === 0);

// 6. isIdle: needs a running context, the full window elapsed, and real silence.
const T = 180;
ok('idle after silent window', isIdle({ status: { ctx: 'running', heardAgo: 181 }, joinedAt: 0, now: 200000, idleSec: T }));
ok('not idle while someone talks', !isIdle({ status: { ctx: 'running', heardAgo: 5 }, joinedAt: 0, now: 200000, idleSec: T }));
ok('not idle before the window', !isIdle({ status: { ctx: 'running', heardAgo: 999 }, joinedAt: 190000, now: 200000, idleSec: T }));
ok('not idle when context is suspended', !isIdle({ status: { ctx: 'suspended', heardAgo: 999 }, joinedAt: 0, now: 200000, idleSec: T }));
ok('not idle with no status', !isIdle({ status: null, now: 200000, idleSec: T }));

// 7. connectedChannelId reads self first, then the selected channel, else null.
ok('connected via self', connectedChannelId({ me: { self: { channelId: 'c9' } }, channel: { id: 'c1' } }) === 'c9');
ok('connected via selected channel', connectedChannelId({ channel: { id: 'c1' } }) === 'c1');
ok('not connected', connectedChannelId({ me: { self: { channelId: null } } }) === null);

// 8. isAlone: nobody but your own accounts left; unknown occupancy is not alone.
ok('alone when nobody else is there', isAlone({ occ: { c1: { count: 0, limit: 0 } }, channelId: 'c1' }));
ok('alone with only another of your accounts', isAlone({ occ: { c1: { count: 1, limit: 0 } }, channelId: 'c1', ownHere: 1 }));
ok('not alone with one real person', !isAlone({ occ: { c1: { count: 1, limit: 0 } }, channelId: 'c1' }));
ok('not alone with a person and your account', !isAlone({ occ: { c1: { count: 2, limit: 0 } }, channelId: 'c1', ownHere: 1 }));
ok('unknown occupancy is not alone', !isAlone({ occ: {}, channelId: 'c1' }));
ok('not connected is not alone', !isAlone({ occ: { c1: { count: 0 } }, channelId: null }));

// 9. busiestFirst: favourites first, then the most people.
const busy = busiestFirst(candidates({ directory, flags, occ, now: 1000, cooldowns: { cFav: 5000 } }));
ok('busiest open channel first', busy[0] && busy[0].channelId === 'cHot');
const busy2 = busiestFirst([{ channelId: 'a', name: 'A', fav: false, count: 2 }, { channelId: 'b', name: 'B', fav: false, count: 6 }, { channelId: 'f', name: 'F', fav: true, count: 1 }]);
ok('favourite, then most people', busy2.map((c) => c.channelId).join() === 'f,b,a');

// 10. classifyLeave: a reload since we last saw it there is ours; anything else is a removal.
ok('page reloaded: goes back', classifyLeave({ lastIn: { channelId: 'c1', at: 1000 }, navAt: 2000 }) === 'reloaded');
ok('no reload: removed (kick, disconnect, Leave)', classifyLeave({ lastIn: { channelId: 'c1', at: 2000 }, navAt: 1000 }) === 'removed');
ok('never in a channel: nothing to do', classifyLeave({ lastIn: null, navAt: 5 }) === null);
ok('guildOf finds the server', guildOf(directory, 'cHot')?.guildId === 'g1' && guildOf(directory, 'nope') === null);

console.log(failed ? '\nFAILED' : '\nPASS');
process.exit(failed ? 1 : 0);
