// Study: overview, word patterns and how people talk, on a copy of real transcripts.
// node test/insights-test.js <scratch dir>
const { chromium } = require('playwright');
const { spawn, execSync } = require('child_process');
const path = require('path'), fs = require('fs');
const SCRATCH = process.argv[2], OUT = path.join(SCRATCH, 'tx-insights');
const ok = (c, m) => { if (!c) { console.log('FAIL', m); process.exitCode = 1; } else console.log('ok  ', m); };
(async () => {
  fs.rmSync(OUT, { recursive: true, force: true });
  execSync(`cp -R "${path.join(__dirname, '..', 'transcripts')}" "${OUT}"`);
  const srv = spawn(path.join(__dirname, '..', 'server', '.venv', 'bin', 'python'), [path.join(__dirname, '..', 'server', 'server.py'), '--port', '8780', '--out', OUT]);
  let log = ''; srv.stdout.on('data', (d) => (log += d)); srv.stderr.on('data', (d) => (log += d));
  while (!log.includes('Dashboard:')) await new Promise((r) => setTimeout(r, 300));
  const b = await chromium.launch(); const page = await b.newPage({ viewport: { width: 1320, height: 900 } });
  const errs = []; page.on('pageerror', (e) => errs.push(e.message));
  await page.goto('http://127.0.0.1:8780/'); await page.waitForTimeout(1500);

  const api = await page.evaluate(() => fetch('/api/insights').then((r) => r.json()));
  ok(api.totals.lines > 1000 && api.people.length > 5 && api.words.length > 20, `api: ${api.totals.lines} lines, ${api.people.length} people, ${api.words.length} words`);
  ok(api.days.reduce((t, d) => t + d.lines, 0) === api.totals.lines && api.hours.reduce((a, b) => a + b, 0) === api.totals.lines, 'per-day and per-hour counts add up to every line');
  ok(!api.words.some((w) => ['the', 'yeah', 'like', 'and'].includes(w.w)), 'small words are left out of the top words');

  await page.click('#rail [data-view="study"]');
  await page.waitForSelector('.ifind li', { timeout: 30000 });
  const ov = await page.evaluate(() => ({ h: $('#view-study h1').textContent, finds: document.querySelectorAll('.ifind li').length, tiles: document.querySelectorAll('.itile').length, charts: document.querySelectorAll('#view-study .ichart').length, chans: document.querySelectorAll('#view-study .stable tr').length }));
  ok(ov.h === 'Overview' && ov.finds >= 6 && ov.tiles === 5 && ov.charts >= 2, `overview: ${ov.finds} findings, ${ov.tiles} tiles, ${ov.charts} charts`);
  await page.screenshot({ path: path.join(SCRATCH, 'ins-overview.png'), fullPage: true });

  // a server filter narrows it
  const sk = api.servers[0].key;
  await page.selectOption('[data-i="srv"]', sk); await page.waitForFunction((k) => insCache?.key === k + '|all', sk, { timeout: 30000 }); await page.waitForTimeout(500);
  const one = await page.evaluate(() => insCache.I);
  ok(one.srv === sk && one.totals.lines <= api.totals.lines && one.channels.every((c) => c.key.startsWith(sk + '|')), `one server: ${one.totals.lines} lines`);
  await page.selectOption('[data-i="srv"]', ''); await page.waitForTimeout(2500);

  await page.click('[data-tab="words"]'); await page.waitForSelector('.ibrow', { timeout: 30000 });
  const w = await page.evaluate(() => ({ bars: document.querySelectorAll('.ibrow').length, chips: document.querySelectorAll('.iword').length, sig: document.querySelectorAll('.isig tr').length }));
  ok(w.bars === 30 && w.chips > 20 && w.sig > 3, `words: ${w.bars} bars, ${w.chips} phrases, ${w.sig - 1} people with their own words`);
  await page.screenshot({ path: path.join(SCRATCH, 'ins-words.png'), fullPage: true });
  const word = await page.evaluate(() => $('.ibrow').dataset.q);
  await page.click('.ibrow'); await page.waitForTimeout(1500);
  ok(await page.evaluate((q) => curView === 'calls' && $('#hq').value === q && document.querySelectorAll('#calls .hit').length > 0, word), `a word opens a search for "${word}"`);
  await page.fill('#hq', '');

  await page.click('#rail [data-view="study"]'); await page.click('[data-tab="talk"]'); await page.waitForSelector('.itable', { timeout: 30000 });
  const t = await page.evaluate(() => ({ rows: document.querySelectorAll('.itable tr').length - 1, pairs: document.querySelectorAll('.ipair').length, first: $('.itable tr:nth-child(2) td').textContent }));
  ok(t.rows > 5 && t.pairs > 3, `talk: ${t.rows} people, ${t.pairs} pairs`);
  await page.click('[data-sort="swearRate"]'); await page.waitForTimeout(200);
  const sorted = await page.evaluate(() => [...document.querySelectorAll('.itable tr')].slice(1).map((r) => +r.children[8].textContent));
  ok(sorted.every((v, i) => !i || sorted[i - 1] >= v), 'sorting by swears puts the most first');
  await page.screenshot({ path: path.join(SCRATCH, 'ins-talk.png'), fullPage: true });

  // with reads off, Study still shows, without the read tabs
  await page.evaluate(() => { serverSettings.personality = false; applySettings(); }); await page.waitForTimeout(800);
  ok(await page.evaluate(() => getComputedStyle($('#rail [data-view="study"]')).display !== 'none' && !$('[data-tab="group"]') && !!$('[data-tab="overview"]')), 'Study stays without the read tabs when reads are off');

  ok(!errs.length, 'no page errors ' + errs.join(' | '));
  await b.close(); srv.kill();
  console.log(process.exitCode ? 'FAIL' : 'PASS');
})().catch((e) => { console.error(e); process.exit(1); });
