// One log per channel, and search across every transcript, on a copy of real transcripts.
// node test/channel-test.js <scratch dir>
const { chromium } = require('playwright');
const { spawn, execSync } = require('child_process');
const path = require('path'), fs = require('fs');
const SCRATCH = process.argv[2], OUT = path.join(SCRATCH, 'tx-channel');
const ok = (c, m) => { if (!c) { console.log('FAIL', m); process.exitCode = 1; } else console.log('ok  ', m); };
(async () => {
  fs.rmSync(OUT, { recursive: true, force: true });
  execSync(`cp -R "${path.join(__dirname, '..', 'transcripts')}" "${OUT}"`);
  const srv = spawn(path.join(__dirname, '..', 'server', '.venv', 'bin', 'python'), [path.join(__dirname, '..', 'server', 'server.py'), '--port', '8779', '--out', OUT]);
  let log = ''; srv.stdout.on('data', (d) => (log += d)); srv.stderr.on('data', (d) => (log += d));
  while (!log.includes('Dashboard:')) await new Promise((r) => setTimeout(r, 300));
  const b = await chromium.launch(); const page = await b.newPage({ viewport: { width: 1320, height: 860 } });
  const errs = []; page.on('pageerror', (e) => errs.push(e.message));
  await page.goto('http://127.0.0.1:8779/'); await page.waitForFunction(() => channels.length > 0, null, { timeout: 60000 }); await page.waitForTimeout(1000);

  const st = await page.evaluate(() => ({ calls: calls.length, channels: channels.length, rows: document.querySelectorAll('#calls .call').length,
    names: channels.map((c) => c.srvKey + '|' + chTitle(c)) }));
  ok(st.channels < st.calls && st.rows === st.channels, `${st.calls} calls shown as ${st.channels} channel rows`);
  ok(new Set(st.names).size === st.names.length, 'each channel name appears once per server');

  // the busiest channel opens as one log, with a divider per call
  const big = await page.evaluate(() => channels.slice().sort((a, b) => b.calls.length - a.calls.length)[0].id);
  await page.click(`#calls .call[data-id="${big}"]`); await page.waitForTimeout(1200);
  const v = await page.evaluate(() => ({ n: currentCall().calls.length, parts: partsOf(currentCall()).length, seps: document.querySelectorAll('#feed .callsep').length,
    lines: lines.length, want: currentCall().calls.reduce((t, c) => t + linesOf(c, callRecs(c)).length, 0), earlier: !!$('#earlier'), kpi: $('#k-talk-s').textContent }));
  ok(v.lines === v.want, `channel log has every line of its ${v.n} calls (${v.lines})`);
  ok(v.seps > 0 && v.seps <= v.parts, `${v.seps} call dividers on screen`);
  ok(/over \d+ calls/.test(v.kpi), `length card: "${v.kpi}"`);
  if (v.earlier) {
    await page.click('#earlier'); await page.waitForTimeout(500);
    ok(await page.evaluate(() => document.querySelectorAll('#feed .line').length > 1500), 'Show earlier lines adds older lines');
  }

  // search every transcript; a hit opens its channel at that line
  await page.fill('#hq', 'the'); await page.waitForTimeout(1500);
  const s = await page.evaluate(() => ({ head: $('.hits-h .group').textContent, hits: document.querySelectorAll('#calls .hit').length, chans: $('#hq-ch')?.options.length || 0 }));
  ok(s.hits === 300 && /of [\d,]+/.test(s.head), `search: "${s.head}", ${s.hits} shown`);
  ok(s.chans > 2, `channel picker lists ${s.chans - 1} channels`);
  const pick = await page.evaluate(() => $('#hq-ch').options[2].value);
  await page.selectOption('#hq-ch', pick); await page.waitForTimeout(300);
  ok(await page.evaluate((id) => [...document.querySelectorAll('#calls .hit')].every((e) => e.dataset.ch === id), pick), 'picking a channel shows only its matches');
  const hit = await page.evaluate(() => { const els = document.querySelectorAll('#calls .hit'); const e = els[els.length - 1]; return { ch: e.dataset.ch, t: +e.dataset.t }; });
  await page.click(`#calls .hit[data-t="${hit.t}"]`); await page.waitForTimeout(1500);
  ok(await page.evaluate((h) => currentCall()?.id === h.ch && !!document.querySelector(`#feed .line[data-t="${h.t}"]`), hit), 'a hit opens its channel with that line on screen');
  await page.fill('#hq', ''); await page.waitForTimeout(500);

  // archive, unarchive and delete act on the whole channel
  const target = await page.evaluate(() => { const c = channels.filter((x) => !x.archived).sort((a, b) => a.lines - b.lines)[0]; return { id: c.id, n: c.calls.length, name: chTitle(c), srv: c.srvKey }; });
  await page.click(`#calls .call[data-id="${target.id}"]`, { button: 'right' });
  await page.click('#ctx button[data-a="archive"]'); await page.waitForTimeout(800);
  ok(await page.evaluate((t) => channels.some((c) => c.archived && chTitle(c) === t.name && c.srvKey === t.srv) && !channels.some((c) => !c.archived && chTitle(c) === t.name && c.srvKey === t.srv), target), `archived "${target.name}"`);
  await page.click('#archhead'); await page.waitForTimeout(300);
  await page.click('.archived-tree .call', { button: 'right' });
  await page.click('#ctx button[data-a="unarchive"]'); await page.waitForTimeout(800);
  ok(await page.evaluate((t) => channels.some((c) => !c.archived && c.id === t.id), target), 'unarchived it');
  await page.click(`#calls .call[data-id="${target.id}"]`, { button: 'right' });
  await page.click('#ctx button[data-a="delete"]'); await page.waitForTimeout(600);
  const conf = (await page.textContent('.modal')).replace(/\s+/g, ' ');
  ok(conf.includes(`${target.n} call`), `delete asks about ${target.n} calls: "${conf.slice(0, 120)}"`);
  await page.click('.modal button.danger'); await page.waitForTimeout(1000);
  ok(await page.evaluate((t) => !channels.some((c) => c.id === t.id), target), 'deleted it');
  ok(fs.readdirSync(path.join(OUT, '.trash')).length > 0, 'deleted lines are in .trash');

  ok(!errs.length, 'no page errors ' + errs.join(' | '));
  await page.screenshot({ path: path.join(SCRATCH, 'channel-test.png') });
  await b.close(); srv.kill();
  console.log(process.exitCode ? 'FAIL' : 'PASS');
})().catch((e) => { console.error(e); process.exit(1); });
