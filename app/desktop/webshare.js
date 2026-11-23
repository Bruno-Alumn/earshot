// Webshare (webshare.io) proxy lists, for picking an account's proxy.
//
// The API key stays in the main process (encrypted at rest by main.js). The
// dashboard only ever sees each proxy's id, address, place and whether it
// works; the username and password go straight from here into the account's
// saved proxy.
//
// API: GET /api/v2/proxy/list/?mode=direct|backbone&page=N&page_size=N
//      Authorization: Token <key>
//      -> { count, next, previous, results: [{ id, username, password,
//           proxy_address, port, valid, country_code, city_name, ... }] }
// Backbone (residential) plans connect through p.webshare.io; their
// proxy_address is null.

const BACKBONE_HOST = 'p.webshare.io';
const PAGE_SIZE = 100;
const MAX_PAGES = 20;                 // 2,000 proxies is plenty to pick from (and stays under the rate limit)

class WebshareError extends Error {}

function apiBase() {
  return (process.env.CT_TEST === '1' && process.env.CT_WEBSHARE_API) || 'https://proxy.webshare.io';
}

async function getJson(fetchFn, key, url) {
  let res;
  try {
    res = await fetchFn(url, { headers: { Authorization: `Token ${key}`, Accept: 'application/json' } });
  } catch {
    throw new WebshareError('Couldn\'t reach Webshare. Check your internet connection.');
  }
  if (res.status === 401) throw new WebshareError('Webshare didn\'t accept that API key. Copy it again from Webshare\'s API keys page.');
  if (res.status === 403) throw new WebshareError('Your Webshare plan doesn\'t allow listing proxies through the API.');
  if (res.status === 429) throw new WebshareError('Webshare is limiting requests. Wait a minute and try again.');
  let body = null;
  try { body = await res.json(); } catch {}
  if (!res.ok) return { status: res.status, body };
  if (!body || !Array.isArray(body.results)) throw new WebshareError('Webshare sent back something unexpected.');
  return { status: res.status, body };
}

// Every proxy on the account (direct, or backbone for residential plans).
async function listProxies(key, fetchFn = fetch) {
  key = String(key || '').trim();
  if (!key) throw new WebshareError('Add your Webshare API key first.');
  for (const mode of ['direct', 'backbone']) {
    const out = [];
    let url = `${apiBase()}/api/v2/proxy/list/?mode=${mode}&page=1&page_size=${PAGE_SIZE}`;
    let first = true;
    for (let i = 0; url && i < MAX_PAGES; i++) {
      const r = await getJson(fetchFn, key, url);
      if (r.status === 400 && first && mode === 'direct') { url = null; break; }   // a residential plan: backbone only
      if (r.status >= 400) throw new WebshareError(`Webshare answered with an error (${r.status}).`);
      out.push(...r.body.results.map((p) => normalize(p, mode)).filter(Boolean));
      url = r.body.next || null;
      first = false;
    }
    if (out.length || mode === 'backbone' || !first) return out;
  }
  return [];
}

function normalize(p, mode) {
  if (!p || p.port == null) return null;
  const host = p.proxy_address || BACKBONE_HOST;
  const port = Number(p.port);
  if (!Number.isInteger(port) || port < 1 || port > 65535) return null;
  return {
    id: String(p.id ?? `${host}:${port}:${p.username || ''}`),
    host, port, mode,
    username: p.username || '', password: p.password || '',
    country: p.country_code || '', city: p.city_name || '',
    valid: p.valid !== false,
  };
}

// The proxy string saved on the account (proxy.js reads it).
function proxyString(p) {
  const h = p.host.includes(':') ? `[${p.host}]` : p.host;
  if (!p.username) return `http://${h}:${p.port}`;
  return `http://${encodeURIComponent(p.username)}:${encodeURIComponent(p.password)}@${h}:${p.port}`;
}

// What the dashboard may see: no username or password.
function publicView(p) {
  return { id: p.id, host: p.host, port: p.port, country: p.country, city: p.city, valid: p.valid, backbone: p.mode === 'backbone' };
}

module.exports = { listProxies, proxyString, publicView, WebshareError, BACKBONE_HOST };
