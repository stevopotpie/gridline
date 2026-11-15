// Gridline service worker: lets the installed app open fast and still show the last pages it loaded when offline.
// Pages are fetched fresh from the network first (projections change often) and fall back to the saved copy;
// icons and other files show the saved copy while a fresh one downloads. Other sites (fonts, live scores) are left alone.
const CACHE = "gridline-v1";
const CORE = ["./", "season.html", "in-game.html", "model.html", "bet-log.html", "about.html", "icon-192.png", "manifest.webmanifest"];

self.addEventListener("install", e => {
  e.waitUntil(caches.open(CACHE).then(c => Promise.allSettled(CORE.map(u => c.add(u)))).then(() => self.skipWaiting()));
});
self.addEventListener("activate", e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
const OFFLINE = `<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Gridline offline</title>
<body style="margin:0;font:16px/1.5 system-ui,sans-serif;background:#141a16;color:#f1f5f2;display:grid;place-items:center;min-height:100vh;padding:24px;box-sizing:border-box">
<div style="max-width:32ch;text-align:center"><h1 style="text-transform:uppercase;letter-spacing:.03em">Gridline</h1>
<p>You're offline, and this page hasn't been saved on this device yet. Pages you've opened before still work offline. Reconnect to get the latest projections.</p></div>`;

self.addEventListener("fetch", e => {
  const req = e.request, url = new URL(req.url);
  if (req.method !== "GET" || url.origin !== location.origin || !url.pathname.startsWith("/gridline/")) return;
  if (req.mode === "navigate") {
    e.respondWith(fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req, {ignoreSearch: true}).then(hit => hit || new Response(OFFLINE, {headers: {"Content-Type": "text/html; charset=utf-8"}}))));
    return;
  }
  e.respondWith(caches.match(req).then(hit => {
    const fresh = fetch(req).then(res => { if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); } return res; }).catch(() => hit);
    return hit || fresh;
  }));
});
