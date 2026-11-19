// The Bartop question inbox: a Cloudflare Worker that takes "Ask the bar" submissions from the Gridline and Courtline
// sites, drops garbage, and files the rest as issues in a private GitHub repo for weekly review. Nothing a visitor
// sends is shown on the sites until Steven approves it.
//
// Settings (Cloudflare dashboard > the worker > Settings > Variables and secrets):
//   GITHUB_TOKEN      secret: fine-grained token with Issues read and write on the inbox repo only
//   INBOX_REPO        plain text: owner/repo of the private inbox, e.g. stevopotpie/bartop-inbox
//   TURNSTILE_SECRET  optional secret: Cloudflare Turnstile secret, if the form ever needs a robot check
// Optional KV binding RATE (any namespace) turns on a per-visitor limit of 5 questions a day.

const ORIGINS = ["https://stevopotpie.github.io"];
const LEAGUES = ["NFL", "NBA"];
// words that end a submission on sight; the weekly review catches anything subtler
const BLOCK = /\b(f+u+c+k+\w*|s+h+i+t+\w*|c+u+n+t+\w*|bitch\w*|asshole\w*|dick(head)?s?|pussy|porn\w*|nude\w*|sex\w*|casino\w*|crypto\w*|bitcoin|viagra|cialis|loan\w*|whatsapp|telegram|onlyfans|promo\s*code|free\s*picks|dm\s*me)\b/i;

function cors(origin) {
  return {"Access-Control-Allow-Origin": ORIGINS.includes(origin) ? origin : ORIGINS[0], "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "86400", "Vary": "Origin"};
}
const reply = (origin, status, body) => new Response(JSON.stringify(body), {status, headers: {...cors(origin), "Content-Type": "application/json"}});
const clean = (s, n) => String(s == null ? "" : s).replace(/[\u0000-\u001f\u007f<>]/g, " ").replace(/\s+/g, " ").trim().slice(0, n);

// returns a reason a question is garbage, or "" if it passes
function screen(text, why, name) {
  const all = `${text} ${why} ${name}`;
  if (text.length < 15) return "too short";
  if (text.split(" ").length < 4) return "too few words";
  if (/(https?:|www\.|\.(com|net|org|io|ru|xyz|top|bet)\b|@\w+\.\w|<|>)/i.test(all)) return "links or email";
  if (BLOCK.test(all)) return "blocked word";
  if (/(.)\1{5,}/.test(all)) return "repeated characters";
  const letters = (text.match(/[a-z]/gi) || []).length;
  if (letters / text.length < 0.6) return "not enough words";
  if (text.replace(/[^A-Z]/g, "").length > 0.6 * letters) return "all caps";
  return "";
}

export default {
  async fetch(req, env) {
    const origin = req.headers.get("Origin") || "";
    if (req.method === "OPTIONS") return new Response(null, {status: 204, headers: cors(origin)});
    if (req.method !== "POST" || new URL(req.url).pathname !== "/ask") return reply(origin, 404, {ok: false, error: "not found"});
    if (!ORIGINS.includes(origin)) return reply(origin, 403, {ok: false, error: "Ask from the Gridline or Courtline site."});
    let b; try { b = await req.json(); } catch (e) { return reply(origin, 400, {ok: false, error: "Bad request."}); }

    // honeypot: a hidden field people never see; bots fill it. Pretend it worked.
    if (b.website) return reply(origin, 200, {ok: true});
    const text = clean(b.text, 300), why = clean(b.why, 200), name = clean(b.name, 40);
    const league = LEAGUES.includes(b.league) ? b.league : "NFL", site = b.site === "courtline" ? "courtline" : "gridline";
    const bad = screen(text, why, name);
    if (bad) return reply(origin, 422, {ok: false, error: "That one can't go on tap. Write the theory out in plain words, without links."});

    if (env.TURNSTILE_SECRET) {
      const v = await fetch("https://challenges.cloudflare.com/turnstile/v0/siteverify", {method: "POST",
        body: new URLSearchParams({secret: env.TURNSTILE_SECRET, response: b.turnstile || "", remoteip: req.headers.get("CF-Connecting-IP") || ""})}).then(r => r.json()).catch(() => ({}));
      if (!v.success) return reply(origin, 403, {ok: false, error: "The robot check did not pass. Try again."});
    }
    if (env.RATE) {
      const ip = req.headers.get("CF-Connecting-IP") || "?", day = new Date().toISOString().slice(0, 10), k = `${day}:${ip}`;
      const n = +(await env.RATE.get(k) || 0);
      if (n >= 5) return reply(origin, 429, {ok: false, error: "That's the limit for today. Come back tomorrow."});
      await env.RATE.put(k, String(n + 1), {expirationTtl: 90000});
    }

    const body = [`**Question:** ${text}`, `**Why:** ${why || "(none)"}`, `**Name for credit:** ${name || "(none)"}`,
                  `**League:** ${league}`, `**Asked on:** ${site}`, `**At:** ${new Date().toISOString()}`].join("\n\n");
    const r = await fetch(`https://api.github.com/repos/${env.INBOX_REPO}/issues`, {method: "POST",
      headers: {"Authorization": `Bearer ${env.GITHUB_TOKEN}`, "Accept": "application/vnd.github+json", "User-Agent": "bartop-inbox",
                "Content-Type": "application/json"},
      body: JSON.stringify({title: `[${league}] ${text.slice(0, 90)}`, body, labels: ["pending", league]})});
    if (!r.ok) return reply(origin, 502, {ok: false, error: "Couldn't save that right now. Try again in a minute."});
    return reply(origin, 200, {ok: true});
  }
};
