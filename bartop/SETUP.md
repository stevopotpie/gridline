# The Bartop question inbox: setup

Public visitors ask on the Bartop page (gridline/bartop.html and courtline/bartop.html). The form posts to a small free
Cloudflare Worker (`worker.js` in this folder), which drops garbage and files each remaining question as an issue in a
private GitHub repo. Nothing a visitor sends shows on either site until it is approved in the weekly review.

## Garbage filters

- The worker: a hidden honeypot field (bots fill it and get a fake "thanks"), 15 to 300 characters and at least 4 words,
  no links or email addresses, a blocked word list (profanity, spam and gambling promos), no long repeated characters,
  not mostly symbols or all caps, only accepted from stevopotpie.github.io. Optional: 5 questions a day per visitor
  (KV binding `RATE`) and a Cloudflare Turnstile robot check (`TURNSTILE_SECRET`).
- The weekly review: Claude reads the inbox, closes anything off topic, abusive, duplicate or untestable with a
  `rejected` label, and sends a shortlist. Approved picks go on tap on the sites (`site/bartop.json`) and the board.

## One-time setup (about 10 minutes)

1. **Private inbox repo.** On GitHub, create a new **private** repository named `bartop-inbox` (no files needed).
2. **Token for the worker.** GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens >
   Generate new token. Name it `bartop-inbox`, expiration 1 year, Repository access: **Only select repositories** >
   `bartop-inbox`, Repository permissions: **Issues: Read and write**. Copy the token.
3. **Cloudflare Worker.** Sign up free at dash.cloudflare.com. Go to Workers & Pages > Create > Create Worker, name it
   `bartop-inbox`, Deploy. Then Edit code, replace everything with `worker.js` from this folder, and Deploy.
4. **Worker settings.** In the worker, Settings > Variables and secrets > Add:
   - `INBOX_REPO`, type Text, value `stevopotpie/bartop-inbox`
   - `GITHUB_TOKEN`, type Secret, value the token from step 2
   Deploy again if asked.
5. **Optional daily limit.** Storage & Databases > KV > Create a namespace `bartop-rate`, then in the worker's
   Settings > Bindings add a KV namespace binding named `RATE` pointing to it.
6. Send Claude the worker's address (it looks like `https://bartop-inbox.<your-name>.workers.dev`). Claude sets
   `ASK_URL` to that address plus `/ask` in `bartop/page.html` and `site/bartop.html` (Gridline) and `site/bartop_body.html` (Courtline),
   rebuilds both sites and sends a test question.

Until step 6 the form shows "Asking opens soon."
