# Auction Caddy — Usage Guide

The caddy is a mobile-friendly live advisor at `/caddy` (or `/`).  
It watches the ARM bot's activity and gives **you** independent SCB bid advice in real time.

---

## Prerequisites

- The advisor server must be running and accessible (HuggingFace Space or local)
- Spring Boot must be configured to call the advisor with a `tourney_id` in each request
- You and the ARM bot must use the **same server URL** and **same tourney_id**

---

## Step 1 — Connect

Open `https://rdap-smart-bots.hf.space` (or `http://localhost:8000/caddy` locally).

| Field | What to enter |
|---|---|
| **Server URL** | `https://rdap-smart-bots.hf.space` (auto-filled on HF Space) |
| **Tourney ID** | The exact ID Spring Boot is using — check `GET /api/sessions` to find it |

Tap **Connect**. The dot in the header turns green when connected.

> **Finding your tourney_id:** After Spring Boot places its first bid, open  
> `https://rdap-smart-bots.hf.space/api/sessions` in a browser.  
> It returns `{ "active_sessions": [{ "tourney_id": "abc123", ... }] }`.  
> Use that exact string.

---

## Step 2 — Live recommendations

Once a player is nominated in the auction, the caddy auto-updates within 2 seconds:

| Element | Meaning |
|---|---|
| **BIG banner** | `BID UP TO ₹X.XCr` (green) or `PASS` (red) — your primary signal |
| **Floor (current)** | The current bid on the floor from the ARM bot's last call |
| **⭐ SCB ceiling 🔄** | Your personal SCB bid ceiling — tap to refresh with latest state |
| **🤖 ARM bot** | `BIDDING` (green) or `PASSING` (red) — whether the bot would bid |
| **My max avail** | Your purse minus the minimum needed for remaining mandatory slots |
| **Market pill** | `buyer` / `overbid` / `normal` — overall market heat signal |
| **Priority** | SCB's priority score for this player (0–1) |

### Verdict logic
- **BID UP TO ₹X.XCr** → SCB ceiling is above floor and within your budget
- **PASS — SCB ceiling ≤ floor** → the player is not worth more than the current bid
- **PASS — OVER BUDGET** → SCB wants the player but you can't afford it safely

### ARM bot signal
- **BIDDING + you BID** → you and ARM bot both want this player — expect competition, bid decisively
- **PASSING + you BID** → only you want this player — good value opportunity, the bot won't drive the price up
- **BIDDING + you PASS** → let the ARM bot take it, saves your budget

---

## Step 3 — Keep your state accurate

The caddy tracks your purse and roster independently from the server. You must update it manually when you win players.

### When you win a player
In the **Quick Update** card:
1. The current player name is pre-filled
2. Enter the price you paid (₹Cr)
3. Tap **I Won!**

Purse and slots update automatically, and the SCB ceiling immediately re-fetches.

### Adding players manually
In **My State → Add player**:
1. Type the player name (autocomplete from pool)
2. Enter the price paid
3. Tap **Add**

### Removing a player
Tap the **✕** next to any player in the roster — purse is refunded.

### Override purse
If your displayed purse gets out of sync with the platform:
1. Type the correct value in the **Override purse** field
2. Tap **Set**

---

## Step 4 — Recent sales

The **Recent Sales** card appears automatically and shows the last 10 completed sales (player, price, buyer). This is fed by Spring Boot's `POST /api/advice/sold` calls and drives the market mode signal.

---

## Workflow during the auction

```
New player nominated
        │
        ▼ (within 2 seconds)
Caddy shows BID/PASS verdict
        │
   ┌────┴────┐
  BID       PASS
   │
Check ARM bot status
   │
 ┌─┴──────────────────┐
BIDDING              PASSING
(compete carefully)  (bid freely — no bot competition)
   │
Bid on platform
   │
Win player? → tap "I Won!" with price → SCB auto-refreshes
```

---

## Deploying updates

Any code change → push via git:
```bash
git add <changed files>
git commit -m "description"
git push hf main    # deploys to HF Space
git push origin main  # backs up to GitHub
```

The Space rebuilds automatically after each push (~1–2 minutes).
