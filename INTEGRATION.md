# Integration Plan — Fantasy Cricket Platform ↔ Auction Advisor

This document is the complete integration specification. Any Copilot session on either side can implement from this alone.

---

## Overview

The **Fantasy Cricket Platform** (Spring Boot, Java 17) runs the live IPL fantasy auction — players are auctioned lot by lot, teams bid, and purses are deducted.

The **Auction Advisor** (Python/FastAPI, this repo) runs ML-backed bidding strategies. It holds projected player points, VORP, tier ratings, and two evolved strategies co-optimised via CMA-ES.

The integration makes the advisor act as a **bot team** inside the platform: it polls for live lots, calls itself for a bid ceiling, and submits bids via the platform's API. It also receives sold events after each hammer so its market model stays current.

---

## System Map

```
   Friend's laptop                           Your laptop / ngrok / Fly.io
   ──────────────────────────────────────    ──────────────────────────────────
   Spring Boot  :8080                        Advisor (Python/FastAPI)  :8000

   BotBidService          ── POST /api/advice ──────────────────────────────►
     (polls every 3s)     ◄─ { recommended_bid: 18.5 } ─────────────────────
     (decides to bid)
     │
     │ internal Java call (same JVM, no network)
     ▼
   AuctionServiceImpl.placeBid()

   AuctionServiceImpl.closeAuction()
                          ── POST /api/advice/sold ────────────────────────►
                          ◄─ { ok: true } ─────────────────────────────────
```

### Key clarification — who runs where

| Component | Runs on | Language |
|---|---|---|
| `BotBidService` — polls, decides, places bids | Friend's laptop (Spring Boot) | Java |
| `AuctionAdvisorClient` — HTTP calls to advisor | Friend's laptop (Spring Boot) | Java |
| `AuctionServiceImpl.placeBid()` — actual bid | Friend's laptop (Spring Boot) | Java |
| Strategy logic (`SquadCompletionBidder` etc.) | **Your machine** (Advisor) | Python |
| `MarketAnalyzer` — market signals | **Your machine** (Advisor) | Python |

The **bot logic runs on your friend's laptop**. The advisor is purely a strategy oracle — it receives context, returns one number, and never touches the platform's database.

The only thing your friend needs to configure is:
```properties
advisor.base-url=http://<wherever-advisor-is-running>:8000
```

### Deployment options for the advisor

| Option | When to use | advisor.base-url |
|---|---|---|
| ngrok tunnel | Dev / live session together | `https://abc123.ngrok-free.app` |
| Same WiFi | In the same room | `http://192.168.1.x:8000` |
| Fly.io / Railway | Permanent / real auction | `https://auction-advisor.fly.dev` |
| Docker on your machine + ngrok | Cleanest dev setup | ngrok URL pointing to container |

---

## Platform API — what the bot consumes

All base URLs from `application.properties`:
```properties
advisor.base-url=http://localhost:8000   # add this
bot.team-id=${BOT_TEAM_ID:}              # add this
bot.tourney-id=${BOT_TOURNEY_ID:}        # add this
bot.enabled=${BOT_ENABLED:false}         # add this
```

### Poll for live lots
```
GET /api/auctions/fantasy-tourney/{botTourneyId}/live
→ List<AuctionDto>
```

Relevant `AuctionDto` fields:
```
id                      String   auction UUID
playerName              String   "Ruturaj Gaikwad" — use directly in advisor call
playerId                String   "ruturaj_gaikwad" (slug)
bidAmount               Long     current highest bid, in paisa (NOT currentBid which is count)
currentWinningTeamId    String   skip if == botTeamId
status                  String   "UPCOMING" | "LIVE" | "COMPLETED" | "CANCELLED"
boosterType             String   null for player lots; skip if non-null
```

Only act on auctions where `status == "LIVE"`, `boosterType == null`, and `currentWinningTeamId != botTeamId`.

### Get all teams' state (for building `opponents` array)
```
GET /api/fantasy-tourneys/{botTourneyId}/fantasy-teams
→ List<FantasyTeamDto>
```

Relevant `FantasyTeamDto` fields:
```
id          String    team UUID
name        String    display name
purse       Long      current purse, in paisa
squadJson   String    JSON: { playerList: [{playerId, bid, auctionId}], totalCost }
```

To get player names from squadJson, use the squad endpoint instead:
```
GET /api/auctions/squad/{botTourneyId}/{teamId}
→ List<FantasyTourneySquadPlayerDto>
  { playerId, playerName, playerType, purchasePrice }
```

### Place a bid
```
POST /api/auctions/{auctionId}/bid
Body: { "teamId": "{botTeamId}", "bidAmount": 160000000 }
```

`bidAmount` is the **total new bid in paisa** (not an increment). Server rejects if `≤ currentBidAmount`.

**Mirror the frontend increment logic** (from `common.js`):
```java
long nextBidAmount(long currentBid) {
    if (currentBid < 10_000_000) return currentBid + 1_000_000;   // <₹1Cr  → +₹10L
    if (currentBid < 80_000_000) return currentBid + 2_500_000;   // ₹1–8Cr → +₹25L
    return currentBid + 5_000_000;                                  // ₹8Cr+  → +₹50L
}
```

**Decision logic:**
```java
long nextBid = nextBidAmount(auction.getBidAmount());     // next increment
long ceiling = (long)(advisorRecommendedBid * 10_000_000); // advisor → paisa

if (nextBid <= ceiling && nextBid <= myTeam.getPurse()) {
    auctionService.placeBid(auctionId, botTeamId, nextBid);
}
```

---

## Advisor API — what the platform calls

Both endpoints run on this FastAPI server.

### POST /api/advice — get bid ceiling

Request:
```json
{
  "player_name": "Ruturaj Gaikwad",
  "my_purse":    85.0,
  "my_roster":   ["Sunil Narine", "MS Dhoni"],
  "opponents": [
    { "name": "Alice FC", "purse": 72.0, "roster": ["Virat Kohli"] }
  ],
  "strategy":   "SquadCompletionBidder",
  "evolved":    true,
  "tourney_id": "GroupA_IPL2026"
}
```

- `player_name`: use `AuctionDto.playerName` directly — full display names match `player_master.json`
- `my_purse`: `myTeam.getPurse() / 10_000_000.0` — convert paisa → ₹ Crore
- `my_roster`: player display names from `GET /api/auctions/squad/{tourneyId}/{botTeamId}`
- `opponents`: all teams except the bot, same purse conversion, names from their squad endpoints
- `strategy`: `"SquadCompletionBidder"` (default) — or `"AdaptiveRecoveryManager"`
- `tourney_id`: **include this** — it connects the advice call to the `MarketAnalyzer` session fed by `/api/advice/sold` events

Response:
```json
{
  "recommended_bid": 18.50,
  "auction_name":    "Ruturaj Gaikwad",
  "my_slots":        12,
  "my_max_bid":      84.00
}
```

`recommended_bid` is in ₹ Crore. Convert to paisa: `(long)(recommended_bid * 10_000_000)`.

### POST /api/advice/sold — notify sale

**✅ Built and tested.** Call this from `AuctionServiceImpl.closeAuction()` after the squad entry is persisted and purse is deducted.

Request:
```json
{
  "player_name": "Ruturaj Gaikwad",
  "price":       18.0,
  "buyer":       "Alice FC",
  "tourney_id":  "GroupA_IPL2026"
}
```

- `price`: `auction.getBidAmount() / 10_000_000.0`
- `buyer`: `auction.getCurrentWinningTeamName()`

Response: `{ "ok": true }`

This feeds `MarketAnalyzer` with real sale data — without it, market efficiency signals (`overbid`/`buyer`/`normal`) are blind. This is the most important event for the advisor's dynamic advantage.

---

## What to build — Spring Boot side

### 1. `AuctionAdvisorClient.java` (new `@Service`)

Two outbound HTTP methods using `RestTemplate`:

```java
@Bean RestTemplate restTemplate() { return new RestTemplate(); }  // in a @Configuration class
```

```java
// method 1
AdvisorResponse getAdvice(String playerName, double myPurse, List<String> myRoster,
                          List<OpponentDto> opponents, String strategy);

// method 2
void notifySold(String playerName, double price, String buyerName, String tourneyId);
```

DTOs needed (create as simple POJOs/records):
- `AdvisorAdviceRequest` — serialises to the JSON above
- `AdvisorAdviceResponse` — `recommended_bid`, `auction_name`, `my_slots`, `my_max_bid`
- `AdvisorOpponentDto` — `name`, `purse`, `roster: List<String>`
- `AdvisorSoldEvent` — `player_name`, `price`, `buyer`, `tourney_id`

### 2. `BotBidService.java` (new `@Service`)

```java
@Scheduled(fixedDelay = 3000)   // poll every 3 seconds
void tick() {
    if (!botEnabled || botTeamId.isBlank() || botTourneyId.isBlank()) return;

    // 1. Get live lots
    List<AuctionDto> live = auctionService.getLiveAuctions(botTourneyId)
        .stream()
        .filter(a -> a.getBoosterType() == null)
        .filter(a -> !botTeamId.equals(a.getCurrentWinningTeamId()))
        .toList();
    if (live.isEmpty()) return;

    // 2. Refresh state (cache teams for this tick)
    FantasyTeamDto myTeam = getMyTeam();
    List<OpponentDto> opponents = buildOpponents(myTeam);
    List<String> myRoster = getSquadNames(botTeamId);

    // 3. For each live lot, get advice and bid if warranted
    for (AuctionDto auction : live) {
        double myPurse = myTeam.getPurse() / 10_000_000.0;
        AdvisorAdviceResponse advice = advisorClient.getAdvice(
            auction.getPlayerName(), myPurse, myRoster, opponents, "SquadCompletionBidder"
        );
        long ceiling = (long)(advice.getRecommendedBid() * 10_000_000);
        long nextBid = nextBidAmount(auction.getBidAmount() != null ? auction.getBidAmount() : 0L);
        if (nextBid <= ceiling && nextBid <= myTeam.getPurse()) {
            auctionService.placeBid(auction.getId(), botTeamId, nextBid);
        }
    }
}
```

Add `@EnableScheduling` to the main application class.

### 3. Inject post-sale event in `AuctionServiceImpl.closeAuction()`

At the end of `closeAuction()`, after squad persistence and purse deduction:
```java
// fire-and-forget — don't let advisor failures crash the auction
try {
    advisorClient.notifySold(
        auction.getPlayerName(),
        auction.getBidAmount() / 10_000_000.0,
        auction.getCurrentWinningTeamName(),
        auction.getFantasyTourneyId()
    );
} catch (Exception e) {
    log.warn("Advisor sold notification failed: {}", e.getMessage());
}
```

---

## What to build — Advisor side (this repo)

**✅ All advisor-side changes are complete.** The following endpoints are live on the advisor server:

| Endpoint | Status | Purpose |
|---|---|---|
| `POST /api/advice` | ✅ done | Get bid ceiling (now accepts `tourney_id`) |
| `POST /api/advice/sold` | ✅ done | Record a completed sale into `MarketAnalyzer` |
| `DELETE /api/session/{tourney_id}` | ✅ done | Clear session at tourney end |

Test with:
```bash
# Verify sold event and market mode response
curl -X POST http://localhost:8000/api/advice/sold \
  -H "Content-Type: application/json" \
  -d '{"player_name":"Ruturaj Gaikwad","price":18.0,"buyer":"Alice FC","tourney_id":"GroupA_IPL2026"}'
# → {"ok":true, "known":true, "market_mode":"fair", "sales_count":1}

# Clean up at auction end
curl -X DELETE http://localhost:8000/api/session/GroupA_IPL2026
# → {"ok":true, "existed":true}
```

For reference, the implementation details are in `server.py`.

---

## Docker — running the advisor anywhere

[Dockerfile](Dockerfile) and [.dockerignore](.dockerignore) are committed at the repo root.

```bash
# Build once
docker build -t auction-advisor .

# Run (volume-mount data files so pool/params don't get baked into the image)
docker run -p 8000:8000 \
  -v $(pwd)/player_pool.json:/app/player_pool.json \
  -v $(pwd)/evolved_params.json:/app/evolved_params.json \
  -v $(pwd)/player_master.json:/app/player_master.json \
  auction-advisor

# For dev — run in background and expose via ngrok
docker run -d -p 8000:8000 \
  -v $(pwd)/player_pool.json:/app/player_pool.json \
  -v $(pwd)/evolved_params.json:/app/evolved_params.json \
  -v $(pwd)/player_master.json:/app/player_master.json \
  auction-advisor
ngrok http 8000   # share the URL with your friend as advisor.base-url
```

Volume mounts mean you can re-evolve params or rebuild the pool locally and the running container picks them up on the next request — no image rebuild needed.

---

## Purse conversion reference

| Context | Unit | Value for ₹120 Cr |
|---|---|---|
| Platform `FantasyTeam.purse` | paisa integers | `1_200_000_000` |
| Advisor `my_purse` | ₹ Crore | `120.0` |
| Conversion | `purse / 10_000_000.0` | — |

---

## Data model compatibility

| Platform | Advisor | Notes |
|---|---|---|
| `AuctionDto.playerName` = `"Virat Kohli"` | `auction_name` = `"Virat Kohli"` | **Direct match** — no mapping |
| `Player.id` = `"virat_kohli"` (slug) | internal only | Not used in advisor calls |
| `Player.type` = `BATSMAN` | `role` = `BAT` | Not used in current contract |
| `FantasyTeamDto.purse` paisa | `my_purse` ₹ Cr | ÷ 10,000,000 |
| Squad names from `/api/auctions/squad/{t}/{team}` | `my_roster: []` | Direct string list |
| All opponents from `/api/fantasy-tourneys/{id}/fantasy-teams` | `opponents: []` | Same purse conversion |

---

## Sequence diagram — one auction lot

```
Admin opens lot    →   POST /api/auctions   →   status: LIVE
                                                    │
BotBidService polls (every 3s)                      │
  GET .../live                                      ▼
  ← AuctionDto { playerName, bidAmount, status }
  POST /api/advice  ──────────────────────►  AdvisorService
  ← { recommended_bid: 18.5 }
  nextBid = nextBidAmount(bidAmount)
  if nextBid ≤ 18.5 Cr && nextBid ≤ purse:
    placeBid(auctionId, botTeamId, nextBid)
                                                    │
  [human bids raise price above ceiling]            │
  [bot polls again, nextBid > ceiling → skip]       │
                                                    │
Admin closes lot   →   POST /api/auctions/{id}/close
  AuctionServiceImpl.closeAuction()
    → persist squad entry
    → deduct purse
    → notify advisor: POST /api/advice/sold  ──────► MarketAnalyzer.record_sale()
```

---

## Files to create / modify

### Spring Boot project

| File | Action |
|---|---|
| `src/main/resources/application.properties` | Add `advisor.base-url`, `bot.*` properties |
| `src/main/java/.../advisor/AuctionAdvisorClient.java` | Create — RestTemplate HTTP client |
| `src/main/java/.../advisor/AdvisorAdviceRequest.java` | Create — request DTO |
| `src/main/java/.../advisor/AdvisorAdviceResponse.java` | Create — response DTO |
| `src/main/java/.../advisor/AdvisorOpponentDto.java` | Create — opponent DTO |
| `src/main/java/.../advisor/AdvisorSoldEvent.java` | Create — sold event DTO |
| `src/main/java/.../bot/BotBidService.java` | Create — polling + bidding loop |
| `src/main/java/.../AppConfig.java` (or existing config) | Add `@Bean RestTemplate` + `@EnableScheduling` |
| `src/main/java/.../auction/AuctionServiceImpl.java` | Modify — inject `AuctionAdvisorClient`, call `notifySold` in `closeAuction()` |

### Advisor (this repo)

| File | Action |
|---|---|
| `server.py` | Add `_sessions` store + `_get_session()` + `SoldEvent` model + `POST /api/advice/sold` + `DELETE /api/session/{id}` + `tourney_id` field on `AdviceRequest` |
| `Dockerfile` | Create — containerises the advisor for easy deployment anywhere |
| `.dockerignore` | Create — excludes notebooks, cache, venv from image |
