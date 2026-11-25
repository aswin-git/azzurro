# Azzurro Hotels — Booking.com Review Insights Dashboard

A full-stack review collection, operational topic analysis, and reporting dashboard built for non-technical Azzurro Hotels operations staff across four Sydney properties:

1. **Olympic Hotel Paddington** — [Booking.com Listing](https://www.booking.com/hotel/au/olympic-paddington.html) (308 Moore Park Road, Paddington)
2. **Potts Point** — [Booking.com Listing](https://www.booking.com/hotel/au/venus-potts-point-sydney.html) (141-143 Victoria Street, Potts Point)
3. **Central Sydney** — [Booking.com Listing](https://www.booking.com/hotel/au/venus-surry-hills.html) (90 Wentworth Avenue, Surry Hills)
4. **Darling Harbour** — [Booking.com Listing](https://www.booking.com/hotel/au/chateau-de-venus.html) (22 Allen Street, Pyrmont / Sydney CBD)

---

## 1. Setup & Installation

**Prerequisites:** Python 3.10+ (tested on Python 3.11). No external database server or API keys required.

```bash
# 1. Unzip and enter the project directory
cd azzurro-review-insights

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install Chromium for Playwright (enables direct Booking.com GraphQL review scraping)
playwright install chromium

# 5. Seed the local SQLite database with the collected sample dataset
python -m scraper.booking_scraper --seed-sample
```

---

## 2. How to Run the Scraper, Backend & Frontend

### Run the Operations Dashboard (with One-Click Live Scrape Button)
```bash
uvicorn backend.app:app --reload --port 8000
```
Open **http://localhost:8000** in your browser.
- Click **"Update Reviews"** in the top bar at any time to scrape fresh reviews directly from Booking.com (either for the currently selected property or all 4 properties), deduplicate them into SQLite, and refresh all KPIs, charts, insights, and the review feed automatically.
- You can also open `frontend/index.html` directly without starting a server to explore the embedded sample dataset offline.

### Run the Booking.com Review Scraper from CLI
```bash
# Incremental update across all 4 properties (stops once previously seen reviews are reached)
python -m scraper.booking_scraper --all --max-pages 3

# Scrape a single property
python -m scraper.booking_scraper --property darling-harbour --max-pages 5

# Full refresh (re-scan pages without incremental early stop; duplicates still safely skipped)
python -m scraper.booking_scraper --all --max-pages 5 --full-refresh

# Run without Playwright (uses direct HTTP GraphQL + public review mirror fallback)
python -m scraper.booking_scraper --property darling-harbour --no-playwright
```

### Run Automated Tests
```bash
python -m unittest discover -s tests -v
```

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Booking.com Public Pages                          │
│         (4 Azzurro Hotels Sydney properties in config/properties.json)  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │ 3-Tier Collector:
                                    │ 1. Playwright + dml/graphql intercept
                                    │ 2. Direct HTTP/2 GraphQL (CSRF + cookies)
                                    │ 3. Public review mirror HTML fallback
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    scraper/booking_scraper.py                           │
│  • Parses Booking.com reviewListFrontend.reviewCard JSON + DOM fallback │
│  • Fast-fail on 404, exponential backoff on 429/5xx, polite jitter      │
│  • Incremental early-stop & per-property fault isolation                │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       scraper/storage.py                                │
│  • SQLite (data/reviews.db) + JSON export/import                        │
│  • Deterministic SHA-256 review_hash primary key (zero duplicates)      │
│  • Schema validation (0-10 rating bounds, ISO date check)               │
│  • scrape_runs audit table (status, counts, error logs)                 │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       analytics/insights.py                             │
│  • Sentiment bucketing: Positive (≥7.0), Neutral (5.0–6.9), Neg (<5.0)  │
│  • 8-topic operational classifier (separate positive/negative tagging)  │
│  • Current Week vs. Previous Week comparison + monthly trends           │
│  • Plain-English operational insight generator                          │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                 backend/app.py  +  frontend/index.html                  │
│  • FastAPI REST API + POST /api/scrape + CSV export                     │
│  • Operations Dashboard with one-click "Update Reviews" scrape modal    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Review Collection Method & Reliability Controls

Booking.com does not expose a public guest-reviews API and deprecated its legacy `/reviewlist.html` HTML endpoint (which now returns HTTP 404). Modern Booking.com pages load guest reviews client-side via `POST https://www.booking.com/dml/graphql?lang=en-gb` (`reviewListFrontend.reviewCard`) protected by session cookies, `x-booking-csrf-token`, and AWS WAF.

Our collector implements a **3-tier resilient pipeline**:
1. **Tier 1 — Playwright + Booking.com GraphQL Interception (Primary)**: Launches headless Chromium, opens `<booking_url>?force_referer=#tab-reviews`, switches the sorter to `NEWEST_FIRST`, intercepts Booking.com's own `dml/graphql` `reviewListFrontend` JSON responses, and clicks **Next page** up to `--max-pages` (with DOM `[data-testid='review-card']` fallback).
2. **Tier 2 — Direct HTTP GraphQL (Lightweight Fallback)**: If Playwright is not installed, fetches the property HTML via `httpx`, extracts `b_csrf_token` and `b_hotel_id`, and sends `ReviewList` GraphQL POST requests directly.
3. **Tier 3 — Public Booking.com Mirror Fallback**: If direct HTTP receives a JS/WAF challenge and Playwright is unavailable, falls back to the configured `mirror_url` in `config/properties.json`.
4. **Zero-Duplicate Incremental Sync & Fault Isolation**: Every review is hashed via `SHA-256(property_id | reviewer | date | rating | title | pos_snippet | neg_snippet)[:24]` as the SQLite `PRIMARY KEY` (`INSERT OR IGNORE`). Per-property fault isolation ensures a failure on one property does not block the other three, and every run is recorded in `scrape_runs`.

---

## 5. Review Insights Methodology

Reviews are classified into the **8 core operational topics** (`Cleanliness`, `Check-in experience`, `Staff or receptionist behaviour`, `Noise`, `Facilities`, `Location`, `Room condition`, `Value for money`) using a transparent, deterministic regex/phrase lexicon (`analytics/insights.py`) with section-aware polarity (`positive_text` vs. `negative_text`).

---

## 6. Known Limitations & Assumptions

1. **Booking.com Anti-Bot / WAF Protection**: Headless browser scraping via Playwright solves Booking.com's JS/GraphQL loading locally, though high-frequency runs from datacenter IPs may trigger CAPTCHAs. Run once daily with polite delays (`1.2s–2.8s`).
2. **Sample Data Coverage**: `data/sample_reviews.json` contains **76 real guest reviews** collected across all four properties alongside each property's live Booking.com overall score (`6.3`–`6.6/10`).
3. **Weekly Comparison Window**: Anchors automatically to the latest active 7-day window containing reviews when viewing a static snapshot, and equals the trailing 7 calendar days once daily scraping runs.
4. **Security**: No passwords, cookies, API keys, or personal guest contact credentials are stored or committed anywhere in this repository.
# azzurro
