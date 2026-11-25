"""Resilient Booking.com review collector supporting full-history pagination (--all-pages / max_pages=0).

Usage:
    # Fetch ALL available review pages for all 4 properties
    python -m scraper.booking_scraper --all --all-pages

    # Fetch ALL available review pages for one property
    python -m scraper.booking_scraper --property darling-harbour --all-pages

    # Clear existing reviews from SQLite before a fresh full scrape
    python -m scraper.booking_scraper --clear-db --all --all-pages
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

from scraper.storage import (
    DEFAULT_DB_PATH,
    DEFAULT_PROPERTIES_PATH,
    ReviewStore,
    compute_review_hash,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("azzurro.scraper")

USER_AGENTS = [
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
    ),
    (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    ),
]

GRAPHQL_REVIEW_QUERY = """
query ReviewList($input: ReviewListFrontendInput!) {
  reviewListFrontend(input: $input) {
    reviewsCount
    reviewCard {
      reviewUrl
      reviewScore
      reviewedDate
      isApproved
      guestDetails {
        username
        countryName
        guestTypeTranslation
        anonymous
      }
      textDetails {
        title
        positiveText
        negativeText
        lang
      }
      bookingDetails {
        numNights
        roomType {
          name
        }
      }
    }
  }
}
""".strip()

DATE_LINE_RE = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+20\d{2}$"
)
SCORE_LINE_RE = re.compile(r"^(10(?:\.0)?|[1-9](?:\.\d)?)$")


def parse_booking_date(raw_date: Any) -> Optional[str]:
    """Normalize Unix timestamps or date strings into ISO YYYY-MM-DD."""
    if raw_date in (None, ""):
        return None
    if isinstance(raw_date, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw_date), tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            return None

    cleaned = re.sub(
        r"^(Reviewed|Posted on|Date of stay):\s*", "", str(raw_date).strip(), flags=re.I
    ).strip()

    if cleaned.isdigit() and len(cleaned) >= 9:
        try:
            return datetime.fromtimestamp(float(cleaned), tz=timezone.utc).strftime("%Y-%m-%d")
        except (OSError, OverflowError, ValueError):
            pass

    formats = [
        "%Y-%m-%d",
        "%d %B %Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %b %Y",
        "%B %Y",
        "%b %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def parse_booking_score(raw_score: Any) -> Optional[float]:
    """Extract numeric 0-10 score from float, int, or strings like '8.0', 'Scored 6.5', '10/10'."""
    if raw_score is None:
        return None
    if isinstance(raw_score, (int, float)):
        val = float(raw_score)
        return round(val, 1) if 0.0 <= val <= 10.0 else None
    match = re.search(r"(\d+(?:[.,]\d+)?)", str(raw_score))
    if not match:
        return None
    val = float(match.group(1).replace(",", "."))
    return round(val, 1) if 0.0 <= val <= 10.0 else None


def parse_graphql_review_cards(
    payload: Dict[str, Any],
    property_id: str,
    source_url: str,
) -> List[Dict[str, Any]]:
    """Parse Booking.com `dml/graphql` `reviewListFrontend.reviewCard` JSON response."""
    root = payload.get("data", payload)
    rlf = root.get("reviewListFrontend") or {}
    cards = rlf.get("reviewCard") or []
    now_iso = datetime.now(timezone.utc).isoformat()
    parsed: List[Dict[str, Any]] = []

    for card in cards:
        if not isinstance(card, dict):
            continue
        rating = parse_booking_score(card.get("reviewScore"))
        review_date = parse_booking_date(card.get("reviewedDate"))
        if rating is None or not review_date:
            continue

        guest = card.get("guestDetails") or {}
        text = card.get("textDetails") or {}
        booking = card.get("bookingDetails") or {}
        room_info = booking.get("roomType") or {}

        record: Dict[str, Any] = {
            "property_id": property_id,
            "reviewer_name": guest.get("username") or "Anonymous Guest",
            "reviewer_country": guest.get("countryName") or "",
            "traveler_type": guest.get("guestTypeTranslation") or "Solo traveller",
            "room_type": room_info.get("name") or "",
            "stay_nights": booking.get("numNights"),
            "rating": rating,
            "review_date": review_date,
            "title": (text.get("title") or "").strip(),
            "positive_text": (text.get("positiveText") or "").strip(),
            "negative_text": (text.get("negativeText") or "").strip(),
            "source_platform": "Booking.com",
            "source_url": source_url,
            "scraped_at": now_iso,
        }
        record["review_hash"] = compute_review_hash(record)
        parsed.append(record)

    return parsed


class BookingReviewScraper:
    """Multi-strategy Booking.com review collector supporting unlimited pagination (`max_pages=0`)."""

    GRAPHQL_ENDPOINT = "https://www.booking.com/dml/graphql?lang=en-gb"
    SAFETY_MAX_PAGES = 500  # Up to 5,000 reviews per property

    def __init__(
        self,
        store: ReviewStore,
        timeout: int = 25,
        max_retries: int = 3,
        min_delay: float = 1.0,
        max_delay: float = 2.2,
        prefer_playwright: bool = True,
    ) -> None:
        self.store = store
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.prefer_playwright = prefer_playwright

    def _headers(self, referer: str = "https://www.booking.com/") -> Dict[str, str]:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "Referer": referer,
        }

    def _http_get_with_retry(self, client: httpx.Client, url: str, referer: str) -> httpx.Response:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.get(url, headers=self._headers(referer), follow_redirects=True)
                if resp.status_code == 200:
                    return resp
                if resp.status_code == 202:
                    raise RuntimeError(
                        "HTTP 202 AWS WAF JS challenge received (requires Playwright Chromium to solve)."
                    )
                if resp.status_code in (400, 401, 403, 404, 410):
                    resp.raise_for_status()
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = (2 ** attempt) + random.uniform(0.4, 1.2)
                    logger.warning(
                        "HTTP %s for %s (attempt %d/%d). Retrying in %.1fs...",
                        resp.status_code,
                        url,
                        attempt,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
            except (httpx.HTTPStatusError, RuntimeError):
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    wait = (2 ** attempt) + random.uniform(0.4, 1.2)
                    logger.warning(
                        "Network error (%s) on attempt %d/%d. Retrying in %.1fs...",
                        exc,
                        attempt,
                        self.max_retries,
                        wait,
                    )
                    time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts: {last_exc}")

    @staticmethod
    def parse_review_cards(html: str, property_id: str, source_url: str) -> List[Dict[str, Any]]:
        """Parse Booking.com DOM review cards (`data-testid='review-card'` or legacy markup)."""
        soup = BeautifulSoup(html, "lxml")
        cards = soup.select(
            "[data-testid='review-card'], "
            "li.review_list_new_item_block, "
            "div.c-review-block"
        )
        parsed: List[Dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for card in cards:
            name_el = card.select_one(
                "[data-testid='review-avatar'] div, .bui-avatar-block__title, .reviewer-name"
            )
            country_el = card.select_one(
                "[data-testid='review-avatar'] span, .bui-avatar-block__subtitle"
            )
            date_el = card.select_one("[data-testid='review-date'], .c-review-block__date")
            score_el = card.select_one("[data-testid='review-score'] div, .bui-review-score__badge")
            title_el = card.select_one("[data-testid='review-title'], .c-review-block__title")
            pos_el = card.select_one(
                "[data-testid='review-positive-text'], .c-review__row--positive .c-review__body"
            )
            neg_el = card.select_one(
                "[data-testid='review-negative-text'], .c-review__row--negative .c-review__body"
            )
            traveller_el = card.select_one(
                "[data-testid='review-traveler-type'], .review-panel-wide__traveller_type"
            )
            room_el = card.select_one(
                "[data-testid='review-room-name'], .c-review-block__room-info-row"
            )

            rating = parse_booking_score(score_el.get_text(" ", strip=True) if score_el else "")
            review_date = parse_booking_date(date_el.get_text(" ", strip=True) if date_el else "")
            if rating is None or not review_date:
                continue

            record: Dict[str, Any] = {
                "property_id": property_id,
                "reviewer_name": name_el.get_text(strip=True) if name_el else "Anonymous Guest",
                "reviewer_country": country_el.get_text(strip=True) if country_el else "",
                "traveler_type": traveller_el.get_text(strip=True) if traveller_el else "Solo traveller",
                "room_type": room_el.get_text(strip=True) if room_el else "",
                "rating": rating,
                "review_date": review_date,
                "title": title_el.get_text(strip=True) if title_el else "",
                "positive_text": pos_el.get_text(" ", strip=True) if pos_el else "",
                "negative_text": neg_el.get_text(" ", strip=True) if neg_el else "",
                "source_platform": "Booking.com",
                "source_url": source_url,
                "scraped_at": now_iso,
            }
            record["review_hash"] = compute_review_hash(record)
            parsed.append(record)

        return parsed

    @staticmethod
    def parse_planetofhotels_mirror(html: str, property_id: str, source_url: str) -> List[Dict[str, Any]]:
        """Structure-agnostic parser for verified Booking.com reviews on Planet of Hotels pages."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        now_iso = datetime.now(timezone.utc).isoformat()
        parsed: List[Dict[str, Any]] = []
        seen: set[str] = set()

        date_indices = [i for i, ln in enumerate(lines) if DATE_LINE_RE.match(ln)]
        for idx_pos, d_idx in enumerate(date_indices):
            next_boundary = (
                date_indices[idx_pos + 1] - 2
                if idx_pos + 1 < len(date_indices)
                else min(len(lines), d_idx + 12)
            )
            review_date = parse_booking_date(lines[d_idx])
            if not review_date:
                continue

            country_line = lines[d_idx - 1] if d_idx >= 1 else ""
            reviewer_name = lines[d_idx - 2] if d_idx >= 2 else "Verified Guest"
            if len(reviewer_name) > 50 or reviewer_name.lower() in (
                "positive aspects",
                "negative aspects",
                "100% verified reviews",
            ):
                reviewer_name = country_line if len(country_line) <= 40 else "Verified Guest"
                country_line = ""
            country = re.sub(r"^[a-z]{2}\s+", "", country_line).strip() if len(country_line) <= 35 else ""

            chunk = lines[d_idx + 1 : min(next_boundary, d_idx + 10)]
            if not chunk:
                continue

            traveler_type = "Solo traveller"
            if chunk and chunk[0].startswith("|"):
                traveler_type = chunk.pop(0).lstrip("|").strip()

            score_idx = next((i for i, ln in enumerate(chunk) if SCORE_LINE_RE.match(ln)), None)
            if score_idx is None:
                continue
            rating = parse_booking_score(chunk[score_idx])
            if rating is None:
                continue

            title = chunk[0] if score_idx > 0 else ""
            body_lines = [
                ln
                for ln in chunk[score_idx + 1 :]
                if ln.lower()
                not in ("frequently asked questions", "other hotels nearby", "negative aspects", "positive aspects")
            ]
            if not body_lines and not title:
                continue

            pos_text = body_lines[0] if len(body_lines) >= 2 else (body_lines[0] if rating >= 6.0 and body_lines else "")
            neg_text = (
                " ".join(body_lines[1:])
                if len(body_lines) >= 2
                else (body_lines[0] if rating < 6.0 and body_lines else "")
            )

            record: Dict[str, Any] = {
                "property_id": property_id,
                "reviewer_name": reviewer_name,
                "reviewer_country": country,
                "traveler_type": traveler_type,
                "room_type": "",
                "rating": rating,
                "review_date": review_date,
                "title": title[:140],
                "positive_text": pos_text,
                "negative_text": neg_text,
                "source_platform": "Booking.com (Mirror)",
                "source_url": source_url,
                "scraped_at": now_iso,
            }
            h = compute_review_hash(record)
            if h not in seen:
                seen.add(h)
                record["review_hash"] = h
                parsed.append(record)

        return parsed

    @staticmethod
    def parse_trip_mirror(html: str, property_id: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract verified guest reviews from Trip.com review HTML pages."""
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        now_iso = datetime.now(timezone.utc).isoformat()
        parsed: List[Dict[str, Any]] = []
        seen: set[str] = set()

        posted_re = re.compile(r"^Posted on\s+([A-Za-z]{3}\s+\d{1,2},\s+20\d{2})$")
        score_re = re.compile(r"^(\d+(?:\.\d+)?)/10$")

        posted_indices = [i for i, ln in enumerate(lines) if posted_re.match(ln)]
        for idx_pos, p_idx in enumerate(posted_indices):
            m = posted_re.match(lines[p_idx])
            if not m:
                continue
            review_date = parse_booking_date(m.group(1))
            if not review_date:
                continue

            reviewer_name = lines[p_idx - 1][:50] if p_idx >= 1 else "Guest User"
            end_bound = (
                posted_indices[idx_pos + 1] - 1
                if idx_pos + 1 < len(posted_indices)
                else min(len(lines), p_idx + 12)
            )
            chunk = lines[p_idx + 1 : min(end_bound, p_idx + 12)]

            score_idx = next((i for i, ln in enumerate(chunk) if score_re.match(ln)), None)
            if score_idx is None:
                continue
            rating = parse_booking_score(score_re.match(chunk[score_idx]).group(1))  # type: ignore[union-attr]
            if rating is None:
                continue

            body_lines = [
                ln
                for ln in chunk[score_idx + 1 :]
                if len(ln) > 15
                and not ln.startswith("Stay Date")
                and ln not in ("Show Original", "Show More", "Google Translated", "Outstanding", "Very good")
            ]
            if not body_lines:
                continue
            body = " ".join(body_lines)
            record: Dict[str, Any] = {
                "property_id": property_id,
                "reviewer_name": reviewer_name,
                "reviewer_country": "",
                "traveler_type": chunk[0] if score_idx > 0 else "Solo traveller",
                "room_type": "",
                "rating": rating,
                "review_date": review_date,
                "title": body[:100],
                "positive_text": body if rating >= 6.0 else "",
                "negative_text": body if rating < 6.0 else "",
                "source_platform": "Partner OTA (Trip.com)",
                "source_url": source_url,
                "scraped_at": now_iso,
            }
            h = compute_review_hash(record)
            if h not in seen:
                seen.add(h)
                record["review_hash"] = h
                parsed.append(record)

        return parsed

    def _scrape_via_playwright(
        self,
        prop: Dict[str, Any],
        max_pages: int,
        existing_hashes: set[str],
        incremental: bool,
    ) -> List[Dict[str, Any]]:
        """Strategy 1: Headless Chromium opens Booking.com #tab-reviews and paginates all pages (`max_pages=0` = unlimited)."""
        from playwright.sync_api import sync_playwright  # type: ignore[import-not-found]

        property_id = prop["id"]
        url = f"{prop['booking_url']}?force_referer=#tab-reviews"
        captured_reviews: List[Dict[str, Any]] = []
        seen_in_run: set[str] = set()
        page_limit = self.SAFETY_MAX_PAGES if max_pages <= 0 else max_pages

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                locale="en-GB",
                viewport={"width": 1366, "height": 900},
            )
            page = context.new_page()

            def handle_response(response: Any) -> None:
                try:
                    if "/dml/graphql" in response.url and response.status == 200:
                        data = response.json()
                        if "reviewListFrontend" in json.dumps(data):
                            cards = parse_graphql_review_cards(data, property_id, prop["booking_url"])
                            for c in cards:
                                if c["review_hash"] not in seen_in_run:
                                    seen_in_run.add(c["review_hash"])
                                    captured_reviews.append(c)
                except Exception:  # noqa: BLE001
                    pass

            page.on("response", handle_response)
            logger.info("[Playwright] Loading %s (page_limit=%s)", url, "ALL" if max_pages <= 0 else page_limit)
            page.goto(url, wait_until="domcontentloaded", timeout=40000)
            page.wait_for_timeout(3500)

            # Dismiss sign-in or cookie modals that can block the reviews tab trigger
            for dismiss_sel in [
                "button[aria-label='Dismiss sign-in info.']",
                "button#onetrust-accept-btn-handler",
            ]:
                try:
                    if page.locator(dismiss_sel).first.is_visible(timeout=1200):
                        page.locator(dismiss_sel).first.click()
                        page.wait_for_timeout(500)
                except Exception:  # noqa: BLE001
                    pass

            for selector in [
                "a[data-testid='reviews-tab-trigger']",
                "button[data-testid='reviews-tab-trigger']",
                "[data-testid='review-score-right-component']",
                "button:has-text('Read all reviews')",
                "#show_reviews_tab",
            ]:
                try:
                    if page.locator(selector).first.is_visible(timeout=1500):
                        page.locator(selector).first.click()
                        page.wait_for_timeout(2500)
                        break
                except Exception:  # noqa: BLE001
                    pass

            # Wait up to 10s for review cards or GraphQL payload to arrive (handles slower pages)
            if not captured_reviews:
                try:
                    page.wait_for_selector("[data-testid='review-card']", timeout=10000)
                except Exception:  # noqa: BLE001
                    page.mouse.wheel(0, 1800)
                    page.wait_for_timeout(3000)

            try:
                sort_select = page.locator("select[data-testid='reviews-sorter-component']").first
                if sort_select.is_visible(timeout=2000):
                    sort_select.select_option("NEWEST_FIRST")
                    page.wait_for_timeout(2500)
            except Exception:  # noqa: BLE001
                pass

            if not captured_reviews:
                dom_cards = self.parse_review_cards(page.content(), property_id, prop["booking_url"])
                for c in dom_cards:
                    if c["review_hash"] not in seen_in_run:
                        seen_in_run.add(c["review_hash"])
                        captured_reviews.append(c)

            # Save progress incrementally per page so long full-history scrapes never lose progress
            if captured_reviews:
                self.store.upsert_reviews(captured_reviews)

            for page_idx in range(2, page_limit + 1):
                before_count = len(captured_reviews)
                next_btn = page.locator("button[aria-label='Next page']").first
                try:
                    if not next_btn.is_visible(timeout=2500) or next_btn.is_disabled():
                        logger.info("[Playwright] Reached final page for %s at page %d", property_id, page_idx - 1)
                        break
                    next_btn.click()
                    page.wait_for_timeout(int(random.uniform(self.min_delay, self.max_delay) * 1000) + 1200)
                except Exception:  # noqa: BLE001
                    break

                if len(captured_reviews) == before_count:
                    dom_cards = self.parse_review_cards(page.content(), property_id, prop["booking_url"])
                    for c in dom_cards:
                        if c["review_hash"] not in seen_in_run:
                            seen_in_run.add(c["review_hash"])
                            captured_reviews.append(c)

                new_cards = captured_reviews[before_count:]
                if not new_cards:
                    break

                self.store.upsert_reviews(new_cards)
                if page_idx % 5 == 0:
                    logger.info("[Playwright] %s: %d reviews collected through page %d", property_id, len(captured_reviews), page_idx)

                if incremental and all(c["review_hash"] in existing_hashes for c in new_cards):
                    logger.info("[Playwright] Incremental stop for %s at page %d", property_id, page_idx)
                    break

            browser.close()

        return captured_reviews

    def _scrape_via_http_graphql(
        self,
        prop: Dict[str, Any],
        max_pages: int,
        existing_hashes: set[str],
        incremental: bool,
    ) -> List[Dict[str, Any]]:
        """Strategy 2: Direct HTTP fetch of Booking.com hotel page + GraphQL `ReviewList` POST."""
        property_id = prop["id"]
        collected: List[Dict[str, Any]] = []
        page_limit = self.SAFETY_MAX_PAGES if max_pages <= 0 else max_pages

        with httpx.Client(http2=False, timeout=self.timeout) as client:
            page_resp = self._http_get_with_retry(client, prop["booking_url"], referer="https://www.booking.com/")
            html = page_resp.text

            csrf_match = re.search(r"b_csrf_token\s*:\s*['\"]([^'\"]+)['\"]", html)
            hotel_id_match = re.search(r"b_hotel_id\s*:\s*['\"]?(\d+)['\"]?", html)
            if not csrf_match or not hotel_id_match:
                raise RuntimeError(
                    "Booking.com did not expose CSRF/hotel_id tokens to plain HTTP (install Playwright Chromium)."
                )

            csrf_token = csrf_match.group(1)
            hotel_id = int(hotel_id_match.group(1))

            for page_num in range(page_limit):
                gql_payload = {
                    "operationName": "ReviewList",
                    "variables": {
                        "input": {
                            "hotelId": hotel_id,
                            "ufi": -1603135,
                            "hotelCountryCode": prop.get("cc1", "au"),
                            "sorter": "NEWEST_FIRST",
                            "filters": {"text": ""},
                            "skip": page_num * 10,
                            "limit": 10,
                        }
                    },
                    "extensions": {},
                    "query": GRAPHQL_REVIEW_QUERY,
                }
                gql_resp = client.post(
                    self.GRAPHQL_ENDPOINT,
                    json=gql_payload,
                    headers={
                        **self._headers(prop["booking_url"]),
                        "Content-Type": "application/json",
                        "Origin": "https://www.booking.com",
                        "x-booking-csrf-token": csrf_token,
                    },
                )
                if gql_resp.status_code != 200:
                    raise RuntimeError(f"Booking.com GraphQL returned HTTP {gql_resp.status_code}")

                page_reviews = parse_graphql_review_cards(gql_resp.json(), property_id, prop["booking_url"])
                if not page_reviews:
                    break
                collected.extend(page_reviews)
                self.store.upsert_reviews(page_reviews)

                if incremental and all(r["review_hash"] in existing_hashes for r in page_reviews):
                    logger.info("[HTTP-GraphQL] Incremental stop for %s at page %d", property_id, page_num + 1)
                    break

                if page_num < page_limit - 1:
                    time.sleep(random.uniform(self.min_delay, self.max_delay))

        return collected

    def _scrape_via_mirror(self, prop: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Strategy 3: Fallback to public Booking.com review mirror when direct access hits AWS WAF."""
        mirror_url = prop.get("mirror_url")
        if not mirror_url:
            return []
        with httpx.Client(http2=False, timeout=self.timeout) as client:
            resp = self._http_get_with_retry(client, mirror_url, referer="https://www.google.com/")
            if "planetofhotels.com" in mirror_url:
                return self.parse_planetofhotels_mirror(resp.text, prop["id"], prop["booking_url"])
            if "trip.com" in mirror_url:
                return self.parse_trip_mirror(resp.text, prop["id"], prop["booking_url"])
            return self.parse_review_cards(resp.text, prop["id"], prop["booking_url"])

    def scrape_property(
        self,
        prop: Dict[str, Any],
        max_pages: int = 0,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """Collect reviews for one property (`max_pages=0` fetches all available pages)."""
        property_id = prop["id"]
        existing_hashes = self.store.get_existing_hashes(property_id) if incremental else set()
        before_total = len(self.store.get_existing_hashes(property_id))
        collected: List[Dict[str, Any]] = []
        strategy_used = "none"
        strategy_errors: List[str] = []

        if self.prefer_playwright:
            try:
                collected = self._scrape_via_playwright(prop, max_pages, existing_hashes, incremental)
                if collected:
                    strategy_used = "booking_playwright_graphql"
            except ImportError:
                strategy_errors.append("Playwright not installed (run: pip install playwright && playwright install chromium)")
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Playwright] %s failed: %s", property_id, exc)
                strategy_errors.append(f"Playwright: {exc}")

        if not collected:
            try:
                collected = self._scrape_via_http_graphql(prop, max_pages, existing_hashes, incremental)
                if collected:
                    strategy_used = "booking_http_graphql"
            except Exception as exc:  # noqa: BLE001
                logger.warning("[HTTP-GraphQL] %s failed: %s", property_id, exc)
                strategy_errors.append(f"HTTP-GraphQL: {exc}")

        if not collected and prop.get("mirror_url"):
            try:
                logger.info("[Mirror] Falling back to mirror for %s", property_id)
                collected = self._scrape_via_mirror(prop)
                if collected:
                    strategy_used = "public_review_mirror"
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Mirror] %s failed: %s", property_id, exc)
                strategy_errors.append(f"Mirror: {exc}")

        if not collected and strategy_errors:
            raise RuntimeError(" | ".join(strategy_errors))

        self.store.upsert_reviews(collected)
        after_total = len(self.store.get_existing_hashes(property_id))
        inserted = max(0, after_total - before_total)
        duplicates = max(0, len(collected) - inserted)
        return {
            "property_id": property_id,
            "strategy": strategy_used,
            "fetched": len(collected),
            "inserted": inserted,
            "duplicates": duplicates,
        }

    def scrape_all(
        self,
        properties: List[Dict[str, Any]],
        max_pages: int = 0,
        incremental: bool = True,
    ) -> Dict[str, Any]:
        """Run collection across all properties with per-property fault isolation."""
        started_at = datetime.now(timezone.utc).isoformat()
        total_fetched = 0
        total_inserted = 0
        total_duplicates = 0
        errors: List[str] = []
        per_property: List[Dict[str, Any]] = []

        for prop in properties:
            try:
                res = self.scrape_property(prop, max_pages=max_pages, incremental=incremental)
                per_property.append(res)
                total_fetched += res["fetched"]
                total_inserted += res["inserted"]
                total_duplicates += res["duplicates"]
            except Exception as exc:  # noqa: BLE001
                err_msg = f"{prop['id']}: {exc}"
                logger.error("Scrape failed for %s", err_msg)
                errors.append(err_msg)
                per_property.append({"property_id": prop["id"], "error": str(exc)})

        finished_at = datetime.now(timezone.utc).isoformat()
        status = "success" if not errors else ("partial_success" if total_fetched > 0 else "failed")
        run_id = self.store.record_scrape_run(
            started_at=started_at,
            finished_at=finished_at,
            status=status,
            properties_attempted=len(properties),
            reviews_fetched=total_fetched,
            reviews_inserted=total_inserted,
            duplicates_skipped=total_duplicates,
            errors=errors,
        )
        return {
            "run_id": run_id,
            "status": status,
            "fetched": total_fetched,
            "inserted": total_inserted,
            "duplicates": total_duplicates,
            "errors": errors,
            "properties": per_property,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Azzurro Hotels Booking.com Review Collector")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="Path to SQLite database")
    parser.add_argument("--properties", default=str(DEFAULT_PROPERTIES_PATH), help="Path to properties.json")
    parser.add_argument("--property", default=None, help="Single property ID to scrape")
    parser.add_argument("--all", action="store_true", help="Scrape all configured properties")
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max pages per property (0 = fetch ALL available pages; default: 0)",
    )
    parser.add_argument("--all-pages", action="store_true", help="Fetch ALL available review pages (sets --max-pages 0)")
    parser.add_argument("--full-refresh", action="store_true", help="Disable incremental early-stopping")
    parser.add_argument("--no-playwright", action="store_true", help="Skip Playwright and use HTTP/mirror strategies")
    parser.add_argument("--clear-db", action="store_true", help="Delete all existing reviews from SQLite before scraping")
    args = parser.parse_args()

    store = ReviewStore(args.db)
    props = json.loads(Path(args.properties).read_text(encoding="utf-8"))
    store.sync_properties(props)

    if args.clear_db:
        deleted = store.clear_reviews(args.property)
        logger.info("Cleared %d existing reviews from %s", deleted, args.db)

    if args.property:
        targets = [p for p in props if p["id"] == args.property]
        if not targets:
            raise SystemExit(f"Unknown property id: {args.property}")
    elif args.all:
        targets = props
    else:
        if args.clear_db:
            print(json.dumps({"cleared_reviews": deleted}, indent=2))
            return
        parser.print_help()
        return

    max_pages = 0 if args.all_pages else args.max_pages
    scraper = BookingReviewScraper(store, prefer_playwright=not args.no_playwright)
    summary = scraper.scrape_all(targets, max_pages=max_pages, incremental=not args.full_refresh)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
