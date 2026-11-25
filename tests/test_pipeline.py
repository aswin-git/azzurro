"""Automated unit and integration tests for scraper, storage, analytics, and FastAPI backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from analytics.insights import compute_weekly_comparison, enrich_review
from backend.app import create_app
from scraper.booking_scraper import (
    BookingReviewScraper,
    parse_booking_date,
    parse_booking_score,
    parse_graphql_review_cards,
)
from scraper.storage import ReviewStore, compute_review_hash, init_properties_only


SAMPLE_BOOKING_HTML = """
<div data-testid="review-card">
  <div data-testid="review-avatar"><div>Alex</div><span>Australia</span></div>
  <span data-testid="review-date">Reviewed: 22 October 2026</span>
  <div data-testid="review-score"><div>3.0</div></div>
  <h3 data-testid="review-title">Noisy aircon and dirty bathroom</h3>
  <div data-testid="review-positive-text">Great location near station</div>
  <div data-testid="review-negative-text">Bathroom had mould and AC was very loud</div>
</div>
"""

SAMPLE_GRAPHQL_PAYLOAD = {
    "data": {
        "reviewListFrontend": {
            "reviewsCount": 4310,
            "reviewCard": [
                {
                    "reviewUrl": "8895d10df583649c",
                    "reviewScore": 8.0,
                    "reviewedDate": 1760961260,
                    "guestDetails": {
                        "username": "Marco",
                        "countryName": "Italy",
                        "guestTypeTranslation": "Couple",
                    },
                    "textDetails": {
                        "title": "Great location near Darling Harbour",
                        "positiveText": "Friendly staff and convenient location",
                        "negativeText": "Shared bathroom could be cleaner",
                    },
                    "bookingDetails": {
                        "numNights": 3,
                        "roomType": {"name": "Mixed Pod 4 Beds"},
                    },
                }
            ],
        }
    }
}


class TestReviewPipeline(unittest.TestCase):
    def test_date_and_score_parsers(self) -> None:
        self.assertEqual(parse_booking_date("Reviewed: 22 October 2026"), "2026-10-22")
        self.assertEqual(parse_booking_date("Oct 6, 2026"), "2026-10-06")
        self.assertEqual(parse_booking_date(1760961260), "2025-10-20")
        self.assertEqual(parse_booking_score("8.5"), 8.5)
        self.assertEqual(parse_booking_score("Scored 6,0"), 6.0)
        self.assertIsNone(parse_booking_score("99.9"))

    def test_html_and_graphql_card_parsing(self) -> None:
        dom_cards = BookingReviewScraper.parse_review_cards(
            SAMPLE_BOOKING_HTML,
            property_id="potts-point",
            source_url="https://www.booking.com/hotel/au/venus-potts-point-sydney.html",
        )
        self.assertEqual(len(dom_cards), 1)
        self.assertEqual(dom_cards[0]["reviewer_name"], "Alex")

        gql_cards = parse_graphql_review_cards(
            SAMPLE_GRAPHQL_PAYLOAD,
            property_id="darling-harbour",
            source_url="https://www.booking.com/hotel/au/chateau-de-venus.html",
        )
        self.assertEqual(len(gql_cards), 1)
        self.assertEqual(gql_cards[0]["reviewer_name"], "Marco")
        self.assertEqual(gql_cards[0]["rating"], 8.0)

    def test_deduplication_clear_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewStore(Path(tmp) / "test.db")
            valid_review = {
                "property_id": "darling-harbour",
                "reviewer_name": "Lesley",
                "rating": 8.0,
                "review_date": "2026-10-21",
                "title": "Friendly staff",
                "positive_text": "Location is great",
                "negative_text": "Bathroom not very clean",
            }
            bad_review = {
                "property_id": "darling-harbour",
                "rating": 15.0,
                "review_date": "not-a-date",
            }
            first = store.upsert_reviews([valid_review, bad_review])
            self.assertEqual(first["inserted"], 1)
            self.assertEqual(first["invalid"], 1)

            second = store.upsert_reviews([valid_review])
            self.assertEqual(second["inserted"], 0)
            self.assertEqual(second["duplicates"], 1)
            self.assertEqual(compute_review_hash(valid_review), store.list_reviews()[0]["review_hash"])

            deleted = store.clear_reviews()
            self.assertEqual(deleted, 1)
            self.assertEqual(len(store.list_reviews()), 0)

    def test_topic_classification_and_weekly_delta(self) -> None:
        enriched = enrich_review(
            {
                "property_id": "potts-point",
                "rating": 2.0,
                "review_date": "2026-10-22",
                "title": "Dirty bathroom and loud AC",
                "positive_text": "Great location near Kings Cross station",
                "negative_text": "Bathroom had mould, check-in keypad broke, AC was noisy",
            }
        )
        self.assertEqual(enriched["sentiment"], "negative")
        self.assertIn("Cleanliness", enriched["negative_topics"])
        self.assertIn("Noise", enriched["negative_topics"])
        self.assertIn("Check-in experience", enriched["negative_topics"])
        self.assertIn("Location", enriched["positive_topics"])

        weekly = compute_weekly_comparison(
            [
                {"rating": 8.0, "review_date": "2026-10-24"},
                {"rating": 6.0, "review_date": "2026-10-21"},
                {"rating": 4.0, "review_date": "2026-10-15"},
                {"rating": 4.0, "review_date": "2026-10-14"},
            ],
            anchor_date="2026-10-25",
        )
        self.assertEqual(weekly["current_week_avg"], 7.0)
        self.assertEqual(weekly["previous_week_avg"], 4.0)
        self.assertEqual(weekly["delta"], 3.0)

    def test_fastapi_endpoints_clean_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "reviews.db"
            count = init_properties_only(db_path=db_path)
            self.assertEqual(count, 4)

            app = create_app(db_path=db_path)
            client = TestClient(app)

            health = client.get("/api/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["properties"], 4)
            self.assertEqual(health.json()["reviews"], 0)

            dash = client.get("/api/dashboard")
            self.assertEqual(dash.status_code, 200)
            payload = dash.json()
            self.assertEqual(len(payload["properties"]), 4)
            self.assertEqual(len(payload["topics"]), 8)
            self.assertEqual(len(payload["reviews"]), 0)

            clear_resp = client.delete("/api/reviews")
            self.assertEqual(clear_resp.status_code, 200)
            self.assertEqual(clear_resp.json()["deleted"], 0)



    def test_langchain_schema_and_classification_cache(self) -> None:
        from analytics.langchain_classifier import ReviewTopicClassification

        parsed = ReviewTopicClassification(
            sentiment="negative",
            positive_topics=["Location"],
            negative_topics=["Cleanliness", "Noise"],
            summary="Great location, but dirty bathroom and noisy aircon.",
        )
        self.assertEqual(parsed.sentiment, "negative")
        self.assertIn("Cleanliness", parsed.negative_topics)

        with tempfile.TemporaryDirectory() as tmp:
            store = ReviewStore(Path(tmp) / "cache.db")
            rev = {
                "property_id": "darling-harbour",
                "reviewer_name": "TestGuest",
                "rating": 3.0,
                "review_date": "2026-10-22",
                "title": "Mixed stay",
                "positive_text": "Close to CBD",
                "negative_text": "Needed deep clean",
            }
            store.upsert_reviews([rev])
            stored = store.list_reviews()[0]
            store.save_classifications(
                [
                    {
                        "review_hash": stored["review_hash"],
                        "sentiment": "negative",
                        "topics": ["Cleanliness", "Location"],
                        "positive_topics": ["Location"],
                        "negative_topics": ["Cleanliness"],
                        "ai_summary": "Close to CBD, but room needed deep cleaning.",
                        "classifier_source": "gemini-langchain (gemini-2.0-flash)",
                    }
                ]
            )
            ai_cache = store.get_classifications()
            enriched = enrich_review(stored, ai_cache=ai_cache)
            self.assertEqual(enriched["classifier_source"], "gemini-langchain (gemini-2.0-flash)")
            self.assertEqual(enriched["ai_summary"], "Close to CBD, but room needed deep cleaning.")

if __name__ == "__main__":
    unittest.main()
