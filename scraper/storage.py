"""SQLite storage layer with deterministic SHA-256 deduplication, AI classification cache, and scrape audit logs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "reviews.db"
DEFAULT_PROPERTIES_PATH = Path(__file__).resolve().parent.parent / "config" / "properties.json"


def compute_review_hash(review: Dict[str, Any]) -> str:
    """Create a stable SHA-256 fingerprint to prevent duplicate reviews across incremental runs."""
    parts = [
        str(review.get("property_id", "")).strip().lower(),
        str(review.get("reviewer_name", "")).strip().lower(),
        str(review.get("review_date", "")).strip()[:10],
        f"{float(review.get('rating', 0.0)):.1f}",
        str(review.get("title", "")).strip().lower()[:120],
        str(review.get("positive_text", "")).strip().lower()[:160],
        str(review.get("negative_text", "")).strip().lower()[:160],
    ]
    payload = "|".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def validate_review(review: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate incoming review payload before persistence."""
    required = ["property_id", "rating", "review_date"]
    for field in required:
        if review.get(field) in (None, ""):
            return False, f"Missing required field: {field}"

    try:
        rating = float(review["rating"])
    except (TypeError, ValueError):
        return False, f"Invalid rating value: {review.get('rating')}"

    if not (0.0 <= rating <= 10.0):
        return False, f"Rating out of 0-10 range: {rating}"

    date_str = str(review["review_date"]).strip()[:10]
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return False, f"Invalid ISO date format (expected YYYY-MM-DD): {date_str}"

    return True, None


class ReviewStore:
    """SQLite-backed repository for properties, reviews, AI topic classifications, and scrape runs."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS properties (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    booking_title TEXT,
                    booking_url TEXT NOT NULL,
                    address TEXT,
                    booking_overall_rating REAL,
                    booking_total_reviews INTEGER,
                    wifi_rating REAL,
                    subscores_json TEXT,
                    operational_notes TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS reviews (
                    review_hash TEXT PRIMARY KEY,
                    property_id TEXT NOT NULL,
                    reviewer_name TEXT,
                    reviewer_country TEXT,
                    traveler_type TEXT,
                    room_type TEXT,
                    stay_nights INTEGER,
                    rating REAL NOT NULL,
                    review_date TEXT NOT NULL,
                    title TEXT,
                    positive_text TEXT,
                    negative_text TEXT,
                    source_platform TEXT DEFAULT 'Booking.com',
                    source_url TEXT,
                    scraped_at TEXT NOT NULL,
                    FOREIGN KEY (property_id) REFERENCES properties(id)
                );

                CREATE TABLE IF NOT EXISTS review_classifications (
                    review_hash TEXT PRIMARY KEY,
                    sentiment TEXT NOT NULL,
                    topics_json TEXT NOT NULL,
                    positive_topics_json TEXT NOT NULL,
                    negative_topics_json TEXT NOT NULL,
                    ai_summary TEXT,
                    classifier_source TEXT NOT NULL,
                    classified_at TEXT NOT NULL,
                    FOREIGN KEY (review_hash) REFERENCES reviews(review_hash) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_reviews_property_date
                    ON reviews(property_id, review_date DESC);

                CREATE INDEX IF NOT EXISTS idx_reviews_date
                    ON reviews(review_date DESC);

                CREATE TABLE IF NOT EXISTS scrape_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    status TEXT NOT NULL,
                    properties_attempted INTEGER DEFAULT 0,
                    reviews_fetched INTEGER DEFAULT 0,
                    reviews_inserted INTEGER DEFAULT 0,
                    duplicates_skipped INTEGER DEFAULT 0,
                    errors_json TEXT
                );
                """
            )

    def clear_reviews(self, property_id: Optional[str] = None) -> int:
        """Delete stored reviews and cached classifications to start from a clean slate."""
        with self._connect() as conn:
            if property_id and property_id != "all":
                conn.execute(
                    "DELETE FROM review_classifications WHERE review_hash IN "
                    "(SELECT review_hash FROM reviews WHERE property_id = ?)",
                    (property_id,),
                )
                cursor = conn.execute("DELETE FROM reviews WHERE property_id = ?", (property_id,))
            else:
                conn.execute("DELETE FROM review_classifications")
                cursor = conn.execute("DELETE FROM reviews")
            return int(cursor.rowcount)

    def save_classifications(self, items: Iterable[Dict[str, Any]]) -> int:
        """Cache LangChain + Gemini topic classifications in SQLite."""
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        with self._connect() as conn:
            for item in items:
                conn.execute(
                    """
                    INSERT INTO review_classifications (
                        review_hash, sentiment, topics_json, positive_topics_json,
                        negative_topics_json, ai_summary, classifier_source, classified_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(review_hash) DO UPDATE SET
                        sentiment=excluded.sentiment,
                        topics_json=excluded.topics_json,
                        positive_topics_json=excluded.positive_topics_json,
                        negative_topics_json=excluded.negative_topics_json,
                        ai_summary=excluded.ai_summary,
                        classifier_source=excluded.classifier_source,
                        classified_at=excluded.classified_at
                    """,
                    (
                        item["review_hash"],
                        item["sentiment"],
                        json.dumps(item.get("topics", [])),
                        json.dumps(item.get("positive_topics", [])),
                        json.dumps(item.get("negative_topics", [])),
                        item.get("ai_summary") or "",
                        item.get("classifier_source") or "gemini-langchain",
                        item.get("classified_at") or now,
                    ),
                )
                count += 1
        return count

    def get_classifications(self) -> Dict[str, Dict[str, Any]]:
        """Load cached AI classifications keyed by review_hash."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM review_classifications").fetchall()
        out: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            out[row["review_hash"]] = {
                "sentiment": row["sentiment"],
                "topics": json.loads(row["topics_json"] or "[]"),
                "positive_topics": json.loads(row["positive_topics_json"] or "[]"),
                "negative_topics": json.loads(row["negative_topics_json"] or "[]"),
                "ai_summary": row["ai_summary"] or "",
                "classifier_source": row["classifier_source"],
                "classified_at": row["classified_at"],
            }
        return out

    def sync_properties(self, properties: Iterable[Dict[str, Any]]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        count = 0
        with self._connect() as conn:
            for prop in properties:
                conn.execute(
                    """
                    INSERT INTO properties (
                        id, name, booking_title, booking_url, address,
                        booking_overall_rating, booking_total_reviews, wifi_rating,
                        subscores_json, operational_notes, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        booking_title=excluded.booking_title,
                        booking_url=excluded.booking_url,
                        address=excluded.address,
                        booking_overall_rating=excluded.booking_overall_rating,
                        booking_total_reviews=excluded.booking_total_reviews,
                        wifi_rating=excluded.wifi_rating,
                        subscores_json=excluded.subscores_json,
                        operational_notes=excluded.operational_notes,
                        updated_at=excluded.updated_at
                    """,
                    (
                        prop["id"],
                        prop["name"],
                        prop.get("booking_title"),
                        prop["booking_url"],
                        prop.get("address"),
                        prop.get("booking_overall_rating"),
                        prop.get("booking_total_reviews"),
                        prop.get("wifi_rating"),
                        json.dumps(prop.get("subscores", {})),
                        prop.get("operational_notes"),
                        now,
                    ),
                )
                count += 1
        return count

    def upsert_reviews(self, reviews: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Insert new reviews and skip exact duplicates based on deterministic hash."""
        now = datetime.now(timezone.utc).isoformat()
        inserted = 0
        duplicates = 0
        invalid = 0

        with self._connect() as conn:
            for raw in reviews:
                ok, _ = validate_review(raw)
                if not ok:
                    invalid += 1
                    continue

                review_hash = raw.get("review_hash") or compute_review_hash(raw)
                cursor = conn.execute(
                    """
                    INSERT OR IGNORE INTO reviews (
                        review_hash, property_id, reviewer_name, reviewer_country,
                        traveler_type, room_type, stay_nights, rating, review_date,
                        title, positive_text, negative_text, source_platform,
                        source_url, scraped_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        review_hash,
                        raw["property_id"],
                        raw.get("reviewer_name") or "Anonymous Guest",
                        raw.get("reviewer_country") or "",
                        raw.get("traveler_type") or "Solo traveller",
                        raw.get("room_type") or "",
                        raw.get("stay_nights"),
                        round(float(raw["rating"]), 1),
                        str(raw["review_date"]).strip()[:10],
                        (raw.get("title") or "").strip(),
                        (raw.get("positive_text") or "").strip(),
                        (raw.get("negative_text") or "").strip(),
                        raw.get("source_platform") or "Booking.com",
                        raw.get("source_url") or "",
                        raw.get("scraped_at") or now,
                    ),
                )
                if cursor.rowcount == 1:
                    inserted += 1
                else:
                    duplicates += 1

        return {"inserted": inserted, "duplicates": duplicates, "invalid": invalid}

    def get_existing_hashes(self, property_id: Optional[str] = None) -> set[str]:
        with self._connect() as conn:
            if property_id:
                rows = conn.execute(
                    "SELECT review_hash FROM reviews WHERE property_id = ?", (property_id,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT review_hash FROM reviews").fetchall()
        return {row["review_hash"] for row in rows}

    def list_properties(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM properties ORDER BY name").fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["subscores"] = json.loads(item.pop("subscores_json") or "{}")
            result.append(item)
        return result

    def list_reviews(
        self,
        property_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if property_id and property_id != "all":
            clauses.append("property_id = ?")
            params.append(property_id)
        if start_date:
            clauses.append("review_date >= ?")
            params.append(start_date[:10])
        if end_date:
            clauses.append("review_date <= ?")
            params.append(end_date[:10])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM reviews {where_sql} ORDER BY review_date DESC, review_hash ASC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def record_scrape_run(
        self,
        started_at: str,
        finished_at: str,
        status: str,
        properties_attempted: int,
        reviews_fetched: int,
        reviews_inserted: int,
        duplicates_skipped: int,
        errors: List[str],
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scrape_runs (
                    started_at, finished_at, status, properties_attempted,
                    reviews_fetched, reviews_inserted, duplicates_skipped, errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    started_at,
                    finished_at,
                    status,
                    properties_attempted,
                    reviews_fetched,
                    reviews_inserted,
                    duplicates_skipped,
                    json.dumps(errors),
                ),
            )
            return int(cursor.lastrowid)

    def get_latest_scrape_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scrape_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["errors"] = json.loads(item.pop("errors_json") or "[]")
            out.append(item)
        return out


def init_properties_only(
    db_path: Path | str = DEFAULT_DB_PATH,
    properties_path: Path | str = DEFAULT_PROPERTIES_PATH,
) -> int:
    """Sync configured properties into SQLite without inserting any sample reviews."""
    store = ReviewStore(db_path)
    props = json.loads(Path(properties_path).read_text(encoding="utf-8"))
    return store.sync_properties(props)
