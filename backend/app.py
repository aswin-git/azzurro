"""FastAPI backend serving the Azzurro Hotels Booking.com Review Insights Dashboard."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from analytics.insights import build_dashboard_summary
from analytics.langchain_classifier import (
    GeminiLangChainClassifier,
    get_gemini_model_name,
    is_gemini_configured,
)
from scraper.booking_scraper import BookingReviewScraper
from scraper.storage import (
    DEFAULT_DB_PATH,
    DEFAULT_PROPERTIES_PATH,
    ReviewStore,
    init_properties_only,
)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_HTML = BASE_DIR / "frontend" / "index.html"


def create_app(
    db_path: Path | str = DEFAULT_DB_PATH,
    properties_path: Path | str = DEFAULT_PROPERTIES_PATH,
) -> FastAPI:
    app = FastAPI(
        title="Azzurro Hotels Booking.com Review Insights API",
        version="1.2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = ReviewStore(db_path)
    if not store.list_properties():
        init_properties_only(db_path, properties_path)

    @app.get("/", include_in_schema=False)
    def serve_frontend() -> FileResponse:
        if not FRONTEND_HTML.exists():
            raise HTTPException(status_code=404, detail="Frontend index.html not found")
        return FileResponse(FRONTEND_HTML)

    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "properties": len(store.list_properties()),
            "reviews": len(store.list_reviews()),
            "gemini_configured": is_gemini_configured(),
            "gemini_model": get_gemini_model_name(),
        }

    @app.get("/api/properties")
    def list_properties() -> List[Dict[str, Any]]:
        return store.list_properties()

    @app.get("/api/dashboard")
    def get_dashboard(
        property_id: Optional[str] = Query(default=None),
        start_date: Optional[str] = Query(default=None),
        end_date: Optional[str] = Query(default=None),
        sentiment: Optional[str] = Query(default=None),
        topic: Optional[str] = Query(default=None),
        search: Optional[str] = Query(default=None),
        anchor_date: Optional[str] = Query(default=None),
    ) -> Dict[str, Any]:
        properties = store.list_properties()
        reviews = store.list_reviews(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
        )
        ai_cache = store.get_classifications()
        summary = build_dashboard_summary(
            reviews, properties, anchor_date=anchor_date, ai_cache=ai_cache
        )

        filtered = summary["reviews"]
        if sentiment and sentiment != "all":
            filtered = [r for r in filtered if r["sentiment"] == sentiment]
        if topic and topic != "all":
            filtered = [r for r in filtered if topic in r["topics"]]
        if search:
            q = search.strip().lower()
            filtered = [
                r
                for r in filtered
                if q in (r.get("title") or "").lower()
                or q in (r.get("positive_text") or "").lower()
                or q in (r.get("negative_text") or "").lower()
                or q in (r.get("reviewer_name") or "").lower()
            ]
        summary["filtered_reviews"] = filtered
        summary["scrape_runs"] = store.get_latest_scrape_runs(limit=5)
        summary["ai_status"] = {
            "gemini_configured": is_gemini_configured(),
            "model": get_gemini_model_name(),
            "classified_count": summary["kpis"]["ai_classified_count"],
            "total_reviews": summary["kpis"]["total_reviews"],
        }
        return summary

    @app.post("/api/classify")
    def classify_reviews(
        property_id: Optional[str] = Query(default=None),
        max_reviews: int = Query(default=50, ge=1, le=500),
        force: bool = Query(default=False),
    ) -> Dict[str, Any]:
        """Classify unclassified reviews using LangChain + Gemini API and cache in SQLite."""
        reviews = store.list_reviews(property_id=property_id)
        existing = store.get_classifications() if not force else {}
        pending = [r for r in reviews if r["review_hash"] not in existing]

        if not pending:
            return {
                "status": "up_to_date",
                "classified_count": 0,
                "remaining_unclassified": 0,
                "model": get_gemini_model_name(),
                "message": "All reviews are already classified with Gemini + LangChain.",
            }

        try:
            classifier = GeminiLangChainClassifier()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        result = classifier.classify_batch(
            pending,
            max_reviews=max_reviews,
            on_chunk_done=store.save_classifications,
        )
        saved = result["classified_count"]
        remaining = max(0, len(pending) - saved)
        return {
            "status": "success" if result["error_count"] == 0 else "partial_success",
            "classified_count": saved,
            "error_count": result["error_count"],
            "errors": result["errors"],
            "remaining_unclassified": remaining,
            "model": result["model"],
        }

    @app.post("/api/scrape")
    def trigger_scrape(
        property_id: Optional[str] = Query(default=None),
        max_pages: int = Query(default=0, ge=0, le=500),
        full_refresh: bool = Query(default=False),
        auto_classify: bool = Query(default=True),
    ) -> Dict[str, Any]:
        props = json.loads(Path(properties_path).read_text(encoding="utf-8"))
        store.sync_properties(props)
        targets = [p for p in props if p["id"] == property_id] if property_id else props
        if not targets:
            raise HTTPException(status_code=404, detail=f"Unknown property_id: {property_id}")
        scraper = BookingReviewScraper(store)
        scrape_res = scraper.scrape_all(targets, max_pages=max_pages, incremental=not full_refresh)

        # Automatically classify newly scraped reviews with Gemini + LangChain if API key is configured
        if auto_classify and is_gemini_configured() and scrape_res.get("inserted", 0) > 0:
            try:
                reviews = store.list_reviews(property_id=property_id)
                existing = store.get_classifications()
                pending = [r for r in reviews if r["review_hash"] not in existing]
                if pending:
                    classifier = GeminiLangChainClassifier()
                    cls_res = classifier.classify_batch(pending, max_reviews=50)
                    saved = store.save_classifications(cls_res["classified"])
                    scrape_res["ai_classified"] = saved
            except Exception as exc:  # noqa: BLE001
                scrape_res["ai_classify_error"] = str(exc)

        return scrape_res

    @app.delete("/api/reviews")
    def clear_reviews(property_id: Optional[str] = Query(default=None)) -> Dict[str, int]:
        deleted = store.clear_reviews(property_id)
        return {"deleted": deleted}

    @app.get("/api/scrape-runs")
    def get_scrape_runs(limit: int = Query(default=10, ge=1, le=50)) -> List[Dict[str, Any]]:
        return store.get_latest_scrape_runs(limit=limit)

    @app.get("/api/export.csv")
    def export_csv(
        property_id: Optional[str] = Query(default=None),
        start_date: Optional[str] = Query(default=None),
        end_date: Optional[str] = Query(default=None),
    ) -> StreamingResponse:
        properties = store.list_properties()
        reviews = store.list_reviews(
            property_id=property_id,
            start_date=start_date,
            end_date=end_date,
        )
        summary = build_dashboard_summary(
            reviews, properties, ai_cache=store.get_classifications()
        )

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "review_hash",
                "property_id",
                "review_date",
                "rating",
                "sentiment",
                "topics",
                "ai_summary",
                "classifier_source",
                "reviewer_name",
                "reviewer_country",
                "traveler_type",
                "title",
                "positive_text",
                "negative_text",
                "source_platform",
            ]
        )
        for r in summary["reviews"]:
            writer.writerow(
                [
                    r["review_hash"],
                    r["property_id"],
                    r["review_date"],
                    r["rating"],
                    r["sentiment"],
                    "; ".join(r["topics"]),
                    r.get("ai_summary", ""),
                    r.get("classifier_source", ""),
                    r["reviewer_name"],
                    r["reviewer_country"],
                    r["traveler_type"],
                    r["title"],
                    r["positive_text"],
                    r["negative_text"],
                    r["source_platform"],
                ]
            )
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=azzurro_reviews_export.csv"},
        )

    return app


app = create_app()
