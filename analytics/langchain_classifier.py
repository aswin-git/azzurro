"""LangChain + Gemini API operational topic & sentiment classifier for Azzurro Hotels reviews.

Engineered for strict free-tier rate limits (default: 3.5 RPM / ~17s spacing between calls,
safely below a 5 RPM ceiling) by batching 20 reviews per structured-output Gemini API call
and caching every completed chunk directly into SQLite.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("azzurro.langchain")

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    load_dotenv()
except ImportError:
    pass

CANONICAL_TOPICS: List[str] = [
    "Cleanliness",
    "Check-in experience",
    "Staff or receptionist behaviour",
    "Noise",
    "Facilities",
    "Location",
    "Room condition",
    "Value for money",
]

TopicLiteral = Literal[
    "Cleanliness",
    "Check-in experience",
    "Staff or receptionist behaviour",
    "Noise",
    "Facilities",
    "Location",
    "Room condition",
    "Value for money",
]


class ReviewTopicClassification(BaseModel):
    """Structured output schema for a single review."""

    review_hash: str = Field(default="", description="Exact review_hash of the input review.")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall guest sentiment based on rating and review text."
    )
    positive_topics: List[TopicLiteral] = Field(
        default_factory=list,
        description="Operational topics praised by the guest.",
    )
    negative_topics: List[TopicLiteral] = Field(
        default_factory=list,
        description="Operational topics complained about or criticized by the guest.",
    )
    summary: str = Field(
        default="",
        description="One concise sentence (<=20 words) summarizing the key operational takeaway.",
    )


class BatchReviewClassification(BaseModel):
    """Structured output schema for classifying a batch of reviews in a single API call."""

    items: List[ReviewTopicClassification] = Field(
        description="One classification entry per input review, preserving its exact review_hash."
    )


SYSTEM_PROMPT = """You are a hospitality operations analyst for Azzurro Hotels in Sydney.
Classify each Booking.com guest review into the exact 8 operational topics below:
- Cleanliness
- Check-in experience
- Staff or receptionist behaviour
- Noise
- Facilities
- Location
- Room condition
- Value for money

Rules:
1. Preserve each review's exact `review_hash` in the output.
2. Only assign topics genuinely mentioned or clearly implied in the review.
3. Put praised topics in `positive_topics` and complaints/issues in `negative_topics`.
4. Set `sentiment` to "positive" (>=7.0), "neutral" (5.0-6.9), or "negative" (<5.0) adjusted if text strongly contradicts the score.
5. Write a concise operational `summary` (<=20 words) for non-technical hotel staff.
"""


def get_gemini_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def is_gemini_configured() -> bool:
    return bool(get_gemini_api_key())


def get_gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


class GeminiLangChainClassifier:
    """Wraps LangChain `ChatGoogleGenerativeAI` with 3.5 RPM pacing and 20-review batching."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
        max_rpm: Optional[float] = None,
    ) -> None:
        self.api_key = api_key or get_gemini_api_key()
        self.model_name = model or get_gemini_model_name()
        env_batch = int(os.getenv("GEMINI_BATCH_SIZE", "20"))
        env_rpm = float(os.getenv("GEMINI_RPM_LIMIT", "3.5"))
        self.chunk_size = max(1, min(chunk_size or env_batch, 25))
        self.max_rpm = max(1.0, min(max_rpm or env_rpm, 4.5))
        self.min_interval_sec = 60.0 / self.max_rpm  # Default 3.5 RPM => 17.14s between calls
        self._last_call_ts: float = 0.0

        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Add GEMINI_API_KEY=... to your .env file to enable LangChain + Gemini classification."
            )

        try:
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore[import-not-found]
            from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "LangChain Gemini packages not installed. Run: pip install langchain-core langchain-google-genai"
            ) from exc

        llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=self.api_key,
            temperature=0.0,
            max_retries=2,
        )
        self.batch_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", "Classify all reviews in this JSON array:\n{reviews_json}"),
                ]
            )
            | llm.with_structured_output(BatchReviewClassification)
        )
        self.single_chain = (
            ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    (
                        "human",
                        "review_hash: {review_hash}\nProperty: {property_id}\nRating: {rating}/10\n"
                        "Title: {title}\nLiked: {positive_text}\nDisliked: {negative_text}",
                    ),
                ]
            )
            | llm.with_structured_output(ReviewTopicClassification)
        )

    def _pace(self) -> None:
        """Enforce strict <= 3.5 RPM pacing so we never exceed a 5 RPM API quota."""
        if self._last_call_ts > 0:
            elapsed = time.monotonic() - self._last_call_ts
            remaining = self.min_interval_sec - elapsed
            if remaining > 0:
                logger.info(
                    "[Gemini+LangChain] Pacing %.1fs to stay within %.1f RPM limit...",
                    remaining,
                    self.max_rpm,
                )
                time.sleep(remaining)
        self._last_call_ts = time.monotonic()

    def _to_record(self, item: ReviewTopicClassification, fallback_hash: str) -> Dict[str, Any]:
        pos = sorted(set(item.positive_topics))
        neg = sorted(set(item.negative_topics))
        return {
            "review_hash": item.review_hash or fallback_hash,
            "sentiment": item.sentiment,
            "topics": sorted(set(pos) | set(neg)),
            "positive_topics": pos,
            "negative_topics": neg,
            "ai_summary": item.summary.strip(),
            "classifier_source": f"gemini-langchain ({self.model_name})",
            "classified_at": datetime.now(timezone.utc).isoformat(),
        }

    def classify_one(self, review: Dict[str, Any]) -> Dict[str, Any]:
        self._pace()
        result: ReviewTopicClassification = self.single_chain.invoke(
            {
                "review_hash": review["review_hash"],
                "property_id": review.get("property_id", ""),
                "rating": review.get("rating", 0.0),
                "title": review.get("title") or "(none)",
                "positive_text": review.get("positive_text") or "(none)",
                "negative_text": review.get("negative_text") or "(none)",
            }
        )
        return self._to_record(result, review["review_hash"])

    def classify_batch(
        self,
        reviews: List[Dict[str, Any]],
        max_reviews: int = 80,
        on_chunk_done: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Classify reviews in chunks of 20 at <= 3.5 RPM (e.g. 60 reviews = 3 API calls)."""
        target = reviews[:max_reviews]
        classified: List[Dict[str, Any]] = []
        errors: List[str] = []

        for start in range(0, len(target), self.chunk_size):
            chunk = target[start : start + self.chunk_size]
            compact_input = [
                {
                    "review_hash": r["review_hash"],
                    "property_id": r.get("property_id", ""),
                    "rating": r.get("rating", 0.0),
                    "title": (r.get("title") or "")[:140],
                    "liked": (r.get("positive_text") or "")[:350],
                    "disliked": (r.get("negative_text") or "")[:350],
                }
                for r in chunk
            ]
            chunk_results: List[Dict[str, Any]] = []
            try:
                self._pace()
                batch_out: BatchReviewClassification = self.batch_chain.invoke(
                    {"reviews_json": json.dumps(compact_input, ensure_ascii=False)}
                )
                valid_hashes = {r["review_hash"] for r in chunk}
                for item in batch_out.items:
                    if item.review_hash in valid_hashes:
                        chunk_results.append(self._to_record(item, item.review_hash))
                logger.info(
                    "[Gemini+LangChain] Classified chunk %d-%d (%d/%d reviews)",
                    start + 1,
                    start + len(chunk),
                    len(chunk_results),
                    len(chunk),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Chunk %d-%d failed (%s). Skipping chunk to preserve RPM budget.",
                    start + 1,
                    start + len(chunk),
                    exc,
                )
                errors.append(f"chunk {start + 1}-{start + len(chunk)}: {exc}")

            if chunk_results:
                classified.extend(chunk_results)
                if on_chunk_done:
                    on_chunk_done(chunk_results)

        return {
            "classified": classified,
            "classified_count": len(classified),
            "error_count": len(errors),
            "errors": errors[:10],
            "model": self.model_name,
            "rpm_limit": self.max_rpm,
            "batch_size": self.chunk_size,
        }
