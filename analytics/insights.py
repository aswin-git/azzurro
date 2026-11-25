"""Operational topic classification, sentiment analysis, and weekly KPI analytics engine.

Prefers cached LangChain + Gemini API classifications (`review_classifications` table)
and falls back transparently to a deterministic 8-topic lexicon when a review has not
yet been classified by Gemini.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional


TOPIC_LEXICON: Dict[str, List[str]] = {
    "Cleanliness": [
        r"\bclean\w*\b", r"\bdirty\b", r"\bfilth\w*\b", r"\bmould\w*\b", r"\bmold\w*\b",
        r"\bstain\w*\b", r"\bhygien\w*\b", r"\bsmell\w*\b", r"\bstank\b", r"\bstink\w*\b",
        r"\bodour\b", r"\bodor\b", r"\bvacuum\w*\b", r"\bhairs?\b", r"\btrash\b",
        r"\brubbish\b", r"\bdisgusting\b", r"\bgross\b", r"\bsanitiz\w*\b", r"\bhousekeep\w*\b",
        r"\bcockroach\w*\b", r"\bants\b", r"\bflies\b", r"\bmessy\b", r"\btoilet paper\b", r"\bclogged\b",
    ],
    "Check-in experience": [
        r"\bcheck[\s-]?in\b", r"\bcheck[\s-]?out\b", r"\bdoor code\w*\b", r"\baccess code\w*\b",
        r"\bkeypad\b", r"\blocked out\b", r"\bself check\b", r"\bonline check\b",
        r"\bvirtual reception\b", r"\bluggage\b", r"\bbag(?:s)? storage\b", r"\bstorage area\b", r"\bpassport\b",
    ],
    "Staff or receptionist behaviour": [
        r"\bstaff\w*\b", r"\breception\w*\b", r"\bmanager\b", r"\bhost\b", r"\bconcierge\b",
        r"\bcustomer service\b", r"\bservice\b", r"\bfriendly\b", r"\bhelpful\b", r"\brude\b",
        r"\babusive\b", r"\babused\b", r"\bempathet\w*\b", r"\bempathy\b", r"\brobot\w*\b",
        r"\bno offline staff\b", r"\bno human staff\b", r"\bno staff\b", r"\bvirtual\b", r"\bscreen\b", r"\bipad\b", r"\bzoom\b",
    ],
    "Noise": [
        r"\bnois\w*\b", r"\bloud\w*\b", r"\bquiet\b", r"\bhear everything\b", r"\bthin walls?\b",
        r"\bslamming\b", r"\bcreak\w*\b", r"\bjet engine\b", r"\bearplug\w*\b", r"\btraffic\b",
        r"\bconstruction\b", r"\bscream\w*\b", r"\balarm\b",
    ],
    "Facilities": [
        r"\bkitchen\w*\b", r"\bbathroom\w*\b", r"\bshower\w*\b", r"\btoilet\w*\b", r"\bwi[\s-]?fi\b",
        r"\blift\b", r"\belevator\b", r"\bstairs?\b", r"\blaundry\b", r"\bwashing machine\b",
        r"\bdryer\b", r"\bfridge\b", r"\bmicrowave\b", r"\bair[\s-]?con\w*\b", r"\bac\b",
        r"\bheating\b", r"\blockers?\b", r"\btowels?\b", r"\bbreakfast\b", r"\bdinner\b",
        r"\bmeals?\b", r"\bfacilit\w*\b", r"\bamenit\w*\b", r"\bqueue\w*\b", r"\bcommon area\b",
    ],
    "Location": [
        r"\blocation\b", r"\bcentral\b", r"\bstation\b", r"\btransport\b", r"\bwalking distance\b",
        r"\bcbd\b", r"\bdarling harbour\b", r"\bkings cross\b", r"\bnear\b", r"\bclose to\b",
        r"\bconvenient\b", r"\bcafes?\b", r"\brestaurants?\b", r"\bneighbourhood\b", r"\bneighborhood\b", r"\baccessible\b", r"\bspot\b",
    ],
    "Room condition": [
        r"\brooms?\b", r"\bbeds?\b", r"\bmattress\b", r"\bpillow\w*\b", r"\bcurtain\w*\b",
        r"\bbunk\b", r"\block\b", r"\bbroken\b", r"\brundown\b", r"\bcramped\b", r"\bsmall\w*\b",
        r"\btiny\b", r"\bdark\b", r"\bcarpet\w*\b", r"\bhumid\b", r"\bventilation\b",
        r"\bwobbly\b", r"\bpictures?\b", r"\bphotos?\b", r"\bmaintained\b", r"\brenovated\b", r"\bpower outlet\b",
    ],
    "Value for money": [
        r"\bvalue\b", r"\bprice\w*\b", r"\bcheap\w*\b", r"\boverpriced\b", r"\bbudget\b",
        r"\bworth\b", r"\bmoney\b", r"\bexpensive\b", r"\bfree dinner\b", r"\bfree breakfast\b",
        r"\brefund\w*\b", r"\bcharge\w*\b", r"\bfines?\b", r"\baffordable\b", r"\bten dollars\b", r"\$10\b", r"\$15\b",
    ],
}

COMPILED_LEXICON: Dict[str, List[re.Pattern[str]]] = {
    topic: [re.compile(pat, flags=re.IGNORECASE) for pat in patterns]
    for topic, patterns in TOPIC_LEXICON.items()
}


def classify_sentiment(rating: float) -> str:
    if rating >= 7.0:
        return "positive"
    if rating >= 5.0:
        return "neutral"
    return "negative"


def extract_topics_from_text(text: str) -> List[str]:
    if not text or not text.strip():
        return []
    return [topic for topic, patterns in COMPILED_LEXICON.items() if any(p.search(text) for p in patterns)]


def enrich_review(
    review: Dict[str, Any],
    ai_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach sentiment & 8-topic tags using Gemini+LangChain cache first, falling back to lexicon."""
    review_hash = review.get("review_hash", "")
    if ai_cache and review_hash in ai_cache:
        cached = ai_cache[review_hash]
        enriched = dict(review)
        enriched["sentiment"] = cached["sentiment"]
        enriched["topics"] = cached["topics"]
        enriched["positive_topics"] = cached["positive_topics"]
        enriched["negative_topics"] = cached["negative_topics"]
        enriched["ai_summary"] = cached.get("ai_summary", "")
        enriched["classifier_source"] = cached.get("classifier_source", "gemini-langchain")
        return enriched

    rating = float(review.get("rating", 0.0))
    sentiment = classify_sentiment(rating)
    title = review.get("title") or ""
    pos_text = review.get("positive_text") or ""
    neg_text = review.get("negative_text") or ""

    pos_topics = extract_topics_from_text(pos_text)
    neg_topics = extract_topics_from_text(neg_text)
    title_topics = extract_topics_from_text(title)

    if sentiment == "negative":
        for t in title_topics:
            if t not in neg_topics:
                neg_topics.append(t)
    elif sentiment == "positive":
        for t in title_topics:
            if t not in pos_topics:
                pos_topics.append(t)

    all_topics = sorted(set(pos_topics) | set(neg_topics) | set(title_topics))
    enriched = dict(review)
    enriched["sentiment"] = sentiment
    enriched["topics"] = all_topics
    enriched["positive_topics"] = sorted(set(pos_topics))
    enriched["negative_topics"] = sorted(set(neg_topics))
    enriched["ai_summary"] = ""
    enriched["classifier_source"] = "lexicon-fallback"
    return enriched


def _parse_iso(d: str) -> date:
    return datetime.strptime(d[:10], "%Y-%m-%d").date()


def compute_weekly_comparison(
    reviews: List[Dict[str, Any]],
    anchor_date: Optional[str] = None,
) -> Dict[str, Any]:
    if not reviews:
        return {
            "current_week_avg": None,
            "current_week_count": 0,
            "previous_week_avg": None,
            "previous_week_count": 0,
            "delta": None,
            "current_window": None,
            "previous_window": None,
        }

    dates = [_parse_iso(r["review_date"]) for r in reviews]
    end_dt = _parse_iso(anchor_date) if anchor_date else max(dates)
    curr_start = end_dt - timedelta(days=6)
    prev_end = curr_start - timedelta(days=1)
    prev_start = prev_end - timedelta(days=6)

    curr_reviews = [r for r in reviews if curr_start <= _parse_iso(r["review_date"]) <= end_dt]
    prev_reviews = [r for r in reviews if prev_start <= _parse_iso(r["review_date"]) <= prev_end]

    if not anchor_date and (len(curr_reviews) < 3 or len(prev_reviews) == 0):
        unique_dates = sorted(set(dates), reverse=True)
        for cand_end in unique_dates:
            c_start = cand_end - timedelta(days=6)
            p_end = c_start - timedelta(days=1)
            p_start = p_end - timedelta(days=6)
            c_revs = [r for r in reviews if c_start <= _parse_iso(r["review_date"]) <= cand_end]
            p_revs = [r for r in reviews if p_start <= _parse_iso(r["review_date"]) <= p_end]
            if len(c_revs) >= 3 and len(p_revs) >= 2:
                end_dt, curr_start, prev_start, prev_end = cand_end, c_start, p_start, p_end
                curr_reviews, prev_reviews = c_revs, p_revs
                break

    curr_avg = (
        round(sum(float(r["rating"]) for r in curr_reviews) / len(curr_reviews), 2)
        if curr_reviews
        else None
    )
    prev_avg = (
        round(sum(float(r["rating"]) for r in prev_reviews) / len(prev_reviews), 2)
        if prev_reviews
        else None
    )
    delta = round(curr_avg - prev_avg, 2) if (curr_avg is not None and prev_avg is not None) else None

    return {
        "current_week_avg": curr_avg,
        "current_week_count": len(curr_reviews),
        "previous_week_avg": prev_avg,
        "previous_week_count": len(prev_reviews),
        "delta": delta,
        "current_window": f"{curr_start.isoformat()} to {end_dt.isoformat()}",
        "previous_window": f"{prev_start.isoformat()} to {prev_end.isoformat()}",
    }


def build_dashboard_summary(
    raw_reviews: List[Dict[str, Any]],
    properties: List[Dict[str, Any]],
    anchor_date: Optional[str] = None,
    ai_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    reviews = [enrich_review(r, ai_cache=ai_cache) for r in raw_reviews]
    total = len(reviews)
    ai_classified_count = sum(1 for r in reviews if r["classifier_source"].startswith("gemini"))

    avg_rating = round(sum(float(r["rating"]) for r in reviews) / total, 2) if total else None
    pos_reviews = [r for r in reviews if r["sentiment"] == "positive"]
    neu_reviews = [r for r in reviews if r["sentiment"] == "neutral"]
    neg_reviews = [r for r in reviews if r["sentiment"] == "negative"]

    weekly = compute_weekly_comparison(reviews, anchor_date=anchor_date)

    by_prop: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in reviews:
        by_prop[r["property_id"]].append(r)

    property_breakdown: List[Dict[str, Any]] = []
    for prop in properties:
        pid = prop["id"]
        previews = by_prop.get(pid, [])
        pcount = len(previews)
        pavg = round(sum(float(r["rating"]) for r in previews) / pcount, 2) if pcount else None
        pneg = [r for r in previews if r["sentiment"] == "negative"]
        ppos = [r for r in previews if r["sentiment"] == "positive"]

        neg_topic_counter: Counter[str] = Counter()
        for r in pneg:
            for t in r["negative_topics"] or r["topics"]:
                neg_topic_counter[t] += 1
        top_issue = neg_topic_counter.most_common(1)[0][0] if neg_topic_counter else "None flagged"

        property_breakdown.append(
            {
                "id": pid,
                "name": prop["name"],
                "booking_title": prop.get("booking_title"),
                "booking_url": prop.get("booking_url"),
                "address": prop.get("address"),
                "booking_overall_rating": prop.get("booking_overall_rating"),
                "booking_total_reviews": prop.get("booking_total_reviews"),
                "sample_count": pcount,
                "sample_avg_rating": pavg,
                "positive_count": len(ppos),
                "negative_count": len(pneg),
                "negative_share_pct": round((len(pneg) / pcount) * 100, 1) if pcount else 0.0,
                "top_negative_topic": top_issue,
            }
        )

    neg_topic_counts: Counter[str] = Counter()
    pos_topic_counts: Counter[str] = Counter()
    all_topic_counts: Counter[str] = Counter()

    for r in reviews:
        for t in r["topics"]:
            all_topic_counts[t] += 1
        if r["sentiment"] == "negative":
            for t in r["negative_topics"] or r["topics"]:
                neg_topic_counts[t] += 1
        elif r["sentiment"] == "positive":
            for t in r["positive_topics"] or r["topics"]:
                pos_topic_counts[t] += 1

    topic_breakdown: List[Dict[str, Any]] = []
    for topic in TOPIC_LEXICON.keys():
        nc = neg_topic_counts[topic]
        pc = pos_topic_counts[topic]
        topic_breakdown.append(
            {
                "topic": topic,
                "total_mentions": all_topic_counts[topic],
                "negative_mentions": nc,
                "positive_mentions": pc,
                "negative_share_pct": round((nc / len(neg_reviews)) * 100, 1) if neg_reviews else 0.0,
                "positive_share_pct": round((pc / len(pos_reviews)) * 100, 1) if pos_reviews else 0.0,
            }
        )
    topic_breakdown.sort(key=lambda x: x["negative_mentions"], reverse=True)

    monthly_buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"positive": 0, "neutral": 0, "negative": 0, "sum_rating": 0.0, "count": 0}
    )
    for r in reviews:
        ym = r["review_date"][:7]
        b = monthly_buckets[ym]
        b[r["sentiment"]] += 1
        b["sum_rating"] += float(r["rating"])
        b["count"] += 1

    trends = [
        {
            "period": ym,
            "positive": vals["positive"],
            "neutral": vals["neutral"],
            "negative": vals["negative"],
            "count": vals["count"],
            "avg_rating": round(vals["sum_rating"] / vals["count"], 2),
        }
        for ym, vals in sorted(monthly_buckets.items())
    ]

    insights: List[Dict[str, str]] = []
    if neg_reviews and topic_breakdown and topic_breakdown[0]["negative_mentions"] > 0:
        top_neg = topic_breakdown[0]
        insights.append(
            {
                "severity": "high",
                "headline": (
                    f"{round(top_neg['negative_share_pct'])}% of negative reviews "
                    f"mentioned {top_neg['topic'].lower()}-related issues."
                ),
                "detail": (
                    f"Flagged in {top_neg['negative_mentions']} of {len(neg_reviews)} "
                    f"negative guest reviews across the active filter."
                ),
            }
        )
        if len(topic_breakdown) > 1 and topic_breakdown[1]["negative_mentions"] > 0:
            second_neg = topic_breakdown[1]
            insights.append(
                {
                    "severity": "medium",
                    "headline": (
                        f"{round(second_neg['negative_share_pct'])}% of negative reviews "
                        f"cited {second_neg['topic'].lower()}."
                    ),
                    "detail": (
                        f"Mentioned in {second_neg['negative_mentions']} negative reviews — "
                        f"second-highest operational driver."
                    ),
                }
            )

    if pos_reviews:
        top_pos = max(topic_breakdown, key=lambda x: x["positive_mentions"])
        if top_pos["positive_mentions"] > 0:
            insights.append(
                {
                    "severity": "positive",
                    "headline": (
                        f"{top_pos['topic']} is the #1 praised strength "
                        f"({round(top_pos['positive_share_pct'])}% of positive reviews)."
                    ),
                    "detail": (
                        f"Highlighted in {top_pos['positive_mentions']} of {len(pos_reviews)} "
                        f"positive reviews."
                    ),
                }
            )

    active_props = [p for p in property_breakdown if p["sample_count"] > 0]
    if len(active_props) > 1:
        hotspot = max(active_props, key=lambda x: x["negative_share_pct"])
        if hotspot["negative_count"] > 0:
            insights.append(
                {
                    "severity": "medium",
                    "headline": (
                        f"{hotspot['name']} has the highest negative review rate "
                        f"({hotspot['negative_share_pct']}%)."
                    ),
                    "detail": (
                        f"Primary recurring complaint at {hotspot['name']}: "
                        f"{hotspot['top_negative_topic']}."
                    ),
                }
            )

    return {
        "kpis": {
            "total_reviews": total,
            "overall_avg_rating": avg_rating,
            "positive_count": len(pos_reviews),
            "neutral_count": len(neu_reviews),
            "negative_count": len(neg_reviews),
            "negative_share_pct": round((len(neg_reviews) / total) * 100, 1) if total else 0.0,
            "ai_classified_count": ai_classified_count,
            **weekly,
        },
        "properties": property_breakdown,
        "topics": topic_breakdown,
        "trends": trends,
        "insights": insights,
        "reviews": reviews,
    }
