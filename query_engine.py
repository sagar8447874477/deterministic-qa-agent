import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, List, Tuple, Any

import pandas as pd
from rapidfuzz import process, fuzz


METRIC_ALIASES = {
    "dau": ["dau", "daily active users", "daily active user", "active users per day"],
    "wau": ["wau", "weekly active users", "weekly active user", "active users per week"],
    "mau": ["mau", "monthly active users", "monthly active user", "active users per month"],
}

TIME_PATTERNS = {
    "last week": "last_week",
    "this week": "this_week",
    "past 7 days": "past_7_days",
    "last 7 days": "past_7_days",
    "this month": "this_month",
    "last month": "last_month",
    "past 30 days": "past_30_days",
    "last 30 days": "past_30_days",
}

USER_COL_HINTS = [
    "user_id",
    "userid",
    "uid",
    "user",
    "customer_id",
    "account_id",
    "member_id",
    "visitor_id",
]

DATE_COL_HINTS = [
    "date",
    "datetime",
    "timestamp",
    "time",
    "created_at",
    "updated_at",
    "event_time",
    "event_date",
]


@dataclass
class QueryResult:
    answer_type: str
    data: Any
    message: str = ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_metric(query: str) -> Optional[str]:
    q = normalize_text(query)
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return metric
    return None


def detect_time_window(query: str) -> Optional[str]:
    q = normalize_text(query)
    for phrase, window in TIME_PATTERNS.items():
        if phrase in q:
            return window
    return None


def detect_column_by_fuzzy(columns: List[str], hints: List[str]) -> Optional[str]:
    if not columns:
        return None
    normalized = {c: c.lower() for c in columns}
    choices = list(normalized.values())
    for hint in hints:
        match = process.extractOne(hint, choices, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 80:
            for orig, low in normalized.items():
                if low == match[0]:
                    return orig
    return None


def infer_datetime_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    candidates = []
    for col in df.columns:
        name = col.lower()
        if any(h in name for h in DATE_COL_HINTS):
            candidates.append(col)

    for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        
