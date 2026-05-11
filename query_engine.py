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
    "user_id", "userid", "user id", "uid", "customer_id", "customer id", "account_id", "account id",
    "member_id", "member id", "visitor_id", "visitor id", "anonymous_id", "anonymous id", "device_id",
    "device id", "client_id", "client id", "profile_id", "profile id", "session_id", "session id",
    "email", "phone"
]

DATE_COL_HINTS = [
    "date", "datetime", "timestamp", "time", "created_at", "updated_at", "event_time", "event_date"
]

COHORT_DATE_HINTS = [
    "signup_date", "signup", "registration_date", "registration", "first_purchase_date",
    "first_order_date", "acquisition_date", "cohort_date", "join_date", "created_at",
    "first_seen", "install_date", "onboarding_date", "start_date"
]

ACTIVITY_DATE_HINTS = [
    "event_date", "activity_date", "login_date", "session_date", "return_date",
    "purchase_date", "transaction_date", "engagement_date", "visit_date", "date"
]

AGG_ALIASES = {
    "sum": ["total", "sum", "sales", "revenue", "profit", "amount"],
    "mean": ["average", "avg", "mean"],
    "max": ["highest", "max", "maximum", "largest", "top", "best"],
    "min": ["lowest", "min", "minimum", "smallest", "bottom", "worst"],
    "count": ["count", "number of", "how many"],
}

RETENTION_DAY_PATTERNS = {
    "d1": 1, "day 1": 1, "1 day": 1,
    "d3": 3, "day 3": 3, "3 day": 3,
    "d7": 7, "day 7": 7, "7 day": 7,
    "d14": 14, "day 14": 14, "14 day": 14,
    "d30": 30, "day 30": 30, "30 day": 30,
    "d60": 60, "day 60": 60, "60 day": 60,
    "d90": 90, "day 90": 90, "90 day": 90,
}


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


def detect_agg_func(query: str) -> str:
    q = normalize_text(query)
    for func, aliases in AGG_ALIASES.items():
        for alias in aliases:
            if alias in q:
                return func
    return "sum"


def detect_retention_day(query: str) -> Optional[int]:
    q = normalize_text(query)
    for pattern, days in RETENTION_DAY_PATTERNS.items():
        if pattern in q:
            return days
    match = re.search(r"d(\d+)", q)
    if match:
        return int(match.group(1))
    match = re.search(r"day (\d+)", q)
    if match:
        return int(match.group(1))
    return None


def detect_cohort_query(query: str) -> bool:
    q = normalize_text(query)
    cohort_keywords = ["cohort", "retention", "d1", "d3", "d7", "d14", "d30", "d60", "d90",
                       "day 1", "day 3", "day 7", "day 14", "day 30", "day 60", "day 90",
                       "retained", "churn", "stickiness"]
    return any(kw in q for kw in cohort_keywords)


def detect_wants_lowest(query: str) -> bool:
    """Check if user is asking for lowest/worst/minimum retention."""
    q = normalize_text(query)
    lowest_keywords = ["lowest", "worst", "minimum", "min", "smallest", "bottom", "least"]
    return any(kw in q for kw in lowest_keywords)


def detect_wants_highest(query: str) -> bool:
    """Check if user is asking for highest/best/maximum retention."""
    q = normalize_text(query)
    highest_keywords = ["highest", "best", "maximum", "max", "largest", "top", "most"]
    return any(kw in q for kw in highest_keywords)


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

    candidates = [col for col in df.columns if any(h in col.lower() for h in DATE_COL_HINTS)]
    for col in candidates:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.6:
            return col

    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return col

    return None


def infer_user_column(df: pd.DataFrame) -> Optional[str]:
    col = detect_column_by_fuzzy(list(df.columns), USER_COL_HINTS)
    if col:
        return col

    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["user", "customer", "member", "visitor", "account", "anonymous", "device", "client", "profile", "session"]):
            return c

    return None


def infer_cohort_date_column(df: pd.DataFrame) -> Optional[str]:
    col = detect_column_by_fuzzy(list(df.columns), COHORT_DATE_HINTS)
    if col:
        return col

    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["signup", "registration", "first", "cohort", "acquisition", "join", "created", "install", "onboarding", "start"]):
            return c

    return None


def infer_activity_date_column(df: pd.DataFrame) -> Optional[str]:
    col = detect_column_by_fuzzy(list(df.columns), ACTIVITY_DATE_HINTS)
    if col:
        return col

    for c in df.columns:
        name = c.lower()
        if any(k in name for k in ["event", "activity", "login", "session", "return", "purchase", "transaction", "engagement", "visit"]):
            return c

    return None


def infer_numeric_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    metric_name = metric_name.lower().strip()
    for c in df.columns:
        if metric_name and metric_name in c.lower():
            return c

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        return numeric_cols[0]

    for c in df.columns:
        if df[c].dtype == "object":
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().mean() >= 0.7:
                return c

    return None


def infer_group_column(df: pd.DataFrame, group_name: str) -> Optional[str]:
    group_name = group_name.lower().strip()
    for c in df.columns:
        if group_name and group_name in c.lower():
            return c

    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if obj_cols:
        return obj_cols[0]

    non_numeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    return non_numeric[0] if non_numeric else None


def parse_time_window(df: pd.DataFrame, date_col: str, window: str) -> Tuple[pd.DataFrame, str]:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col])

    now = work[date_col].max()
    if pd.isna(now):
        return work.iloc[0:0], "No valid dates found."

    if window == "last_week":
        end = now.normalize()
        start = end - timedelta(days=7)
        return work[(work[date_col] >= start) & (work[date_col] < end)], f"Filtered to last 7 days up to {end.date()}."
    if window == "this_week":
        start = now.normalize() - timedelta(days=now.dayofweek)
        end = now.normalize() + timedelta(days=1)
        return work[(work[date_col] >= start) & (work[date_col] < end)], f"Filtered to this week starting {start.date()}."
    if window == "past_7_days":
        end = now.normalize() + timedelta(days=1)
        start = end - timedelta(days=7)
        return work[(work[date_col] >= start) & (work[date_col] < end)], f"Filtered to past 7 days ending {end.date()}."
    if window == "this_month":
        start = now.replace(day=1).normalize()
        end = now.normalize() + timedelta(days=1)
        return work[(work[date_col] >= start) & (work[date_col] < end)], f"Filtered to this month starting {start.date()}."
    if window == "last_month":
        first_this_month = now.replace(day=1).normalize()
        last_prev_month = first_this_month - timedelta(days=1)
        start = last_prev_month.replace(day=1).normalize()
        end = first_this_month
        return work[(work[date_col] >= start) & (work[date_col] < end)], f"Filtered to last month from {start.date()} to {end.date()}."
    if window == "past_30_days":
        end = now.normalize() + timedelta(days=1)
        start = end - timedelta(days=30)
        return work[(work[date_col] >= start) & (work[date_col] < end)], f"Filtered to past 30 days ending {end.date()}."

    return work, "No time filter applied."


def compute_dau(df: pd.DataFrame, user_col: str, date_col: str, window: Optional[str] = None) -> QueryResult:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, user_col])

    if work.empty:
        return QueryResult("error", None, "No usable rows after date/user cleanup.")

    if window:
        work, msg = parse_time_window(work, date_col, window)
    else:
        msg = "No time filter applied."

    if work.empty:
        return QueryResult("error", None, "No rows matched the requested time window.")

    work["__day"] = work[date_col].dt.floor("D")
    daily = work.groupby("__day")[user_col].nunique().reset_index(name="dau").sort_values("__day").rename(columns={"__day": "date"})
    return QueryResult("dataframe", daily, f"Computed DAU successfully. {msg}")


def compute_wau(df: pd.DataFrame, user_col: str, date_col: str, window: Optional[str] = None) -> QueryResult:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, user_col])

    if work.empty:
        return QueryResult("error", None, "No usable rows after date/user cleanup.")

    if window:
        work, msg = parse_time_window(work, date_col, window)
    else:
        msg = "No time filter applied."

    if work.empty:
        return QueryResult("error", None, "No rows matched the requested time window.")

    work["__week"] = work[date_col].dt.to_period("W").dt.start_time
    weekly = work.groupby("__week")[user_col].nunique().reset_index(name="wau").sort_values("__week").rename(columns={"__week": "week"})
    return QueryResult("dataframe", weekly, f"Computed WAU successfully. {msg}")


def compute_mau(df: pd.DataFrame, user_col: str, date_col: str, window: Optional[str] = None) -> QueryResult:
    work = df.copy()
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col, user_col])

    if work.empty:
        return QueryResult("error", None, "No usable rows after date/user cleanup.")

    if window:
        work, msg = parse_time_window(work, date_col, window)
    else:
        msg = "No time filter applied."

    if work.empty:
        return QueryResult("
