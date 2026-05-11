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
    "user id",
    "uid",
    "customer_id",
    "customer id",
    "account_id",
    "account id",
    "member_id",
    "member id",
    "visitor_id",
    "visitor id",
    "anonymous_id",
    "anonymous id",
    "device_id",
    "device id",
    "client_id",
    "client id",
    "profile_id",
    "profile id",
    "session_id",
    "session id",
    "email",
    "phone",
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

AGG_ALIASES = {
    "sum": ["total", "sum", "sales", "revenue", "profit", "amount", "by"],
    "mean": ["average", "avg", "mean"],
    "max": ["highest", "max", "maximum", "largest", "top"],
    "min": ["lowest", "min", "minimum", "smallest", "bottom"],
    "count": ["count", "number of", "how many"],
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
        if any(k in name for k in [
            "user", "customer", "member", "visitor", "account",
            "anonymous", "device", "client", "profile", "session"
        ]):
            return c

    return None


def infer_numeric_column(df: pd.DataFrame, metric_name: str) -> Optional[str]:
    metric_name = metric_name.lower().strip()
    if metric_name:
        for c in df.columns:
            if metric_name in c.lower():
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
    if group_name:
        for c in df.columns:
            if group_name in c.lower():
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
    daily = (
        work.groupby("__day")[user_col]
        .nunique()
        .reset_index(name="dau")
        .sort_values("__day")
        .rename(columns={"__day": "date"})
    )

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
    weekly = (
        work.groupby("__week")[user_col]
        .nunique()
        .reset_index(name="wau")
        .sort_values("__week")
        .rename(columns={"__week": "week"})
    )

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
        return QueryResult("error", None, "No rows matched the requested time window.")

    work["__month"] = work[date_col].dt.to_period("M").dt.start_time
    monthly = (
        work.groupby("__month")[user_col]
        .nunique()
        .reset_index(name="mau")
        .sort_values("__month")
        .rename(columns={"__month": "month"})
    )

    return QueryResult("dataframe", monthly, f"Computed MAU successfully. {msg}")


def count_rows(df: pd.DataFrame) -> QueryResult:
    return QueryResult("text", len(df), f"Total rows: {len(df)}")


def show_columns(df: pd.DataFrame) -> QueryResult:
    return QueryResult("dataframe", pd.DataFrame({"columns": list(df.columns)}), "Showing columns.")


def describe_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null": [int(df[c].notna().sum()) for c in df.columns],
        "nulls": [int(df[c].isna().sum()) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })


def handle_by_query(df: pd.DataFrame, query: str) -> Optional[QueryResult]:
    q = normalize_text(query)

    m = re.match(
        r"^(?:show|find|give|what is|what are)?\s*([a-zA-Z0-9_ ]+?)\s+by\s+([a-zA-Z0-9_ ]+?)\s*$",
        q
    )
    if not m:
        return None

    metric_part = m.group(1).strip()
    group_part = m.group(2).strip()

    metric_col = infer_numeric_column(df, metric_part)
    group_col = infer_group_column(df, group_part)

    if metric_col is None:
        return QueryResult("error", None, f"I could not detect a numeric column for '{metric_part}'.")
    if group_col is None:
        return QueryResult("error", None, f"I could not detect a group column for '{group_part}'.")

    agg_func = detect_agg_func(q)

    if agg_func == "sum":
        result = df.groupby(group_col, dropna=False)[metric_col].sum().reset_index()
    elif agg_func == "mean":
        result = df.groupby(group_col, dropna=False)[metric_col].mean().reset_index()
    elif agg_func 
