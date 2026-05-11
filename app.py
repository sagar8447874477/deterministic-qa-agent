import os
import sys
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, List, Tuple, Any

import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz


# ==================== QUERY ENGINE (INLINE) ====================

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
    q = normalize_text(query)
    lowest_keywords = ["lowest", "worst", "minimum", "min", "smallest", "bottom", "least"]
    return any(kw in q for kw in lowest_keywords)


def detect_wants_highest(query: str) -> bool:
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
        return QueryResult("error", None, "No rows matched the requested time window.")

    work["__month"] = work[date_col].dt.to_period("M").dt.start_time
    monthly = work.groupby("__month")[user_col].nunique().reset_index(name="mau").sort_values("__month").rename(columns={"__month": "month"})
    return QueryResult("dataframe", monthly, f"Computed MAU successfully. {msg}")


def compute_cohort_retention(
    df: pd.DataFrame,
    user_col: str,
    cohort_date_col: str,
    activity_date_col: str,
    retention_day: int,
    query: str = ""
) -> QueryResult:
    if user_col not in df.columns:
        return QueryResult("error", None, f"User column '{user_col}' not found.")
    if cohort_date_col not in df.columns:
        return QueryResult("error", None, f"Cohort date column '{cohort_date_col}' not found.")
    if activity_date_col not in df.columns:
        return QueryResult("error", None, f"Activity date column '{activity_date_col}' not found.")
    
    work = df.copy()
    
    work[cohort_date_col] = pd.to_datetime(work[cohort_date_col], errors="coerce")
    work[activity_date_col] = pd.to_datetime(work[activity_date_col], errors="coerce")
    
    same_column = (cohort_date_col == activity_date_col)
    
    if same_column:
        work = work.dropna(subset=[cohort_date_col, user_col])
        
        if work.empty:
            return QueryResult("error", None, "No rows with valid dates after parsing.")
        
        user_first_date = work.groupby(user_col)[cohort_date_col].min().reset_index()
        user_first_date.columns = [user_col, "__cohort_date"]
        work = work.merge(user_first_date, on=user_col, how="left")
        
        work["__cohort_day"] = work["__cohort_date"].dt.floor("D")
        work["__activity_day"] = work[cohort_date_col].dt.floor("D")
    else:
        work = work.dropna(subset=[cohort_date_col, activity_date_col, user_col])
        
        if work.empty:
            return QueryResult("error", None, "No usable rows after date cleanup.")
        
        work["__cohort_day"] = work[cohort_date_col].dt.floor("D")
        work["__activity_day"] = work[activity_date_col].dt.floor("D")
    
    work["__days_since_cohort"] = (work["__activity_day"] - work["__cohort_day"]).dt.days
    
    cohort_users = work.groupby("__cohort_day")[user_col].nunique().reset_index(name="total_users")
    
    retained = work[work["__days_since_cohort"] == retention_day].groupby("__cohort_day")[user_col].nunique().reset_index(name="retained_users")
    
    result = cohort_users.merge(retained, on="__cohort_day", how="left")
    result["retained_users"] = result["retained_users"].fillna(0).astype(int)
    result["retention_rate"] = (result["retained_users"] / result["total_users"] * 100).round(2)
    result = result.sort_values("__cohort_day").rename(columns={"__cohort_day": "cohort_date"})
    
    wants_lowest = detect_wants_lowest(query)
    wants_highest = detect_wants_highest(query)
    
    if not result.empty and result["retention_rate"].notna().any():
        valid_results = result[result["retention_rate"].notna()]
        
        if wants_lowest:
            target_cohort = valid_results.loc[valid_results["retention_rate"].idxmin()]
            target_date = target_cohort["cohort_date"].strftime("%Y-%m-%d")
            target_rate = target_cohort["retention_rate"]
            target_label = "Lowest"
        else:
            target_cohort = valid_results.loc[valid_results["retention_rate"].idxmax()]
            target_date = target_cohort["cohort_date"].strftime("%Y-%m-%d")
            target_rate = target_cohort["retention_rate"]
            target_label = "Highest"
        
        avg_rate = valid_results["retention_rate"].mean()
        median_rate = valid_results["retention_rate"].median()
        
        msg = (f"D{retention_day} retention by cohort. "
               f"{target_label}: {target_rate}% on cohort {target_date}. "
               f"Average: {avg_rate:.1f}%, Median: {median_rate:.1f}%.")
    else:
        msg = f"D{retention_day} retention by cohort. No retention data found."
    
    return QueryResult("dataframe", result, msg)


def compute_retention_summary(
    df: pd.DataFrame,
    user_col: str,
    cohort_date_col: str,
    activity_date_col: str
) -> QueryResult:
    if user_col not in df.columns:
        return QueryResult("error", None, f"User column '{user_col}' not found.")
    if cohort_date_col not in df.columns:
        return QueryResult("error", None, f"Cohort date column '{cohort_date_col}' not found.")
    if activity_date_col not in df.columns:
        return QueryResult("error", None, f"Activity date column '{activity_date_col}' not found.")
    
    work = df.copy()
    work[cohort_date_col] = pd.to_datetime(work[cohort_date_col], errors="coerce")
    work[activity_date_col] = pd.to_datetime(work[activity_date_col], errors="coerce")
    
    same_column = (cohort_date_col == activity_date_col)
    
    if same_column:
        work = work.dropna(subset=[cohort_date_col, user_col])
        if work.empty:
            return QueryResult("error", None, "No rows with valid dates.")
        
        user_first_date = work.groupby(user_col)[cohort_date_col].min().reset_index()
        user_first_date.columns = [user_col, "__cohort_date"]
        work = work.merge(user_first_date, on=user_col, how="left")
        
        work["__cohort_day"] = work["__cohort_date"].dt.floor("D")
        work["__activity_day"] = work[cohort_date_col].dt.floor("D")
    else:
        work = work.dropna(subset=[cohort_date_col, activity_date_col, user_col])
        if work.empty:
            return QueryResult("error", None, "No usable rows after date cleanup.")
        
        work["__cohort_day"] = work[cohort_date_col].dt.floor("D")
        work["__activity_day"] = work[activity_date_col].dt.floor("D")
    
    work["__days_since_cohort"] = (work["__activity_day"] - work["__cohort_day"]).dt.days
    
    cohort_users = work.groupby("__cohort_day")[user_col].nunique().reset_index(name="total_users")
    
    retention_days = [1, 3, 7, 14, 30]
    result = cohort_users.copy()
    
    for day in retention_days:
        retained = work[work["__days_since_cohort"] == day].groupby("__cohort_day")[user_col].nunique().reset_index(name=f"d{day}_retained")
        result = result.merge(retained, on="__cohort_day", how="left")
        result[f"d{day}_retained"] = result[f"d{day}_retained"].fillna(0).astype(int)
        result[f"d{day}_rate"] = (result[f"d{day}_retained"] / result["total_users"] * 100).round(2)
    
    result = result.sort_values("__cohort_day").rename(columns={"__cohort_day": "cohort_date"})
    
    return QueryResult("dataframe", result, "Cohort retention summary (D1, D3, D7, D14, D30).")


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
    m = re.match(r"^(?:show|find|give|what is|what are)?\s*([a-zA-Z0-9_ ]+?)\s+by\s+([a-zA-Z0-9_ ]+?)\s*$", q)
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
    elif agg_func == "max":
        result = df.groupby(group_col, dropna=False)[metric_col].max().reset_index()
    elif agg_func == "min":
        result = df.groupby(group_col, dropna=False)[metric_col].min().reset_index()
    else:
        result = df.groupby(group_col, dropna=False)[metric_col].sum().reset_index()

    result = result.sort_values(metric_col, ascending=False)
    return QueryResult("dataframe", result, f"Grouped {metric_col} by {group_col} using {agg_func}.")


def handle_query(
    df: pd.DataFrame,
    query: str,
    date_col_override: Optional[str] = None,
    user_col_override: Optional[str] = None,
    cohort_date_override: Optional[str] = None,
    activity_date_override: Optional[str] = None
) -> QueryResult:
    q = normalize_text(query)

    if any(p in q for p in ["how many rows", "row count", "total rows"]):
        return count_rows(df)

    if any(p in q for p in ["show columns", "list columns", "columns"]):
        return show_columns(df)

    if any(p in q for p in ["describe dataset", "dataset description", "describe data"]):
        return QueryResult("dataframe", describe_dataset(df), "Dataset description.")

    by_result = handle_by_query(df, query)
    if by_result is not None:
        return by_result

    # Cohort / retention queries
    if detect_cohort_query(q):
        retention_day = detect_retention_day(q)
        
        user_col = user_col_override or infer_user_column(df)
        cohort_date_col = cohort_date_override or infer_cohort_date_column(df)
        activity_date_col = activity_date_override or infer_activity_date_column(df)
        
        general_date_col = date_col_override or infer_datetime_column(df)
        
        if cohort_date_col is None:
            cohort_date_col = general_date_col
        if activity_date_col is None:
            activity_date_col = general_date_col
        
        if user_col is None:
            return QueryResult("error", None, "I could not detect a user identifier column for cohort analysis.")
        if cohort_date_col is None:
            return QueryResult("error", None, "I could not detect a cohort date column. Try selecting one manually.")
        if activity_date_col is None:
            return QueryResult("error", None, "I could not detect an activity date column. Try selecting one manually.")
        
        if retention_day is not None:
            return compute_cohort_retention(df, user_col, cohort_date_col, activity_date_col, retention_day, query=query)
        else:
            return compute_retention_summary(df, user_col, cohort_date_col, activity_date_col)

    metric = detect_metric(q)
    window = detect_time_window(q)
    date_col = date_col_override or infer_datetime_column(df)
    user_col = user_col_override or infer_user_column(df)

    if metric == "dau":
        if date_col is None:
            return QueryResult("error", None, "I could not detect a date/timestamp column.")
        if user_col is None:
            return QueryResult("error", None, "I could not detect a user identifier column.")
        return compute_dau(df, user_col, date_col, window)

    if metric == "wau":
        if date_col is None:
            return QueryResult("error", None, "I could not detect a date/timestamp column.")
        if user_col is None:
            return QueryResult("error", None, "I could not detect a user identifier column.")
        return compute_wau(df, user_col, date_col, window)

    if metric == "mau":
        if date_col is None:
            return QueryResult("error", None, "I could not detect a date/timestamp column.")
        if user_col is None:
            return QueryResult("error", None, "I could not detect a user identifier column.")
        return compute_mau(df, user_col, date_col, window)

    return QueryResult("unsupported", None, "Query not recognized by deterministic engine.")


# ==================== STREAMLIT UI ====================

st.set_page_config(page_title="Deterministic CSV Analytics QA Agent", layout="wide")

st.title("Deterministic CSV Analytics QA Agent")
st.write("Upload a CSV and ask questions like: how many rows, show columns, average sales, profit by category, what was dau last week, or which cohort has highest d7 retention?")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

df = st.session_state.df

if df is None:
    st.info("Upload a CSV file to start.")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Column Mapping")

auto_date = infer_datetime_column(df)
auto_user = infer_user_column(df)
auto_cohort = infer_cohort_date_column(df)
auto_activity = infer_activity_date_column(df)

if auto_activity is None and auto_date is not None:
    auto_activity = auto_date

date_options = ["Auto-detect"] + list(df.columns)
user_options = ["Auto-detect"] + list(df.columns)
cohort_options = ["Auto-detect"] + list(df.columns)
activity_options = ["Auto-detect"] + list(df.columns)

default_date_index = date_options.index(auto_date) if auto_date in df.columns else 0
default_user_index = user_options.index(auto_user) if auto_user in df.columns else 0
default_cohort_index = cohort_options.index(auto_cohort) if auto_cohort in df.columns else 0
default_activity_index = activity_options.index(auto_activity) if auto_activity in df.columns else 0

col1, col2 = st.columns(2)
with col1:
    selected_date_col = st.selectbox("Date/timestamp column", date_options, index=default_date_index)
    selected_cohort_col = st.selectbox("Cohort date column (for retention)", cohort_options, index=default_cohort_index)
with col2:
    selected_user_col = st.selectbox("User identifier column", user_options, index=default_user_index)
    selected_activity_col = st.selectbox("Activity date column (for retention)", activity_options, index=default_activity_index)

date_override = None if selected_date_col == "Auto-detect" else selected_date_col
user_override = None if selected_user_col == "Auto-detect" else selected_user_col
cohort_override = None if selected_cohort_col == "Auto-detect" else selected_cohort_col
activity_override = None if selected_activity_col == "Auto-detect" else selected_activity_col

detections = []
if auto_date:
    detections.append(f"Date: {auto_date}")
if auto_user:
    detections.append(f"User: {auto_user}")
if auto_cohort:
    detections.append(f"Cohort: {auto_cohort}")
if auto_activity:
    detections.append(f"Activity: {auto_activity}")

if detections:
    st.caption("Auto-detected: " + " | ".join(detections))
else:
    st.caption("No columns auto-detected.")

st.subheader("Ask a question")
query = st.text_input(
    "Examples: how many rows, show columns, average sales, total profit, profit by category, sales by region, what was dau last week, which cohort has highest d7 retention?"
)

if st.button("Run query") and query.strip():
    result = handle_query(
        df, 
        query, 
        date_col_override=date_override, 
        user_col_override=user_override,
        cohort_date_override=cohort_override,
        activity_date_override=activity_override
    )

    if result.answer_type == "dataframe":
        st.success(result.message)
        st.dataframe(result.data, use_container_width=True)
        st.download_button(
            "Download result as CSV",
            result.data.to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
        )
    elif result.answer_type == "text":
        st.success(result.message)
        st.write(result.data)
    elif result.answer_type == "error":
        st.error(result.message)
    else:
        st.warning(result.message)

with st.expander("Describe dataset"):
    st.dataframe(describe_dataset(df), use_container_width=True)
