import pandas as pd
from rapidfuzz import fuzz
from datetime import datetime

# -----------------------------------
# Detect column using fuzzy matching
# -----------------------------------

def find_best_column(question, df):

    best_score = 0
    best_col = None

    for col in df.columns:

        score = fuzz.partial_ratio(
            question.lower(),
            col.lower()
        )

        if score > best_score:

            best_score = score
            best_col = col

    if best_score > 60:
        return best_col

    return None

# -----------------------------------
# Detect date column
# -----------------------------------

def detect_date_column(df):

    for col in df.columns:

        if "date" in col.lower():
            return col

    return None

# -----------------------------------
# Main Query Engine
# -----------------------------------

def process_query(question, df):

    question = question.lower().strip()

    # -----------------------------------
    # Convert date column if exists
    # -----------------------------------

    date_col = detect_date_column(df)

    if date_col:

        try:

            df[date_col] = pd.to_datetime(df[date_col])

            df["month"] = df[date_col].dt.month_name()

            df["year"] = df[date_col].dt.year

        except:
            pass

    # -----------------------------------
    # ROW COUNT
    # -----------------------------------

    if "row" in question:

        return f"Dataset contains {len(df)} rows"

    # -----------------------------------
    # COLUMNS
    # -----------------------------------

    elif "column" in question:

        return list(df.columns)

    # -----------------------------------
    # SHOW DATA
    # -----------------------------------

    elif "show data" in question or "head" in question:

        return df.head().to_string()

    # -----------------------------------
    # SUMMARY
    # -----------------------------------

    elif "summary" in question or "describe" in question:

        return df.describe().to_string()

    # -----------------------------------
    # AVERAGE
    # -----------------------------------

    elif "average" in question or "mean" in question:

        col = find_best_column(question, df)

        if col:

            return round(df[col].mean(), 2)

        return "Column not found"

    # -----------------------------------
    # TOTAL / SUM
    # -----------------------------------

    elif "sum" in question or "total" in question:

        col = find_best_column(question, df)

        if col:

            return round(df[col].sum(), 2)

        return "Column not found"

    # -----------------------------------
    # MAXIMUM
    # -----------------------------------

    elif "highest" in question or "max" in question:

        col = find_best_column(question, df)

        if col:

            return df[col].max()

        return "Column not found"

    # -----------------------------------
    # MINIMUM
    # -----------------------------------

    elif "lowest" in question or "min" in question:

        col = find_best_column(question, df)

        if col:

            return df[col].min()

        return "Column not found"

    # -----------------------------------
    # UNIQUE VALUES
    # -----------------------------------

    elif "unique" in question:

        col = find_best_column(question, df)

        if col:

            return df[col].unique().tolist()

        return "Column not found"

    # -----------------------------------
    # GROUPBY ANALYTICS
    # Example:
    # sales by region
    # profit by category
    # -----------------------------------

    elif "by" in question:

        words = question.split("by")

        if len(words) == 2:

            value_text = words[0].strip()

            group_text = words[1].strip()

            value_col = find_best_column(
                value_text,
                df
            )

            group_col = find_best_column(
                group_text,
                df
            )

            if value_col and group_col:

                grouped = df.groupby(
                    group_col
                )[value_col].sum()

                return grouped.to_string()

    # -----------------------------------
    # MONTHLY ANALYTICS
    # Example:
    # profit every month
    # sales by month
    # -----------------------------------

    elif "month" in question:

        numeric_col = None

        for col in df.columns:

            if pd.api.types.is_numeric_dtype(df[col]):

                if col.lower() in question:

                    numeric_col = col
                    break

        if numeric_col and "month" in df.columns:

            grouped = df.groupby(
                "month"
            )[numeric_col].sum()

            return grouped.to_string()

    # -----------------------------------
    # THIS MONTH VS LAST MONTH
    # -----------------------------------

    elif "this month" in question and "last month" in question:

        if "month" not in df.columns:

            return "No date column found"

        numeric_col = find_best_column(question, df)

        if not numeric_col:

            return "Numeric column not found"

        current_month = datetime.now().month

        last_month = current_month - 1

        current_sales = df[
            df[date_col].dt.month == current_month
        ][numeric_col].sum()

        last_sales = df[
            df[date_col].dt.month == last_month
        ][numeric_col].sum()

        difference = current_sales - last_sales

        return {
            "this_month": round(current_sales, 2),
            "last_month": round(last_sales, 2),
            "difference": round(difference, 2)
        }

    return "Question is above my head"