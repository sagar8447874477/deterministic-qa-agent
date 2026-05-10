import pandas as pd

# -----------------------------------
# Detect column
# -----------------------------------

def detect_column(question, df):

    question = question.lower()

    for col in df.columns:

        if col.lower() in question:

            return col

    return None

# -----------------------------------
# Main Query Engine
# -----------------------------------

def process_query(question, df):

    question = question.lower()

    # -----------------------------------
    # Total rows
    # -----------------------------------

    if "rows" in question:

        return f"Dataset contains {len(df)} rows"

    # -----------------------------------
    # Columns
    # -----------------------------------

    elif "columns" in question:

        return list(df.columns)

    # -----------------------------------
    # Show Data
    # -----------------------------------

    elif "show data" in question or "head" in question:

        return df.head().to_string()

    # -----------------------------------
    # Dataset Summary
    # -----------------------------------

    elif "summary" in question or "describe" in question:

        return df.describe().to_string()

    # -----------------------------------
    # Average
    # -----------------------------------

    elif "average" in question or "mean" in question:

        col = detect_column(question, df)

        if col:

            return df[col].mean()

        return "Column not found"

    # -----------------------------------
    # Sum
    # -----------------------------------

    elif "sum" in question or "total" in question:

        col = detect_column(question, df)

        if col:

            return df[col].sum()

        return "Column not found"

    # -----------------------------------
    # Maximum
    # -----------------------------------

    elif "max" in question or "highest" in question:

        col = detect_column(question, df)

        if col:

            return df[col].max()

        return "Column not found"

    # -----------------------------------
    # Minimum
    # -----------------------------------

    elif "min" in question or "lowest" in question:

        col = detect_column(question, df)

        if col:

            return df[col].min()

        return "Column not found"

    # -----------------------------------
    # Unique values
    # -----------------------------------

    elif "unique" in question:

        col = detect_column(question, df)

        if col:

            return df[col].unique().tolist()

        return "Column not found"

    # -----------------------------------
    # Group By
    # Example:
    # sales by region
    # -----------------------------------

    elif "by" in question:

        words = question.split("by")

        if len(words) == 2:

            value_col = words[0].strip().split()[-1]

            group_col = words[1].strip()

            actual_value_col = None
            actual_group_col = None

            for col in df.columns:

                if col.lower() == value_col:
                    actual_value_col = col

                if col.lower() == group_col:
                    actual_group_col = col

            if actual_value_col and actual_group_col:

                result = df.groupby(
                    actual_group_col
                )[actual_value_col].sum()

                return result.to_string()

    return "Question not supported yet"