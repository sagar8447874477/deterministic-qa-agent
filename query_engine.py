import pandas as pd

# -----------------------------------
# Detect column name from question
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

    question = question.lower().strip()

    print("QUESTION:", question)

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

        col = detect_column(question, df)

        if col:

            return df[col].mean()

        return "Column not found"

    # -----------------------------------
    # SUM / TOTAL
    # -----------------------------------

    elif "sum" in question or "total" in question:

        col = detect_column(question, df)

        if col:

            return df[col].sum()

        return "Column not found"

    # -----------------------------------
    # MAXIMUM
    # -----------------------------------

    elif "max" in question or "highest" in question:

        col = detect_column(question, df)

        if col:

            return df[col].max()

        return "Column not found"

    # -----------------------------------
    # MINIMUM
    # -----------------------------------

    elif "min" in question or "lowest" in question:

        col = detect_column(question, df)

        if col:

            return df[col].min()

        return "Column not found"

    # -----------------------------------
    # UNIQUE VALUES
    # -----------------------------------

    elif "unique" in question:

        col = detect_column(question, df)

        if col:

            return df[col].unique().tolist()

        return "Column not found"

    # -----------------------------------
    # GROUPBY
    # Example:
    # sales by region
    # -----------------------------------

    elif "by" in question:

        words = question.split("by")

        if len(words) == 2:

            value_col = words[0].strip().split()[-1]

            group_col = words[1].strip()

            print("VALUE COL:", value_col)
            print("GROUP COL:", group_col)

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
