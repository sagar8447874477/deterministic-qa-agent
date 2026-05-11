import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import streamlit as st

from query_engine import (
    handle_query, 
    describe_dataset, 
    infer_datetime_column, 
    infer_user_column,
    infer_cohort_date_column,
    infer_activity_date_column
)


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
