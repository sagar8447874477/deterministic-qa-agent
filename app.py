import os
import sys

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from query_engine import handle_query, describe_dataset


st.set_page_config(page_title="Deterministic CSV QA Agent", layout="wide")

st.title("Deterministic CSV Analytics QA Agent")

st.write("Upload a CSV and ask questions like: how many rows, show columns, average sales, or what was dau last week?")

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

st.subheader("Ask a question")
query = st.text_input(
    "Examples: how many rows, show columns, average sales, total profit, what was dau last week?"
)

if st.button("Run query") and query.strip():
    result = handle_query(df, query)

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
