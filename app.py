import streamlit as st
import pandas as pd

from query_engine import process_query

# -----------------------------------
# Streamlit Page Config
# -----------------------------------

st.set_page_config(
    page_title="CSV Analytics QA Agent",
    layout="wide"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("Deterministic CSV Analytics QA Agent")

st.write(
    "Upload any CSV and ask analytics questions."
)

# -----------------------------------
# Upload CSV
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# -----------------------------------
# If CSV Uploaded
# -----------------------------------

if uploaded_file is not None:

    try:

        # Read CSV
        df = pd.read_csv(uploaded_file)

        # -----------------------------------
        # Dataset Preview
        # -----------------------------------

        st.subheader("Dataset Preview")

        st.dataframe(df.head())

        # -----------------------------------
        # Dataset Info
        # -----------------------------------

        st.subheader("Dataset Information")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Rows",
                len(df)
            )

        with col2:

            st.metric(
                "Columns",
                len(df.columns)
            )

        # -----------------------------------
        # Show Columns
        # -----------------------------------

        st.subheader("Detected Columns")

        st.write(df.columns.tolist())

        # -----------------------------------
        # Example Questions
        # -----------------------------------

        st.subheader("Example Questions")

        st.write("""
        - how many rows
        - average sales
        - total profit
        - highest sales
        - profit by category
        - sales by region
        - profit every month
        - sales this month vs last month
        - describe dataset
        """)

        # -----------------------------------
        # Question Input
        # -----------------------------------

        question = st.text_input(
            "Ask a question about your data"
        )

        # -----------------------------------
        # Ask Button
        # -----------------------------------

        if st.button("Ask Question"):

            answer = process_query(
                question,
                df
            )

            st.subheader("Answer")

            st.write(answer)

    except Exception as e:

        st.error(str(e))