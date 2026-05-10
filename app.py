import streamlit as st
import pandas as pd

from query_engine import process_query

# -----------------------------------
# Page Title
# -----------------------------------

st.title("Deterministic CSV Analytics QA Agent")

# -----------------------------------
# Upload CSV
# -----------------------------------

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

# -----------------------------------
# If CSV Uploaded
# -----------------------------------

if uploaded_file is not None:

    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Show preview
    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # Show columns
    st.subheader("Detected Columns")

    st.write(df.columns.tolist())

    # Ask Question
    question = st.text_input(
        "Ask a question about your dataset"
    )

    # Process Question
    if st.button("Ask"):

        answer = process_query(question, df)

        st.subheader("Answer")

        st.write(answer)