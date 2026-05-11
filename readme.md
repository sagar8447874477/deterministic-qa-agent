# Deterministic CSV Analytics QA Agent

A fully local Streamlit app that lets users upload a CSV and ask questions in plain English.  
It uses Pandas for analytics and RapidFuzz for fuzzy matching, with no OpenAI or cloud APIs.

## What it does
Users can ask:
- how many rows
- show columns
- average sales
- total profit
- profit by category
- sales by region
- highest revenue
- lowest employees
- dau last week
- wau this month
- mau past 30 days

## Why I built it
I wanted a CSV analytics tool that is:
- offline-capable
- deterministic
- easy to use
- flexible for different CSV schemas

## Tech stack
- **Streamlit** for the UI and file upload
- **Pandas** for grouping, filtering, and date-based analytics
- **RapidFuzz** for typo-tolerant fuzzy matching
- **Python** for the rule-based query engine

## How it works
1. Upload a CSV file.
2. The app reads it into a Pandas DataFrame.
3. The query engine detects the user intent.
4. It matches likely columns using fuzzy matching.
5. It runs deterministic Pandas operations.
6. Results are shown in the app and can be downloaded.

## Major improvements made
- Added support for general aggregation queries like `profit by category`.
- Added DAU / WAU / MAU support with date filtering.
- Added fuzzy column detection for better schema flexibility.
- Added manual date and user column selection in the UI.
- Kept the repo flat with just `app.py` and `query_engine.py` in the root.
- Removed all cloud AI dependencies.

## Project structure
```text
deterministic-qa-agent/
├── app.py
├── query_engine.py
├── requirements.txt
└── README.md
```

## Example questions
- how many rows
- show columns
- profit by category
- sales by region
- dau last week

## Requirements
```txt
streamlit
pandas
rapidfuzz
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes
This app is deterministic, so it works best on structured CSVs with clear columns for dates, users, and numeric metrics.
