# Deterministic CSV Analytics QA Agent

A fully local AI-style analytics assistant that allows users to upload CSV files and ask natural language questions about their data using a deterministic pandas-based query engine.

Built with:
- Streamlit
- Pandas
- RapidFuzz
- Python

No OpenAI APIs. No cloud LLMs. Fully offline-capable.

---

# Features

## CSV Upload
Users can upload any CSV dataset dynamically.

## Natural Language Analytics
Ask questions like:

- how many rows
- show columns
- average sales
- total profit
- highest revenue
- lowest employees
- sales by region
- profit by category
- profit every month
- sales this month vs last month
- describe dataset

## Deterministic Query Engine
Questions are converted into pandas operations using:
- intent detection
- fuzzy matching
- dynamic column detection
- aggregation logic
- groupby analytics

## Fuzzy NLP Matching
Supports typo-tolerant queries:

Examples:
- avrage sales
- higest revenue
- totl profit

## Dynamic Dataset Support
Works with any CSV schema without hardcoding column names.

## Offline Capability
Runs completely locally without internet after installation.

---

# Tech Stack

| Component | Technology |
|---|---|
| Frontend/UI | Streamlit |
| Analytics Engine | Pandas |
| NLP Layer | RapidFuzz |
| Backend Logic | Python |
| Query Processing | Deterministic Rule Engine |

---

# Project Structure

```bash
deterministic-qa-agent/
│
├── app.py
├── query_engine.py
├── requirements.txt
└── README.md
