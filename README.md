# SIC/SOC RAG Classifier

An end-to-end RAG pipeline for automated SIC/SOC occupation and industry coding.

## Architecture
1. **Ingest** — ONS SIC 2007 code descriptions embedded via OpenAI and stored in ChromaDB
2. **Retrieve** — Vector similarity search returns top-n candidate codes for a given text
3. **Classify** — LLM selects best match with confidence scoring and agentic retry logic
4. **Observe** — All calls traced via Langfuse for latency, token usage, and eval metrics

## Stack
Python · LangChain · OpenAI API · ChromaDB · Langfuse · Streamlit · Docker

## Setup
1. Clone the repo and activate a virtual environment
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and add your API keys
4. Run `python ingest.py` to build the vector store
5. Run `streamlit run app.py` to launch the UI

## Live Demo
[Coming soon]