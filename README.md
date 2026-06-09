# BookDB — Interactive Showcase

An interactive static showcase for **BookDB**, a full-stack book discovery platform that combines
collaborative filtering, semantic search, and a conversational AI to recommend books you'll
actually want to read.

🔗 **Live site:** https://andreaisabelmontana.github.io/bookdb-discovery/

## What it does
- **Conversational recommendations** — tell the chatbot your mood; it rewrites the query and routes tools (search, recommend, review RAG).
- **Multiple ML models** — BPR, SAR, and NCF collaborative filtering, plus vector similarity.
- **Fusion + RRF** — outputs combined with weighted scoring and reciprocal-rank-fusion reranking.
- **Explains itself** — says *why* it recommends a book, drawing from real user reviews.
- **MCP server** — a Go-based Model Context Protocol server exposes tools to Claude Desktop, Cursor, etc.

Built on **229M Goodreads interactions** across a **2.3M-book** catalogue.

**Stack:** Python (`bookdb` library) · FastAPI (REST + SSE) · React · BPR/SAR/NCF · vector DB · Marimo + MLflow · Go MCP server.

## About this repo
An original, hand-built static site (single `index.html`, no framework) presenting the project, with a
scripted interactive chat-recommender demo over a small sample catalog (titles real, blurbs illustrative).
Built from scratch.
