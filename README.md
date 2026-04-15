# Atlas RAG Assistant

Atlas is a full-stack retrieval-augmented chatbot for exploring private knowledge bases through grounded conversations.
Full-stack RAG assistant with document upload, retrieval, citations, and persistent chat sessions.
It is built as a portfolio-quality project, not a one-shot demo:

- document ingestion from the UI
- SQLite-backed persistence for documents, chunks, sessions, and messages
- retrieval over chunked knowledge-base content
- source citations on each answer
- optional OpenAI-powered embeddings and answer generation
- clean browser interface for live demo use

## Why this project is strong

Atlas demonstrates the kind of end-to-end engineering product companies care about:

- backend API design
- storage and persistence
- retrieval system design
- full-stack UI work
- AI product integration
- error handling and fallback behavior
- testing for core logic

## What RAG means

RAG stands for Retrieval-Augmented Generation.

Instead of asking a model to answer from memory alone, the app:

1. stores your documents
2. splits them into chunks
3. retrieves the most relevant chunks for a user question
4. generates an answer grounded in those retrieved chunks

That grounding is what makes the chatbot useful for company docs, support content, policies, and internal knowledge.

## Quick start

1. Make sure Python 3.11+ is installed.
2. Run the app:

```powershell
python app.py
```

3. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Demo highlights

During a live demo, you can show:

- preloaded knowledge-base documents
- file upload and ingestion
- multi-session chat history
- grounded answers with citations
- local-only fallback mode
- OpenAI-enhanced mode with the same retrieval pipeline

## OpenAI mode

The app works without any API key using local retrieval plus grounded fallback responses.

To enable OpenAI-backed generation and embeddings:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python app.py
```

When the key is present, the app will:

- create embeddings for ingested chunks
- use those embeddings during retrieval
- generate more natural final answers from retrieved context

## Features

- Upload `.txt` and `.md` files from the UI
- Persist documents and chat sessions in SQLite
- Retrieve relevant document chunks per question
- Show grounded source pills for every response
- Keep multiple chat sessions for demo flow
- Fall back gracefully when an LLM is not configured

## Architecture

1. Documents are loaded from the seed knowledge base or uploaded through the UI.
2. Content is chunked and stored in SQLite.
3. Retrieval ranks chunks with lexical scoring and optional embedding similarity.
4. The assistant returns a grounded answer and attaches the most relevant sources.
5. Chats are persisted so sessions remain available across refreshes.

## Project structure

- `app.py` - HTTP server and API routes
- `rag_engine.py` - storage, ingestion, retrieval, and answer generation logic
- `knowledge_base/` - seed documents loaded at startup
- `static/` - frontend assets
- runtime data is stored outside the main source tree when possible and otherwise falls back to `runtime_data/`

## Demo flow

For a live demo:

1. Start the server
2. Show the seeded knowledge base loading
3. Upload a new markdown or text document
4. Create a new chat session
5. Ask a question tied to the uploaded document
6. Point out the source-backed response and session history

## Running tests

```powershell
$env:PYTHONDONTWRITEBYTECODE="1"
python -m unittest discover -s tests
```

## Resume positioning

Good framing for internships and product companies:

- Built a full-stack RAG assistant with document ingestion, retrieval, citations, and persistent chat sessions
- Implemented a retrieval pipeline with chunking, lexical search, optional embeddings, and grounded answer generation
- Designed a demo-ready web product using Python, SQLite, and vanilla JavaScript without heavy frameworks

## GitHub checklist

Before publishing:

- add screenshots or a short demo GIF to this README
- add your GitHub repo URL and live demo URL once deployed
- set `OPENAI_API_KEY` locally if you want to demo LLM-backed answers
- keep runtime artifacts out of Git using the included `.gitignore`

## Next upgrades

- add PDF/docx parsing
- add auth and per-user workspaces
- deploy to Render/Railway/Fly
- add analytics and evaluation datasets
- swap SQLite search for a dedicated vector database at larger scale
