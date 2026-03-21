# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Course Materials RAG (Retrieval-Augmented Generation) System** — a full-stack web application that answers questions about course materials using semantic search and Claude AI.

**Stack:**
- **Backend:** FastAPI (Python 3.13+) with Uvicorn, running on port 8000
- **Frontend:** Vanilla JavaScript with HTML/CSS served as static files
- **Vector Storage:** ChromaDB (local persistent database)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **LLM:** Anthropic Claude API (claude-sonnet-4-20250514)
- **Package Manager:** uv (Python)

## Architecture

### Core Components (Backend)

**RAGSystem** (`rag_system.py`) — Main orchestrator:
- `add_course_document(file_path)` — Processes a single course file
- `add_course_folder(folder_path)` — Batch loads all courses from `../docs` directory
- `query(query, session_id)` — Processes user queries and returns answers + sources
- `get_course_analytics()` — Returns course catalog statistics

**Key subsystems:**

1. **DocumentProcessor** (`document_processor.py`)
   - Parses PDF, DOCX, TXT files into Course objects (title, lessons, content)
   - Chunks text into CHUNK_SIZE (800 chars) with CHUNK_OVERLAP (100 chars)
   - Creates `CourseChunk` objects with metadata

2. **VectorStore** (`vector_store.py`) — ChromaDB wrapper
   - Stores course metadata (course-level search)
   - Stores course content chunks (semantic search across all lessons)
   - `add_course_metadata()` and `add_course_content()` for ingestion
   - `search()` returns top 5 matching chunks + sources

3. **AIGenerator** (`ai_generator.py`) — Claude interface
   - Calls Claude API with tools and conversation history
   - Uses agent-like agentic loop: Claude can call search tool, process results, refine answer
   - System prompt instructs Claude to search only for course-specific content
   - Maintains temperature=0 for consistency, max_tokens=800

4. **ToolManager + CourseSearchTool** (`search_tools.py`)
   - `CourseSearchTool` wraps VectorStore.search() as a Claude tool
   - Allows Claude to trigger semantic search during response generation
   - Tracks sources for each tool call (returned to frontend)

5. **SessionManager** (`session_manager.py`)
   - Tracks conversation history per session_id
   - Stores last MAX_HISTORY (2) exchanges
   - Used to provide context for multi-turn conversations

### Data Flow

```
User Query (API)
  ↓
RAGSystem.query()
  ↓
AIGenerator.generate_response()
  ↓
Claude API (with tools)
  ├─ Claude may call → CourseSearchTool.search()
  │                     ↓
  │                  VectorStore.search() (ChromaDB semantic search)
  │                     ↓
  │                  ToolManager.register_sources()
  └─ Claude synthesizes → Response
  ↓
SessionManager.add_exchange() (if session_id provided)
  ↓
API Response (answer + sources + session_id)
```

### Frontend

Static HTML/CSS/JS served from `/frontend`:
- `index.html` — UI structure
- `script.js` — Handles API calls to `/api/query` and `/api/courses`
- `style.css` — Styling

The frontend makes POST requests to `/api/query` with `{ query, session_id }` and receives `{ answer, sources, session_id }`.

## Configuration

**Config** (`backend/config.py`):
- `ANTHROPIC_API_KEY` — Read from `.env` (required)
- `ANTHROPIC_MODEL` — Currently claude-sonnet-4-20250514
- `EMBEDDING_MODEL` — all-MiniLM-L6-v2 (for sentence-transformers)
- `CHUNK_SIZE` — 800 characters
- `CHUNK_OVERLAP` — 100 characters
- `MAX_RESULTS` — 5 search results
- `MAX_HISTORY` — 2 conversation exchanges
- `CHROMA_PATH` — `./chroma_db` (persisted locally)

## Common Development Tasks

### Setup

```bash
# Install dependencies
uv sync

# Create .env file with your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

### Running the Application

```bash
# Quick start (runs both backend and frontend)
chmod +x run.sh
./run.sh

# Or manually from backend directory
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application is then available at:
- **Web UI:** `http://localhost:8000`
- **API Docs:** `http://localhost:8000/docs` (Swagger)

### Adding Course Materials

Place PDF, DOCX, or TXT files in the `docs/` directory. They're automatically loaded on startup via the `startup_event()` in `app.py`, which calls `RAGSystem.add_course_folder()`.

To programmatically add documents:
```python
from backend.rag_system import RAGSystem
from backend.config import config

rag = RAGSystem(config)
courses, chunks = rag.add_course_folder("./docs", clear_existing=False)
print(f"Loaded {courses} courses with {chunks} chunks")
```

### API Endpoints

- **POST `/api/query`** — Query course materials
  - Request: `{ "query": "string", "session_id": "optional-string" }`
  - Response: `{ "answer": "string", "sources": ["list", "of", "sources"], "session_id": "string" }`

- **GET `/api/courses`** — Get course statistics
  - Response: `{ "total_courses": int, "course_titles": ["list", "of", "titles"] }`

### Debugging

1. **Check ChromaDB:** The vector store persists in `backend/chroma_db/`. Delete it to reset all ingested documents.

2. **Claude tool calls:** `AIGenerator.generate_response()` logs when tools are invoked. The agentic loop stops after Claude produces text content.

3. **Session history:** Each session stores conversation exchanges. View via `SessionManager.get_conversation_history(session_id)`.

4. **API responses:** Enable logging in FastAPI by checking the console output from `uvicorn`. The `/docs` endpoint provides interactive testing.

## Key Implementation Details

- **No external database:** All data is in ChromaDB (local file) + conversation history in memory.
- **Agent-based querying:** Claude can choose to search the course database via tools rather than relying on pre-retrieved context. This allows flexible, multi-step reasoning.
- **No caching of embeddings:** Each query re-embeds the user query for semantic search (acceptable for this use case).
- **Stateless API (except sessions):** Each query can be independent; sessions are optional for multi-turn context.

## Notable Code Patterns

- **Config dataclass** — Single source of truth for all settings (imported as `config` throughout)
- **Resource tracker suppression** — `app.py` suppresses ChromaDB multiprocessing warnings
- **CORS/TrustedHost middleware** — Allows frontend to call backend from any origin
- **Static file caching prevention** — `DevStaticFiles` class adds no-cache headers for development (currently unused but available)
