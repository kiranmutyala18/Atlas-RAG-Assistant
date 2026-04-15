from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"[a-zA-Z0-9']+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def keywords(text: str) -> list[str]:
    filtered = [token for token in tokenize(text) if token not in STOPWORDS]
    return filtered or tokenize(text)


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


@dataclass
class SearchResult:
    chunk_id: int
    document_name: str
    content: str
    score: float

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source": self.document_name,
            "text": self.content,
            "score": round(self.score, 4),
        }


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-3-small",
        chat_model: str = "gpt-4o-mini",
    ) -> None:
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.chat_model = chat_model

    def embed(self, text: str) -> list[float]:
        payload = {"input": text, "model": self.embedding_model}
        response = self._post_json("https://api.openai.com/v1/embeddings", payload)
        return response["data"][0]["embedding"]

    def answer(self, question: str, context_chunks: list[SearchResult]) -> str:
        context = "\n\n".join(
            f"Source: {chunk.document_name}\nContent: {chunk.content}"
            for chunk in context_chunks
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a retrieval-augmented assistant. Answer only using the "
                    "provided context. If the context is insufficient, say so clearly. "
                    "Cite source names inline when relevant."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Retrieved context:\n{context}"
                ),
            },
        ]
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "temperature": 0.2,
        }
        response = self._post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
        )
        return response["choices"][0]["message"]["content"].strip()

    def _post_json(self, url: str, payload: dict) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode("utf-8"))


class RAGEngine:
    def __init__(self, database_path: Path, knowledge_dir: Path):
        self.database_path = database_path
        self.knowledge_dir = knowledge_dir
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.client = self._build_openai_client()
        self._initialize_database()
        self.bootstrap_knowledge_base()

    @property
    def document_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS count FROM documents")
        return int(row["count"])

    @property
    def chunk_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS count FROM chunks")
        return int(row["count"])

    @property
    def session_count(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS count FROM sessions")
        return int(row["count"])

    @property
    def llm_enabled(self) -> bool:
        return self.client is not None

    def bootstrap_knowledge_base(self) -> None:
        for file_path in sorted(self.knowledge_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in {".md", ".txt"}:
                continue
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue
            existing = self._fetchone(
                """
                SELECT documents.id AS id, documents.content AS content, COUNT(chunks.id) AS chunk_count
                FROM documents
                LEFT JOIN chunks ON chunks.document_id = documents.id
                WHERE documents.name = ?
                GROUP BY documents.id
                """,
                (file_path.name,),
            )
            if (
                existing
                and existing["content"] == content
                and int(existing["chunk_count"]) > 0
            ):
                continue
            self.ingest_document(file_path.name, content, source_type="seed")

    def create_session(self, title: str | None = None) -> dict:
        session_title = title.strip() if title else "New chat"
        created_at = int(time.time())
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO sessions (title, created_at, updated_at)
                VALUES (?, ?, ?)
                """,
                (session_title, created_at, created_at),
            )
            session_id = int(cursor.lastrowid)
        return {"id": session_id, "title": session_title, "created_at": created_at}

    def list_sessions(self) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT id, title, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC, id DESC
            """
        )
        return [dict(row) for row in rows]

    def get_messages(self, session_id: int) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        )
        return [dict(row) for row in rows]

    def list_documents(self) -> list[dict]:
        rows = self._fetchall(
            """
            SELECT id, name, source_type, created_at
            FROM documents
            ORDER BY created_at DESC, id DESC
            """
        )
        return [dict(row) for row in rows]

    def ingest_document(self, name: str, content: str, source_type: str = "upload") -> dict:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Document content is empty")

        existing = self._fetchone(
            "SELECT id FROM documents WHERE name = ?",
            (name,),
        )
        created_at = int(time.time())
        chunks = chunk_text(normalized_content)
        if not chunks:
            raise ValueError("Document does not contain enough text to ingest")

        with self._connection() as connection:
            if existing:
                document_id = int(existing["id"])
                connection.execute(
                    "DELETE FROM chunks WHERE document_id = ?",
                    (document_id,),
                )
                connection.execute(
                    """
                    UPDATE documents
                    SET content = ?, source_type = ?, created_at = ?
                    WHERE id = ?
                    """,
                    (normalized_content, source_type, created_at, document_id),
                )
            else:
                cursor = connection.execute(
                    """
                    INSERT INTO documents (name, content, source_type, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (name, normalized_content, source_type, created_at),
                )
                document_id = int(cursor.lastrowid)

            for index, chunk in enumerate(chunks):
                embedding = self._safe_embed(chunk)
                connection.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        document_id,
                        index,
                        chunk,
                        json.dumps(embedding) if embedding else None,
                    ),
                )

        return {"document_id": document_id, "name": name, "chunks": len(chunks)}

    def answer(self, session_id: int, question: str) -> dict:
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question is required")

        self._require_session(session_id)
        self._add_message(session_id, "user", cleaned_question)
        results = self.search(cleaned_question, limit=4)

        if not results:
            answer_text = (
                "I could not find enough support in the knowledge base to answer that yet. "
                "Please upload more relevant documents and try again."
            )
        elif self.client is not None:
            try:
                answer_text = self.client.answer(cleaned_question, results)
            except (
                urllib.error.URLError,
                TimeoutError,
                KeyError,
                IndexError,
                json.JSONDecodeError,
            ):
                answer_text = self._fallback_answer(cleaned_question, results)
        else:
            answer_text = self._fallback_answer(cleaned_question, results)

        self._add_message(session_id, "assistant", answer_text)
        return {
            "answer": answer_text,
            "sources": [item.to_dict() for item in results],
            "messages": self.get_messages(session_id),
            "mode": "openai" if self.client is not None else "local",
        }

    def search(self, query: str, limit: int = 4) -> list[SearchResult]:
        cleaned_query = query.strip()
        if not cleaned_query:
            return []

        rows = self._fetchall(
            """
            SELECT
                chunks.id AS chunk_id,
                chunks.content AS content,
                chunks.embedding_json AS embedding_json,
                documents.name AS document_name
            FROM chunks
            JOIN documents ON documents.id = chunks.document_id
            """
        )

        query_terms = Counter(keywords(cleaned_query))
        query_embedding = self._safe_embed(cleaned_query)
        results: list[SearchResult] = []

        for row in rows:
            lexical = self._lexical_score(query_terms, row["content"])
            semantic = 0.0
            if query_embedding and row["embedding_json"]:
                chunk_embedding = json.loads(row["embedding_json"])
                semantic = max(0.0, cosine_similarity(query_embedding, chunk_embedding))

            total = (lexical * 0.55) + (semantic * 0.45)
            if total <= 0:
                continue

            results.append(
                SearchResult(
                    chunk_id=int(row["chunk_id"]),
                    document_name=row["document_name"],
                    content=row["content"],
                    score=total,
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def _fallback_answer(self, question: str, results: list[SearchResult]) -> str:
        summary = "\n".join(
            f"- {item.document_name}: {self._trim_chunk(item.content)}"
            for item in results
        )
        return (
            f"Question: {question}\n\n"
            "Grounded response assembled from the retrieved knowledge base:\n"
            f"{summary}\n\n"
            "If you connect an OpenAI API key, this app will turn the same retrieved context "
            "into a more natural answer while keeping citations."
        )

    def _lexical_score(self, query_terms: Counter[str], content: str) -> float:
        content_terms = keywords(content)
        if not content_terms:
            return 0.0

        content_counts = Counter(content_terms)
        overlap = set(query_terms) & set(content_counts)
        if not overlap:
            return 0.0

        weighted_overlap = sum(
            math.log1p(content_counts[term]) * query_terms[term]
            for term in overlap
        )
        coverage = len(overlap) / max(1, len(set(query_terms)))
        return (weighted_overlap + coverage) / math.log1p(len(content_terms))

    def _trim_chunk(self, text: str, limit: int = 240) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _safe_embed(self, text: str) -> list[float] | None:
        if self.client is None:
            return None
        try:
            return self.client.embed(text)
        except (
            urllib.error.URLError,
            TimeoutError,
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ):
            return None

    def _require_session(self, session_id: int) -> None:
        row = self._fetchone(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        )
        if row is None:
            raise ValueError("Session not found")

    def _add_message(self, session_id: int, role: str, content: str) -> None:
        created_at = int(time.time())
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, created_at),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (created_at, session_id),
            )

    def _build_openai_client(self) -> OpenAIClient | None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return None
        return OpenAIClient(api_key=api_key)

    def _initialize_database(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT,
                    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = MEMORY")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    @contextmanager
    def _connection(self) -> Iterable[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _fetchone(self, query: str, params: Iterable | None = None) -> sqlite3.Row | None:
        with self._connection() as connection:
            cursor = connection.execute(query, tuple(params or ()))
            return cursor.fetchone()

    def _fetchall(self, query: str, params: Iterable | None = None) -> list[sqlite3.Row]:
        with self._connection() as connection:
            cursor = connection.execute(query, tuple(params or ()))
            return cursor.fetchall()
