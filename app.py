from __future__ import annotations

import cgi
import io
import json
import os
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from rag_engine import RAGEngine


BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))


def resolve_app_data_root() -> Path:
    candidates = []
    local_app_data = os.getenv("LOCALAPPDATA", "").strip()
    if local_app_data:
        candidates.append(Path(local_app_data) / "AtlasRagAssistant")
    candidates.append(Path(tempfile.gettempdir()) / "AtlasRagAssistant")
    candidates.append(BASE_DIR / "runtime_data")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue

    raise OSError("Unable to create an application data directory")


APP_DATA_ROOT = resolve_app_data_root()
UPLOADS_DIR = APP_DATA_ROOT / "uploads"
RUNTIME_DIR = APP_DATA_ROOT / "runtime"
DATABASE_PATH = RUNTIME_DIR / "atlas.db"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
rag = RAGEngine(DATABASE_PATH, KNOWLEDGE_DIR)


class RAGRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return

        if path.startswith("/static/"):
            relative_path = path.removeprefix("/static/")
            file_path = STATIC_DIR / relative_path
            self._serve_file(file_path, self._guess_content_type(file_path.suffix))
            return

        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "documents": rag.document_count,
                    "chunks": rag.chunk_count,
                    "sessions": rag.session_count,
                    "generation_mode": "openai" if rag.llm_enabled else "local",
                }
            )
            return

        if path == "/api/search":
            params = parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            self._send_json(
                {"query": query, "results": [item.to_dict() for item in rag.search(query)]}
            )
            return

        if path == "/api/sessions":
            self._send_json({"sessions": rag.list_sessions()})
            return

        if path.startswith("/api/sessions/") and path.endswith("/messages"):
            session_id = self._extract_session_id(path)
            self._send_json({"messages": rag.get_messages(session_id)})
            return

        if path == "/api/documents":
            self._send_json({"documents": rag.list_documents()})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/api/chat":
                payload = self._read_json_body()
                session_id = int(payload.get("session_id", 0))
                message = str(payload.get("message", "")).strip()
                self._send_json(rag.answer(session_id, message))
                return

            if parsed.path == "/api/sessions":
                payload = self._read_json_body(optional=True)
                self._send_json(rag.create_session(payload.get("title")))
                return

            if parsed.path == "/api/documents/upload":
                self._handle_upload()
                return
        except ValueError as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Route not found")

    def log_message(self, format: str, *args) -> None:
        return

    def _handle_upload(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            raise ValueError("Upload endpoint expects multipart/form-data")

        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length)
        environ = {"REQUEST_METHOD": "POST"}
        headers = {"content-type": content_type}
        form = cgi.FieldStorage(
            fp=io.BytesIO(raw_body),
            headers=headers,
            environ=environ,
        )

        file_item = form["file"] if "file" in form else None
        if not file_item or not getattr(file_item, "filename", ""):
            raise ValueError("A file is required")

        filename = Path(file_item.filename).name
        if Path(filename).suffix.lower() not in {".txt", ".md"}:
            raise ValueError("Only .txt and .md uploads are supported")

        raw_content = file_item.file.read()
        text_content = raw_content.decode("utf-8")
        saved_path = UPLOADS_DIR / filename
        saved_path.write_text(text_content, encoding="utf-8")
        result = rag.ingest_document(filename, text_content, source_type="upload")
        self._send_json({"uploaded": result, "documents": rag.list_documents()})

    def _read_json_body(self, optional: bool = False) -> dict:
        content_length = int(self.headers.get("Content-Length", "0"))
        if optional and content_length == 0:
            return {}

        raw_body = self.rfile.read(content_length)
        try:
            return json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Invalid JSON payload") from error

    def _extract_session_id(self, path: str) -> int:
        parts = [part for part in path.strip("/").split("/") if part]
        if len(parts) < 4:
            raise ValueError("Invalid session route")
        return int(parts[2])

    def _serve_file(self, file_path: Path, content_type: str) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return

        content = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _guess_content_type(suffix: str) -> str:
        return {
            ".css": "text/css; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
        }.get(suffix, "application/octet-stream")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), RAGRequestHandler)
    print(f"RAG chatbot running at http://{HOST}:{PORT}")
    print(
        "Loaded "
        f"{rag.document_count} documents, {rag.chunk_count} chunks, "
        f"and {rag.session_count} chat sessions"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
