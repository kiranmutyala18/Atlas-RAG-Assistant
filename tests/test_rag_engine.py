from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rag_engine import RAGEngine


class RAGEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        base = Path(self.temp_dir.name)
        self.knowledge_dir = base / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        (self.knowledge_dir / "handbook.md").write_text(
            (
                "Engineering onboarding happens on Monday. "
                "Every new engineer gets access to product docs and shadow support."
            ),
            encoding="utf-8",
        )
        self.engine = RAGEngine(base / "rag.db", self.knowledge_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_bootstrap_loads_seed_documents(self) -> None:
        self.assertEqual(self.engine.document_count, 1)
        self.assertGreaterEqual(self.engine.chunk_count, 1)

    def test_ingest_document_updates_search_results(self) -> None:
        self.engine.ingest_document(
            "pricing.txt",
            "The Growth plan includes analytics, automations, and webhook integrations.",
        )

        results = self.engine.search("Which plan includes analytics?")

        self.assertTrue(results)
        self.assertEqual(results[0].document_name, "pricing.txt")

    def test_answer_persists_messages_to_session(self) -> None:
        session = self.engine.create_session("Demo")
        response = self.engine.answer(session["id"], "When does engineering onboarding happen?")

        self.assertIn("sources", response)
        self.assertEqual(len(response["messages"]), 2)
        self.assertEqual(response["messages"][0]["role"], "user")
        self.assertEqual(response["messages"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
