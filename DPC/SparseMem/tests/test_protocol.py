"""Independent invariants for experiment accounting, prompts and fairness."""

import json
import re
import tempfile
import unittest
from pathlib import Path
from sparsemem.construct import DEFAULT, PROMPTS, Ledger
from sparsemem.history import project, segment
from sparsemem.dependency import prepare
from sparsemem.recall import keys, retrieve
from sparsemem.run import FixtureEmbedder, FixtureTokenizer, Index, usage


class ProtocolTests(unittest.TestCase):
    def test_document_prompt_identity(self):
        design = (
            Path(__file__).resolve().parents[1] / "SparseMem_Codex_Implementation.md"
        ).read_text()
        for tag, name in [("P1", "refine"), ("P3", "answer")]:
            section = design.split("### " + tag + ".")[1].split("\n### ")[0]
            system, user = re.findall(r"```text\n(.*?)\n```", section, re.S)[:2]
            self.assertEqual(system, PROMPTS[name + "_system"])
            self.assertEqual(user, PROMPTS[name + "_user"])

    def test_cheaper_request_after_rejection_and_overrun(self):
        tok = FixtureTokenizer()
        small = [dict(role="user", content="Hello")]
        big = [dict(role="user", content="Test " * 300)]
        limit = tok.messages(small) + 11
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp) / "ledger.json")
            calls = []

            def send(*args):
                calls.append(1)
                return "ok", dict(prompt_tokens=limit + 3, completion_tokens=1)

            self.assertIsNone(
                ledger.generate("big", "construct", big, 10, 1, limit, tok, send)
            )
            ledger.generate("small", "construct", small, 10, 1, limit, tok, send)
            self.assertIsNone(
                ledger.generate("next", "construct", small, 10, 1, limit, tok, send)
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(ledger.spent("construct"), limit + 4)

    def test_retrieval_preserves_distinct_repeated_advice(self):
        turns = [
            dict(
                id=f"S{i}:M1",
                session=f"S{i}",
                role="assistant",
                date=f"2024-03-0{i}",
                text="Consider switching to the evening class.",
            )
            for i in (1, 2)
        ]
        tok = FixtureTokenizer()
        enc = FixtureEmbedder()
        a, blocks = segment(turns, tok, enc, DEFAULT, "raw_block")
        records = prepare(a, blocks, tok, DEFAULT, "raw_block")
        entries = keys(records, a)
        index = Index(entries, enc.embed([e["text"] for e in entries]))
        result = retrieve(
            index,
            enc,
            records,
            a,
            "Which class did the assistant suggest?",
            "",
            DEFAULT,
            tok,
        )
        self.assertEqual(set(result["packed"]), set(a))
        self.assertEqual(result["evidence"].count(" | assistant | "), 2)
        self.assertNotIn(" | user | ", result["evidence"])
        self.assertLess(
            result["evidence"].index("2024-03-01"),
            result["evidence"].index("2024-03-02"),
        )
        tiny = retrieve(
            index,
            enc,
            records,
            a,
            "Which class?",
            "",
            dict(DEFAULT, evidence_tokens=1),
            tok,
        )
        self.assertEqual(tiny["packed"], [])
