import tempfile
import unittest
from pathlib import Path
from sparsemem.construct import construct, Ledger, DEFAULT
from sparsemem.dependency import prepare
from sparsemem.run import FixtureTokenizer, mock


class Levels(unittest.TestCase):
    def route(self, texts, *, blocks=None, budget=16000):
        archive = {
            str(i): dict(text=text, role=role, session=session, date=date, order=i)
            for i, (text, role, session, date) in enumerate(texts)
        }
        bounds = blocks or [[k] for k in archive]
        cfg = dict(DEFAULT, budget=budget)
        tok = FixtureTokenizer()
        rows = prepare(archive, bounds, tok, cfg, "sparse")
        with tempfile.TemporaryDirectory() as d:
            construct(
                rows, archive, cfg, "sparse", 42, tok, Ledger(Path(d) / "l"), mock
            )
        return rows

    def test_absolute_and_relative(self):
        rows = self.route(
            [
                (t, "user", "S1", "2025-05-20")
                for t in [
                    "I moved to Beijing in August 2024 to work as a software engineer.",
                    "I bought a new laptop yesterday.",
                ]
            ]
        )
        self.assertEqual([r["level"] for r in rows], ["L0", "L1"])
        self.assertEqual(rows[1]["temporal"][0]["event_start"], "2025-05-19")
        self.assertEqual(rows[1]["R"], 0)

    def test_entity_and_question(self):
        rows = self.route(
            [
                (t, r, "S1", "2025-05-20")
                for t, r in [
                    ("I have a dog named Lucky.", "user"),
                    ("He is three years old.", "user"),
                    ("What is your favorite programming language?", "assistant"),
                    ("Python.", "user"),
                ]
            ]
        )
        self.assertEqual(rows[1]["links"], ["0"])
        self.assertEqual(rows[1]["R"], 0)
        self.assertEqual(rows[3]["links"], ["2"])
        self.assertEqual(rows[3]["generation_tokens"], 0)

    def test_residence_update(self):
        ts = [
            ("I currently live in Beijing.", "user", "S1", "2024-01-01"),
            ("I have now moved to Shanghai.", "user", "S8", "2024-07-03"),
        ]
        a, b = self.route(ts)
        self.assertEqual(b["links"], ["0"])
        self.assertEqual(b["level"], "L1")
        self.assertEqual(b["residual_dependency"]["U"], 0)
        self.assertEqual(a["primary"], ["0"])

    def test_offers_and_deferral(self):
        ts = [
            (t, r, "S1", "2025-05-20")
            for t, r in [
                ("I received offers from Google and Microsoft.", "user"),
                ("Which one are you leaning toward?", "assistant"),
                ("The former, initially.", "user"),
                ("Has anything changed?", "assistant"),
                (
                    "Yes, the other one increased its offer by 20%, so I accepted it last Friday.",
                    "user",
                ),
            ]
        ]
        for budget, expected in [(16000, "refined"), (0, "deferred")]:
            r = self.route(ts, blocks=[["0", "1", "2", "3", "4"]], budget=budget)[0]
            self.assertEqual(r["refinement_status"], expected)
            self.assertEqual(r["level"], "L2" if budget else "L1")
            self.assertGreater(r["R"], 0)
            self.assertEqual(r["temporal"][0]["event_start"], "2025-05-16")

    def test_missing_object(self):
        r = self.route([("I canceled it.", "user", "S1", "")])[0]
        self.assertEqual(r["refinement_status"], "insufficient_evidence")
        self.assertEqual(r["generation_tokens"], 0)
