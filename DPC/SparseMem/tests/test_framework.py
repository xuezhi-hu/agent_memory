import copy, json, os, subprocess, sys, tempfile, unittest
from pathlib import Path
from sparsemem.run import *
from sparsemem.construct import validate, request
from sparsemem.dependency import temporal
from sparsemem.history import fragments

FIX = Path(__file__).parent / "fixture.json"


class Framework(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.s = read(FIX)[0]
        self.t = FixtureTokenizer()
        self.e = FixtureEmbedder()
        self.cfg = dict(DEFAULT, split_threshold=1.1)

    def prepared(self, mode="sparse"):
        turns, _ = project(self.s)
        a, b = segment(turns, self.t, self.e, self.cfg, mode)
        return a, prepare(a, b, self.t, self.cfg, mode)

    def test_scope_projection_and_reassembly(self):
        turns, mapping = project(self.s)
        self.assertNotIn("private", str(turns))
        self.assertNotIn("has_answer", str(turns))
        self.assertIn("answer_private", mapping.values())
        other = dict(self.s, question="leak", answer="leak")
        self.assertEqual(turns, project(other)[0])
        long = "First. " + "Some words with spaces. " * 30 + "end"
        spans = list(fragments(long, self.t, 8))
        self.assertEqual(long, "".join(long[a:b] for a, b in spans))
        self.assertTrue(all(self.t.count(long[a:b]) <= 8 for a, b in spans))

    def test_link_and_ambiguity(self):
        a, rs = self.prepared()
        short = next(
            r
            for r in rs
            if any(a[k]["text"] == "Three years old." for k in r["primary"])
        )
        self.assertLess(short["R"], short["D"])
        self.assertTrue(
            any(
                a[k]["text"] == "How old is Lucky?"
                for k in short["primary"] + short["links"]
            )
        )
        turns, _ = project(self.s)
        turns[:] = [
            dict(turns[0], id="x", text="I met Alice and Bob."),
            dict(turns[0], id="y", text="He left."),
        ]
        a, b = segment(turns, self.t, self.e, self.cfg, "sparse")
        r = prepare(a, b, self.t, self.cfg, "sparse")[-1]
        self.assertLess(r["resolution"]["E"], 1)

    def test_time(self):
        self.assertEqual(temporal("last week", "2024-03-04")["event_end"], "2024-03-03")
        self.assertEqual(
            temporal("yesterday", "2024-03-04")["event_start"], "2024-03-03"
        )
        self.assertEqual(
            temporal("next month", "2024-12-31")["event_end"], "2025-01-31"
        )
        self.assertIsNone(temporal("March", "2024-03-04")["event_start"])
        self.assertIsNone(temporal("tomorrow", "")["event_start"])

    def test_modes_and_budget(self):
        for mode in MODES:
            a, rs = self.prepared(mode)
            led = Ledger(self.out / mode)
            count = construct(rs, a, self.cfg, mode, 42, self.t, led, mock)
            self.assertLessEqual(led.spent("construct"), self.cfg["budget"])
            self.assertEqual({k for r in rs for k in r["primary"]}, set(a))
            self.assertEqual(sum(k["kind"] == "raw" for k in keys(rs, a)), len(rs))
            if mode in ("raw", "raw_session", "raw_round", "raw_block", "linked_only"):
                self.assertEqual(len(led.rows), 0)
            if mode in ("full_extract", "sparse"):
                self.assertGreater(count, 0)
                self.assertIn("L2", [r["level"] for r in rs])
                self.assertEqual(count, len(led.rows))
        a, rs = self.prepared()
        led = Ledger(self.out / "zero")
        construct(rs, a, dict(self.cfg, budget=0), "sparse", 42, self.t, led, mock)
        self.assertEqual(led.rows, {})
        self.assertEqual(
            [(r["level"], r["links"]) for r in rs],
            [(r["level"], r["links"]) for r in self.prepared("linked_only")[1]],
        )

    def test_failures_and_resume(self):
        def fail(*a):
            raise TimeoutError()

        for i, fn in enumerate(
            [
                fail,
                lambda *x: ("bad", dict(prompt_tokens=3, completion_tokens=1)),
                lambda *x: ('{"facts":[{}],"unresolved":[]}', None),
            ]
        ):
            a, rs = self.prepared()
            led = Ledger(self.out / str(i))
            construct(rs, a, self.cfg, "sparse", 42, self.t, led, fn)
            self.assertTrue(all(r["level"] != "L2" for r in rs))
            before = copy.deepcopy(led.rows)
            construct(
                rs,
                a,
                self.cfg,
                "sparse",
                42,
                self.t,
                led,
                lambda *x: self.fail("repeat"),
            )
            self.assertEqual(led.rows, before)
            for row in led.rows.values():
                self.assertEqual(row["charged"], 4 if i == 1 else row["reserved"])

    def test_invalid_refs_and_dates(self):
        a, rs = self.prepared()
        r = rs[0]
        obj = json.loads(mock(request(r, a), 384, "construct")[0])
        obj["facts"][0]["source_ids"] = ["other"]
        self.assertRaises(ValueError, validate, json.dumps(obj), r)
        obj["facts"][0].update(
            source_ids=r["primary"], event_start="2024-03-04", event_end="2024-03-01"
        )
        self.assertRaises(ValueError, validate, json.dumps(obj), r)

    def test_cli_metering_resume_and_scopes(self):
        other = copy.deepcopy(self.s)
        other["question_id"] = "synthetic_02"
        other["haystack_sessions"][0][0]["content"] = "UNIQUE_SECOND_HISTORY"
        path = self.out / "input.json"
        save(path, [self.s, other])
        out = self.out / "run"
        args = [
            sys.executable,
            "-m",
            "sparsemem.run",
            "--data",
            str(path),
            "--out",
            str(out),
            "--limit",
            "2",
            "--mock",
        ]
        subprocess.run(args, check=True, capture_output=True)
        m = read(out / "metrics.json")
        self.assertEqual(m["query_calls"], 2)
        self.assertEqual(
            m["total_generation_tokens"],
            sum(
                m[k]
                for k in (
                    "construct_known_input_tokens",
                    "query_known_input_tokens",
                    "construct_known_output_tokens",
                    "query_known_output_tokens",
                )
            ),
        )
        self.assertNotIn("UNIQUE_SECOND", str(read(out / "scope_00000/sources.json")))
        self.assertEqual(len((out / "hypotheses.jsonl").read_text().splitlines()), 2)
        subprocess.run(args + ["--resume"], check=True, capture_output=True)
        self.assertEqual(
            read(out / "metrics.json")["current_invocation"]["total_generation_tokens"],
            0,
        )
        trace = read(out / "scope_00000/retrieval.json")
        self.assertEqual(len(trace["packed"]), len(set(trace["packed"])))
        self.assertEqual(len(trace["ranked"]), len(set(trace["ranked"])))
        self.assertLessEqual(trace["tokens"], 4096)
