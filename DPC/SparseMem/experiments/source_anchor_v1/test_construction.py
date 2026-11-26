import copy
import unittest

from construction import build_entries
from sparsemem.recall import retrieve
from dpc_mem.client import FixtureEmbedder
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import FixtureTokenizer


class SourceAnchorTests(unittest.TestCase):
    def setUp(self):
        self.archive = {
            "S1:M1@0-100": dict(session="S1", order=0, role="user", date="2023-05-01",
                                 text="I bought a bicycle helmet yesterday. It was surprisingly inexpensive."),
            "S1:M2@0-100": dict(session="S1", order=1, role="user", date="2023-05-01",
                                 text="I paid twenty dollars for a bicycle repair."),
            "S2:M1@0-100": dict(session="S2", order=2, role="assistant", date="2023-05-10",
                                 text="Your bicycle repair might be expensive next week."),
        }
        self.records = [
            dict(id="m000000", primary=list(self.archive)[:2], links=[], temporal=[], facts=[]),
            dict(id="m000001", primary=list(self.archive)[2:], links=[], temporal=[], facts=[]),
        ]

    def test_verbatim_source_diversity_and_no_mutation(self):
        original = copy.deepcopy((self.records, self.archive))
        entries = build_entries(self.records, self.archive, allowed_roles={"user"})
        added = [e for e in entries if e.get("variant") == "source_anchor_v1"]
        self.assertEqual({e["source_refs"][0] for e in added}, set(list(self.archive)[:2]))
        self.assertEqual((self.records, self.archive), original)
        self.assertIn("bicycle", added[0]["anchors"])
        for entry in added:
            source = self.archive[entry["source_refs"][0]]
            self.assertEqual(source["text"][entry["source_start"]:entry["source_end"]], entry["text"])
            self.assertEqual(source["date"], entry["source_date"])
            self.assertIn(entry["source_refs"][0], self.records[0]["primary"])

    def test_existing_index_and_retrieval_accept_entries(self):
        entries = build_entries(self.records, self.archive, allowed_roles={"user"})
        encoder, tokenizer = FixtureEmbedder(), FixtureTokenizer()
        index = Index(entries, encoder.embed([e["text"] for e in entries]))
        cfg = dict(channel_k=40, fused_k=40, expand_k=0, final_k=2, rrf_k=60,
                   evidence_tokens=4096, aggregate_mode=False, key_sentence_context=True,
                   user_evidence_only=True)
        result = retrieve(index, encoder, self.records, self.archive,
                          "What did I pay for the bicycle repair?", "2023-05-15", cfg, tokenizer)
        self.assertTrue(result["ranked"])
        self.assertLessEqual(result["tokens"], cfg["evidence_tokens"])
        self.assertTrue(all(self.archive[k]["role"] == "user" for k in result["packed"]))

    def test_zero_and_invalid_limit(self):
        self.assertFalse(any(e.get("variant") for e in build_entries(self.records, self.archive, max_per_record=0)))
        with self.assertRaises(ValueError):
            build_entries(self.records, self.archive, max_per_record=-1)


if __name__ == "__main__":
    unittest.main()
