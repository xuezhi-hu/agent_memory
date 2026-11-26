import unittest

from sparsemem import BASE  # adds the shared backend to sys.path
from sparsemem.recall import key_sentence_entries, keys, retrieve
from dpc_mem.client import FixtureEmbedder
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import FixtureTokenizer


class KeySentenceTest(unittest.TestCase):
    def setUp(self):
        self.archive = {
            "S1:M1@0-95": {"id": "S1:M1", "session": "S1", "order": 0,
                            "role": "user", "date": "2023-05-01",
                            "text": "I bought a model ship last weekend. I will not paint it before June."},
            "S1:M2@0-80": {"id": "S1:M2", "session": "S1", "order": 1,
                            "role": "assistant", "date": "2023-05-01",
                            "text": "Your model ship is a wonderful project to finish."},
        }
        self.records = [{"id": "m000000", "primary": list(self.archive), "links": [],
                         "facts": [], "temporal": [], "entities": []}]

    def test_source_offsets_and_bounded_entries(self):
        entries = key_sentence_entries(self.records, self.archive)
        self.assertLessEqual(len(entries), 2)
        self.assertGreater(len(entries), 0)
        for entry in entries:
            source = self.archive[entry["source_refs"][0]]["text"]
            self.assertEqual(source[entry["source_start"]:entry["source_end"]], entry["text"])
            self.assertEqual(entry["memory"], "m000000")
        self.assertFalse(any(e["kind"] == "key_sentence" for e in keys(self.records, self.archive)))
        self.assertEqual(len(entries), sum(e["kind"] == "key_sentence" for e in keys(self.records, self.archive, key_sentences=2)))

    def test_actual_context_and_record_deduplication(self):
        entries = keys(self.records, self.archive) + key_sentence_entries(self.records, self.archive)
        encoder = FixtureEmbedder()
        index = Index(entries, encoder.embed([e["text"] for e in entries]))
        cfg = dict(channel_k=40, fused_k=40, expand_k=0, final_k=12, rrf_k=60,
                   evidence_tokens=4096, aggregate_mode=False, key_sentence_context=True)
        trace = retrieve(index, encoder, self.records, self.archive, "How many model ships?",
                         "2023-05-10", cfg, FixtureTokenizer())
        self.assertFalse(trace["aggregate"])
        self.assertEqual(trace["ranked"], ["m000000"])
        self.assertTrue(trace["key_sentence_in_context"])
        for hit in trace["key_sentence_in_context"]:
            self.assertIn(hit["text"], trace["evidence"])
        self.assertLessEqual(trace["tokens"], cfg["evidence_tokens"])

    def test_user_only_evidence_excludes_assistant_and_mixed_generated_fact(self):
        self.records[0]['facts'] = [{'text': 'Assistant-derived claim.', 'source_ids': [self.records[0]['primary'][0]]}]
        entries = keys(self.records, self.archive) + key_sentence_entries(self.records, self.archive)
        encoder = FixtureEmbedder()
        index = Index(entries, encoder.embed([e['text'] for e in entries]))
        cfg = dict(channel_k=40, fused_k=40, expand_k=0, final_k=20, rrf_k=60,
                   evidence_tokens=4096, aggregate_mode=False, key_sentence_context=True,
                   user_evidence_only=True)
        trace = retrieve(index, encoder, self.records, self.archive, 'model ship',
                         '2023-05-10', cfg, FixtureTokenizer())
        self.assertIn('I bought a model ship', trace['evidence'])
        self.assertNotIn('| assistant |', trace['evidence'])
        self.assertNotIn('Assistant-derived claim', trace['evidence'])
        self.assertTrue(all(self.archive[k]['role'] == 'user' for k in trace['packed']))


if __name__ == "__main__":
    unittest.main()
