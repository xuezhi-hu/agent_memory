from sparsemem.evidence_packing import pack_local_evidence
from dpc_mem.tokenization import FixtureTokenizer


def test_local_packing_only_uses_candidate_user_sources_and_budget():
    archive = {
        'u1': dict(role='user', date='2024-01-01', order=0, text='I had my bicycle serviced. The chain cost $25. The light cost $40.'),
        'a1': dict(role='assistant', date='2024-01-01', order=1, text='The bicycle cost $999.'),
        'u2': dict(role='user', date='2024-02-01', order=2, text='My bicycle tires cost $30.'),
        'outside': dict(role='user', date='2024-03-01', order=3, text='My bicycle cost $777.'),
    }
    records = [dict(id='m0', primary=['u1', 'a1'], links=[]),
               dict(id='m1', primary=['u2'], links=[])]
    result = pack_local_evidence('How much did my bicycle cost?', ['m0', 'm1'],
                                 records, archive, FixtureTokenizer(), 120,
                                 primary_k=1, candidate_k=2)
    assert result['tokens'] <= 120
    assert '$25' in result['evidence'] and '$40' in result['evidence']
    assert '$999' not in result['evidence'] and '$777' not in result['evidence']
    assert all(archive[row['source_id']]['role'] == 'user' for row in result['selected_sentences'])


def test_local_packing_has_stable_output_and_exact_source_refs():
    archive = {'u': dict(role='user', date='', order=0, text='I bought a plant. It cost $10.')}
    records = [dict(id='m', primary=['u'], links=['u'])]
    a = pack_local_evidence('How much did the plant cost?', ['m'], records, archive, FixtureTokenizer(), 100)
    b = pack_local_evidence('How much did the plant cost?', ['m'], records, archive, FixtureTokenizer(), 100)
    assert a == b
    assert len({(x['source_id'], x['text']) for x in a['selected_sentences']}) == len(a['selected_sentences'])
    assert all(x['text'] in archive[x['source_id']]['text'] for x in a['selected_sentences'])
