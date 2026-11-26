import unittest
from strategy import anchors, build_postings, make_units, pack, retrieve_candidates
from dpc_mem.retrieval import Index
from dpc_mem.tokenization import FixtureTokenizer

class StrategyTests(unittest.TestCase):
    def setUp(self):
        self.archive={
            'S1:M1@0-1':{'role':'user','text':'I repaired my bicycle chain for $25.','session':'S1','order':0,'date':'2024-01-01'},
            'S2:M1@0-1':{'role':'user','text':'I installed a bike light for $40.','session':'S2','order':1,'date':'2024-02-01'},
            'S3:M1@0-1':{'role':'user','text':'I repaired my bicycle chain again for $25.','session':'S3','order':2,'date':'2024-03-01'},
        }
        self.records=[{'id':f'm{i:06d}','primary':[sid],'links':[],'facts':[]} for i,sid in enumerate(self.archive)]

    def test_same_anchor_links_different_sessions_and_never_invents_sources(self):
        p=build_postings(self.records,self.archive)
        self.assertEqual(p['postings']['bicycle'],['m000000','m000001','m000002'])
        self.assertEqual(anchors('Bike and bicycle'),{'bicycle'})
        self.assertEqual(p['added_generation_tokens'],0)

    def test_both_packers_keep_distinct_events_and_obey_budget(self):
        rows=[]
        for n,(sid,src) in enumerate(self.archive.items()):
            rows.append(dict(id=sid+'#0',source_id=sid,memory=f'm{n:06d}',session=src['session'],date=src['date'],order=n,text=src['text'],rank=n,relevance=.9-n*.1,termset=sorted(anchors(src['text'])),values=['25' if n!=1 else '40']))
        tok=FixtureTokenizer()
        for mode in ('A','B'):
            out=pack('How much did I spend on bicycle repairs?',rows,self.archive,tok,200,mode)
            self.assertLessEqual(out['tokens'],200)
            self.assertEqual(len(out['selected_units']),3)
            self.assertEqual(len(out['ranked']),3)
            self.assertEqual(set(out['packed']),set(self.archive))

    def test_strict_user_only(self):
        self.archive['S1:M1@0-1']['role']='assistant'
        with self.assertRaises(ValueError): build_postings(self.records,self.archive)

    def test_cross_session_extra_reaches_both_reader_contexts(self):
        archive={}; records=[]; entries=[]; vectors=[]
        for i in range(24):
            sid=f'S{i:02}:M1@0-30';mid=f'm{i:06d}'
            text=('I repaired my bicycle for $25.' if i==0 else
                  'Another bicycle repair cost $40.' if i==22 else 'Unrelated grocery shopping.')
            archive[sid]=dict(role='user',text=text,session=f'S{i:02}',order=i,date='2024-01-01')
            records.append(dict(id=mid,primary=[sid],links=[],facts=[],session_id=f'S{i:02}'))
            entries.append(dict(id=f'e{i}',memory=mid,text=text,source_refs=[sid]))
            vectors.append([1.,0.] if i in (0,22) else [0.,1.])
        index=Index(entries,vectors);baseline=[r['id'] for r in records[:20]]
        candidate=retrieve_candidates('How much was bicycle repair?',records,archive,index,[1.,0.],
                                      dict(initial=baseline,ranked=baseline),build_postings(records,archive))
        self.assertFalse(candidate['fallback'])
        self.assertIn('m000022',candidate['ranked'])
        units=make_units('How much was bicycle repair?',records,archive,candidate,FixtureTokenizer())
        for mode in ('A','B'):
            result=pack('How much was bicycle repair?',units,archive,FixtureTokenizer(),1000,mode)
            self.assertIn('S22:M1@0-30',result['packed'])
            self.assertIn('$40',result['evidence'])

if __name__=='__main__': unittest.main()
