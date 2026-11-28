"""Study planning does not need Isaac Sim or torch."""
import unittest
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from scripts.run_waterbottle_study import context_args,parse_conditions
from scripts import run_waterbottle_study as study
from scripts.tools.score_waterbottle_study import summarize


class StudyTests(unittest.TestCase):
    def test_manual_success_is_not_inferred_from_missing_labels(self):
        rows=[dict(method='state',condition='3:2',episode=str(i),outcome=label)
              for i,label in enumerate(('completed','dropped',''))]
        self.assertIsNone(summarize(rows)[0]['completion_rate'])
        rows[-1]['outcome']='no_lift'
        self.assertAlmostEqual(summarize(rows)[0]['completion_rate'],1/3)
        with self.assertRaisesRegex(ValueError,'Duplicate'):summarize(rows+[rows[0]])
        rows[-1]['outcome']='time_out'
        with self.assertRaisesRegex(ValueError,'Invalid'):summarize(rows)

    def test_no_privileged_context_in_state_or_base(self):
        self.assertEqual(context_args('state',None),['--context-source','state','--context-dim','64'])
        self.assertEqual(context_args('base',None),context_args('state',None))

    def test_random_and_pretrained_share_template(self):
        b=context_args('random-history','model.pt',64,9)
        c=context_args('history','model.pt')
        self.assertIn('model.pt',b);self.assertIn('model.pt',c)
        self.assertEqual(b[-2:],['--encoder-seed','9'])
        self.assertNotIn('--encoder-seed',c)
        with self.assertRaises(ValueError):context_args('history',None)

    def test_conditions(self):
        self.assertEqual(parse_conditions('1:2,3:2,1:0.7'),[(1.,2.),(3.,2.),(1.,.7)])
        for value in ('1:2,1:2','0:2','nan:2','1:2:3'):
            with self.assertRaises(ValueError):parse_conditions(value)

    def test_train_evaluate_manifest_and_incomplete_child(self):
        import h5py
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def fake_execute(command,directory,environment):
                def arg(name):return command[command.index(name)+1]
                directory=Path(directory);directory.mkdir(parents=True)
                if arg('--mode')=='train':
                    (directory/'adapter_final.pt').write_bytes(b'fake')
                    metadata=dict(base_checkpoint=str(root/'base/model.pt'),reference_path=str(root/'reference.h5'))
                    (directory/'run_metadata.json').write_text(json.dumps(metadata))
                else:
                    with h5py.File(directory/'episodes.h5','w') as f:
                        f.attrs['collection_complete']=True;f.attrs['completed_episodes']=2
                        f.attrs['metadata_json']=json.dumps(dict(context_source=arg('--context-source'),
                            evaluation_policy=arg('--policy'),conditions=[list(map(float,arg('--conditions').split(':')))],seed=42))
                        f.attrs['physics_json']=json.dumps(dict(actual_mass_kg=[[.1]],actual_inertia=[[.01]*9],
                            material_static_dynamic_restitution=[[[2.,2.,0.]]]))
            common=['--output-root',str(root)]
            with patch.object(study,'execute',side_effect=fake_execute),patch('sys.stdout',new_callable=io.StringIO):
                with patch('sys.argv',['study','train','--methods','state','oracle',*common]):study.main()
                train=next(root.glob('train_*'))
                manifest=json.loads((train/'study.json').read_text())
                self.assertTrue(all(j['complete'] for j in manifest['jobs']))
                self.assertEqual(manifest['arguments']['data'],str(root/'reference.h5'))
                with patch('sys.argv',['study','evaluate','--study',str(train),'--methods','base','state','oracle',
                                       '--conditions','3:2','--episodes','2',*common]):study.main()
                evaluation=json.loads((next(root.glob('evaluate_*'))/'study.json').read_text())
                self.assertTrue(all(j['complete'] for j in evaluation['jobs']))
            with patch.object(study,'execute'),patch('sys.stdout',new_callable=io.StringIO),patch('sys.argv',
                ['study','train','--methods','state',*common]):
                with self.assertRaisesRegex(RuntimeError,'Missing final'):study.main()


if __name__=='__main__':unittest.main()
