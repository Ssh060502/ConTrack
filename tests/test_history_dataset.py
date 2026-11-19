import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import h5py
from scripts.tools.history_dataset import Episode,HistoryWindows,fit_normalization,window,split_episodes,load_episodes,physical_label,restore_manifest_split,make_manifest


def example(index=0,condition='a',source='batch0'):
    return Episode(str(index),source,str(index),np.arange(18,dtype=np.float32).reshape(6,3)+index,
                   np.arange(10,dtype=np.float32).reshape(5,2),np.array([index,1],np.float32),condition)


class HistoryDataTests(unittest.TestCase):
    def test_frozen_membership_survives_path_changes_and_ignores_extra_data(self):
        episodes=[example(i,'a' if i<3 else 'b') for i in range(6)]
        splits=split_episodes(episodes)
        info=dict(feature_names=['a','b','c'],action_dim=2,control_dt=1/60,references=['waterbottle.h5'],hand_mapping=None,contact_mapping=None)
        manifest=make_manifest(episodes,splits,info,32,2,'episode')
        moved=copy.deepcopy(episodes[::-1])
        for e in moved:e.identifier='copied/'+e.identifier;e.source='copied/'+e.source
        restored,actual=restore_manifest_split(moved+[example(99)],info,manifest)
        self.assertEqual([e.digest for e in restored],[e.digest for e in episodes])
        self.assertEqual(actual,splits)
        with self.assertRaises(ValueError):restore_manifest_split(moved[:-1],info,manifest)

    def test_future_and_current_action_cannot_leak_into_encoder_history(self):
        episode=example();stats=fit_normalization([episode],[0])
        a=window(episode,2,4,stats)
        changed=copy.deepcopy(episode)
        changed.state[3:]+=1000
        changed.action[2:]+=100
        b=window(changed,2,4,stats)
        np.testing.assert_array_equal(a['history'],b['history'])
        self.assertFalse(np.array_equal(a['delta'],b['delta']))
        self.assertFalse(np.array_equal(a['action'],b['action']))
        self.assertEqual(a['length'],2)
        np.testing.assert_array_equal(a['history'][2:],0)
        dataset=HistoryWindows([episode],[0],stats,history=4,stride=1)
        self.assertEqual(len(dataset),4)
        np.testing.assert_array_equal(dataset[1]['history'],a['history'])

    def test_normalization_uses_train_episodes_only(self):
        episodes=[example(),example(1000)]
        stats=fit_normalization(episodes,[0])
        np.testing.assert_allclose(stats['state_mean'],episodes[0].state.mean(0))
        np.testing.assert_allclose(stats['parameter_mean'],episodes[0].log_parameters)

    def test_splits_keep_full_episodes_and_collection_files_disjoint(self):
        episodes=[example(i,'a' if i<6 else 'b',f'batch{i//2}') for i in range(12)]
        for unit in ('episode','file'):
            splits=split_episodes(episodes,42,unit)
            self.assertEqual(set(sum(splits.values(),[])),set(range(12)))
            self.assertEqual(len(sum(splits.values(),[])),12)
            if unit=='file':
                sources=[{episodes[i].source for i in ids} for ids in splits.values()]
                self.assertFalse(sources[0]&sources[1] or sources[0]&sources[2] or sources[1]&sources[2])
        with self.assertRaises(ValueError):split_episodes(episodes[:2])

    def test_readback_labels_and_duplicate_copy_detection(self):
        physics={'mass_kg':[[.1]],'material_static_dynamic_restitution':[[[2.,2.,0.]]]}
        np.testing.assert_allclose(np.exp(physical_label(physics)),[.1,2.])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'episodes.h5'
            with h5py.File(path,'w') as f:
                f.attrs.update(collection_complete=True,control_dt=1/60,
                               metadata_json=json.dumps({'reference_path':'/data/waterbottle.h5','checkpoint':'base.pt'}),
                               physics_json=json.dumps(physics))
                for n in ('000000','000001'):
                    g=f.create_group('episodes/'+n);g.attrs['completed']=True
                    for field in ('qpos','qvel'):g.create_dataset('states/hands/0/'+field,data=np.zeros((6,2)))
                    g.create_dataset('states/contacts/right/0/net_force_w',data=np.zeros((6,1,3)))
                    g.create_dataset('transitions/applied_action',data=np.zeros((5,2)))
                    g.create_dataset('transitions/hands/0/joint_position_target',data=np.ones((5,2)))
                    g.create_dataset('transitions/done',data=[False,False,False,False,True])
                    # Deliberately poisonous oracle data must not enter features.
                    g.create_dataset('policy_observation',data=np.full((5,999),np.nan))
            episodes,info=load_episodes([path],'waterbottle')
            self.assertEqual(len(episodes),1)
            self.assertEqual(episodes[0].state.shape,(6,7))
            self.assertEqual(info['action_dim'],4)
            self.assertEqual(info['skipped'][0]['reason'],'exact duplicate episode')


if __name__=='__main__':unittest.main()
