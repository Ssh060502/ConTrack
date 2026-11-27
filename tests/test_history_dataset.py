import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import h5py
from scripts.tools.history_dataset import Episode,HistoryWindows,fit_normalization,window,split_episodes,load_episodes,physical_label,restore_manifest_split,make_manifest,select_state_features


def example(index=0,condition='a',source='batch0'):
    return Episode(str(index),source,str(index),np.arange(18,dtype=np.float32).reshape(6,3)+index,
                   np.arange(10,dtype=np.float32).reshape(5,2),np.array([index,1],np.float32),condition)


class HistoryDataTests(unittest.TestCase):
    def test_joint_only_uses_same_raw_split_and_force_cannot_enter_tokens(self):
        episodes=[example(i,'a' if i<3 else 'b') for i in range(6)]
        info=dict(feature_names=['hand0/qpos/0','contact/right/0/net_force_w/0','hand0/qvel/0'],
                  action_dim=2,control_dt=1/60,references=['waterbottle.h5'],hand_mapping=None,contact_mapping=None)
        splits=split_episodes(episodes)
        full=make_manifest(episodes,splits,info,4,1,'episode')
        restored,old_splits=restore_manifest_split(episodes,info,full)
        selected,selected_info=select_state_features(restored,info,'joints')
        self.assertEqual([e.digest for e in selected],[e.digest for e in episodes])
        self.assertEqual(selected_info['feature_names'],['hand0/qpos/0','hand0/qvel/0'])
        new=make_manifest(selected,old_splits,selected_info,4,1,'episode')
        _,new_splits=restore_manifest_split(episodes,info,new)
        self.assertEqual(new_splits,splits)
        stats=fit_normalization(selected,splits['train'])
        token=window(selected[0],2,4,stats)['history']
        changed=copy.deepcopy(episodes);changed[0].state[:,1]=1e9
        alternate,_=select_state_features(changed,info,'joints')
        np.testing.assert_array_equal(window(alternate[0],2,4,stats)['history'],token)
        np.testing.assert_array_equal(selected[0].state,episodes[0].state[:,[0,2]])

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
            # Same tool, different robot: keep them out even when both are
            # recursively discovered under the same collection root.
            crx=Path(directory)/'crx';crx.mkdir()
            import shutil
            shutil.copyfile(path,crx/'episodes.h5')
            with h5py.File(crx/'episodes.h5','a') as f:
                f.attrs['metadata_json']=json.dumps({'reference_path':'/data/waterbottle-crx_sharpa.h5',
                                                    'checkpoint':'crx.pt','task':'Isaac-Crx-Sharpa-Mimic-Manager-v0'})
            old,old_info=load_episodes([directory],'waterbottle')
            new,new_info=load_episodes([directory],'waterbottle','crx')
            self.assertEqual(old_info['robot'],'rby1a');self.assertEqual(new_info['robot'],'crx')
            self.assertEqual(len(old),1);self.assertEqual(len(new),1)
            self.assertTrue(any(x['reason']=='other robot' for x in old_info['skipped']))


if __name__=='__main__':unittest.main()
