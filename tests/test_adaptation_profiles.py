import unittest
from scripts.tools.adaptation_profiles import robot_profile, action_indices, collection_robot


class ProfileTests(unittest.TestCase):
    def test_robot_profiles_do_not_share_task_or_data_environment(self):
        old,new=robot_profile('rby1a'),robot_profile('crx')
        self.assertNotEqual(old['task'],new['task'])
        self.assertNotEqual(old['data_env'],new['data_env'])
        with self.assertRaises(ValueError):robot_profile('unknown')

    def test_diagnostics_identifies_crx_arms_instead_of_treating_all_as_fingers(self):
        for robot,count in [('rby1a',7),('crx',6)]:
            names=[]
            for side in ('left','right'):
                names += ([f'{side}_arm_{i}' for i in range(count)] if robot=='rby1a'
                          else [f'{side}_J{i}' for i in range(1,count+1)])
                names += [f'{side}_finger_{i}' for i in range(22)]
            arms,fingers=action_indices(names)
            self.assertEqual(len(arms),2*count)
            self.assertEqual(len(fingers),44)
            self.assertEqual(sorted(arms+fingers),list(range(len(names))))

    def test_collection_robot_uses_explicit_task_before_legacy_filename(self):
        self.assertEqual(collection_robot({'task':'Isaac-Crx-Sharpa-Mimic-Manager-v0'}),'crx')
        self.assertEqual(collection_robot({'reference_path':'hammer-crx_sharpa.h5'}),'crx')
        self.assertEqual(collection_robot({'reference_path':'old_waterbottle.h5'}),'rby1a')


if __name__=='__main__':unittest.main()
