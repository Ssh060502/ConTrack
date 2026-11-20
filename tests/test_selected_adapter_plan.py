from pathlib import Path
import unittest
from scripts.evaluate_selected_adapters import build_plan,expand_plan,parse_conditions


class PlanTests(unittest.TestCase):
    def test_repeat_and_counterfactual_change_inputs_not_actual_condition(self):
        original=build_plan(Path('/repo'),['waterbottle_mass'])
        plan=expand_plan(original,[42,43,44],True)
        self.assertEqual(len(plan),15)
        self.assertEqual(len({j['name'] for j in plan}),15)
        fixed=[j for j in plan if 'oracle_mass_scale' in j]
        self.assertEqual(len(fixed),3)
        self.assertTrue(all(j['condition']=='3:2' and j['oracle_mass_scale']==1. for j in fixed))
        self.assertFalse(any('oracle_mass_scale' in j for j in original))

    def test_unseen_mass_plan(self):
        jobs=build_plan(Path('/repo'),['waterbottle_mass'],conditions=parse_conditions('2:2,4:2,6:2'))
        self.assertEqual(len(expand_plan(jobs,[42,43,44])),18)
        self.assertTrue(all(float(j['condition'].split(':')[0]) in (2.,4.,6.) for j in jobs))
        for invalid in ('0:2','nan:2','1:2,1:2','1:2:3',''):
            with self.assertRaises(ValueError):parse_conditions(invalid)

    def test_exact_selected_checkpoints_and_deduplicated_baselines(self):
        experiments=['waterbottle_mass','waterbottle_friction','mug_mass','mug_friction']
        jobs=build_plan(Path('/repo'),experiments)
        self.assertEqual(len(jobs),18)
        self.assertEqual(sum(j['policy']=='base' for j in jobs),8)
        self.assertEqual(len({j['name'] for j in jobs}),18)
        for name,model in [('waterbottle_mass','model_1000.pt'),('waterbottle_friction','adapter_final.pt'),
                           ('mug_mass','adapter_final.pt'),('mug_friction','model_1500.pt')]:
            selected=[j for j in jobs if j['name'].startswith(name+'_')]
            self.assertTrue(selected)
            self.assertTrue(all(j['checkpoint'].endswith(model) for j in selected))

    def test_single_experiment_adapter_only(self):
        jobs=build_plan(Path('/repo'),['mug_friction'],False)
        self.assertEqual([j['condition'] for j in jobs],['1:2','1:1','1:0.7'])
        self.assertTrue(all(j['policy']=='adapter' for j in jobs))


if __name__=='__main__':unittest.main()
