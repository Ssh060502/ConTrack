from pathlib import Path
import unittest
from scripts.evaluate_selected_adapters import build_plan


class PlanTests(unittest.TestCase):
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
