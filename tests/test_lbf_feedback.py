import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from envs.lbf_feedback import LBFFeedbackEnv
from lbforaging.foraging.environment import ForagingEnv


class IntegrationTests(unittest.TestCase):
    def test_zero_feedback_preserves_dynamics_and_team_reward(self):
        wrapped = LBFFeedbackEnv(seed=1)
        raw = ForagingEnv(players=3,min_player_level=1,max_player_level=1,
            min_food_level=2,max_food_level=2,field_size=(8,8),max_num_food=2,
            sight=8,max_episode_steps=100,force_coop=False)
        wrapped.reset(seed=77)
        raw.reset(seed=77)
        rng = np.random.default_rng(42)
        for _ in range(100):
            actions = rng.integers(0,6,3)
            obs, rewards, terminal, timeout, info = wrapped.step(actions)
            robs, rrewards, rdone, _, _ = raw.step(actions)
            np.testing.assert_array_equal(obs, robs)
            np.testing.assert_allclose(rewards, sum(rrewards))
            self.assertEqual(terminal or timeout, rdone)
            if terminal or timeout: break
        wrapped.close(); raw.close()

    def test_evaluation_has_no_shaping_or_training_rng_effect(self):
        a = LBFFeedbackEnv(seed=3,mode="persistent",feedback_start=0)
        b = LBFFeedbackEnv(seed=3,mode="none",feedback_start=0)
        for env in (a,b): env.reset_for_run(test_mode=True,evaluation_seed=100001)
        for _ in range(20):
            ra = a.step([0,0,0]); rb = b.step([0,0,0])
            np.testing.assert_array_equal(ra[1], rb[1])
            self.assertEqual(ra[4]["feedback_abs"], 0)
        a.reset_for_run(test_mode=True,evaluation_seed=99999)
        a.reset_for_run(test_mode=False); b.reset_for_run(test_mode=False)
        np.testing.assert_array_equal(a.get_obs(), b.get_obs())
        a.close(); b.close()

    def test_context_and_direct_produce_identical_trajectories(self):
        a = LBFFeedbackEnv(seed=1,mode="context",feedback_start=0)
        b = LBFFeedbackEnv(seed=1,mode="direct",feedback_start=0)
        a.reset(seed=42); b.reset(seed=42)
        rng = np.random.default_rng(7)
        for _ in range(100):
            actions=rng.integers(0,6,3)
            x=a.step(actions); y=b.step(actions)
            np.testing.assert_array_equal(x[1],y[1])
            if x[2] or x[3]: break
        self.assertLessEqual(a.feedback_abs,0.5+1e-8)
        a.close(); b.close()

if __name__ == "__main__": unittest.main()
