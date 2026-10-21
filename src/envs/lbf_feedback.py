"""Controlled T1 experiment from the proposal. Fixed simulated advice only."""
import numpy as np
from lbforaging.foraging.environment import ForagingEnv
from .multiagentenv import MultiAgentEnv
from .feedback_gate import AdviceGate


class LBFFeedbackEnv(MultiAgentEnv):
    def __init__(self, seed, common_reward=False, reward_scalarisation="sum",
                 mode="none", strength=0.1, threshold=3, episode_limit=100,
                 feedback_start=40000, evaluation_seed_offset=0, **kwargs):
        if common_reward:
            raise ValueError("Use common_reward=False for member-specific shaping")
        self.n_agents = 3
        self.episode_limit = int(episode_limit)
        self.mode, self.strength, self.threshold = mode, float(strength), int(threshold)
        self.feedback_start = int(feedback_start)
        self.evaluation_seed_offset = int(evaluation_seed_offset)
        self.train_rng = np.random.default_rng(seed)
        self.env = ForagingEnv(players=3, min_player_level=1, max_player_level=1,
            min_food_level=2, max_food_level=2, field_size=(8, 8), max_num_food=2,
            sight=8, max_episode_steps=self.episode_limit, force_coop=False)
        self.test_mode = False
        self.train_steps = 0
        self.reset()

    def reset_for_run(self, test_mode=False, evaluation_seed=100000, train_steps=0):
        self.test_mode, self.train_steps = bool(test_mode), int(train_steps)
        episode_seed = int(evaluation_seed) + self.evaluation_seed_offset if test_mode else int(self.train_rng.integers(0, 2**31-1))
        return self.reset(seed=episode_seed)

    def reset(self, seed=None, options=None):
        if seed is None:
            seed = int(self.train_rng.integers(0, 2**31-1))
        self._obs, _ = self.env.reset(seed=seed)
        # Stable episode ID is a coordinate; never renumber after collection.
        foods = sorted(map(tuple, np.argwhere(self.env.field > 0)))
        if len(foods) != 2:
            raise RuntimeError("T1 requires exactly two foods")
        self.target = foods[0]
        self.gate = AdviceGate(self.mode)
        self.original_return = 0.0
        self.feedback_abs = 0.0
        self.off_condition_abs = 0.0
        self.condition_steps = 0
        self.valid_steps = 0
        self.condition_changes = 0
        self.last_condition = None
        self.active_steps = 0
        self.age = 0
        self._finished = False
        return self._obs, {}

    def distance(self, member):
        p = self.env.players[member].position
        return max(0, abs(p[0]-self.target[0]) + abs(p[1]-self.target[1]) - 1)

    def step(self, actions):
        if self._finished:
            raise RuntimeError("Reset required after episode end")
        actions = np.asarray(actions).reshape(-1)
        valid = self.env.field[self.target] > 0
        leader_pos = self.env.players[0].position
        leader_distance = abs(leader_pos[0]-self.target[0]) + abs(leader_pos[1]-self.target[1])
        condition = bool(leader_distance <= self.threshold)
        if valid:
            self.valid_steps += 1
            self.condition_steps += int(condition)
            if self.last_condition is not None:
                self.condition_changes += int(condition != self.last_condition)
            self.last_condition = condition
        weight = self.gate.weight(valid, condition, self.age, completed=not valid)
        enabled = not self.test_mode and self.train_steps >= self.feedback_start
        weight *= float(enabled)
        before = self.distance(1) if valid else 0
        self._obs, rewards, _, _, _ = self.env.step(actions)
        original = float(sum(rewards))
        self.original_return += original
        collected = bool(valid and self.env.field[self.target] == 0)
        # Only the named helper receives shaping. Bonus requires its actual participation.
        bonus = 0.1 if collected and actions[1] == 5 and rewards[1] > 0 else 0.0
        # Coordinate remains valid for measuring this terminal transition only.
        after = self.distance(1) if valid else 0
        f = (before - after) / 14.0 + bonus if valid else 0.0
        extra = self.strength * weight * f
        # Common per-episode absolute shaping cap in actual reward units.
        extra = float(np.clip(extra, -max(0, 0.5-self.feedback_abs), max(0, 0.5-self.feedback_abs)))
        self.feedback_abs += abs(extra)
        self.off_condition_abs += abs(extra) * float(not condition)
        self.active_steps += int(weight > 0)
        shaped = np.full(3, original, dtype=np.float32)
        shaped[1] += extra
        self.age += 1
        success = not np.any(self.env.field)
        truncated = self.age >= self.episode_limit and not success
        self._finished = bool(success or truncated)
        info = {"original_return": self.original_return, "success": float(success),
                "feedback_abs": self.feedback_abs, "off_condition_abs": self.off_condition_abs,
                "condition_fraction": self.condition_steps / max(1, self.valid_steps),
                "condition_changes": self.condition_changes,
                "active_steps": self.active_steps, "gate_switches": self.gate.switches,
                "episode_limit": bool(truncated)}
        return self._obs, shaped, bool(success), bool(truncated), info

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return self.env.observation_space[0].shape[0]

    def get_state(self):
        return np.concatenate(self._obs).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * self.get_obs_size()

    def get_total_actions(self):
        return 6

    def get_avail_actions(self):
        return [[1] * 6 for _ in range(3)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * 6

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def save_replay(self):
        pass
