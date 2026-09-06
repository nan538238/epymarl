"""Minimal LBF wrapper with hidden, switching scripted teammates."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import lbforaging  # noqa: F401 - registers the lbforaging Gymnasium IDs


class SwitchingLBFEnv(gym.Env):
    """Single-ego interface over a three-player Level-Based Foraging task."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        base_key="lbforaging:Foraging-2s-10x10-3p-3f-v3",
        switch_min_frac=0.3,
        switch_max_frac=0.7,
        teammate_modes=("greedy", "left_priority", "wait"),
        reveal_teammate_modes=False,
        include_teammate_last_actions=False,
        initial_mode_ids=None,
        switch_mode_ids=None,
        fixed_switch_step=None,
        seed=None,
        max_episode_steps=None,
        **kwargs,
    ):
        del kwargs
        self.base_key = base_key
        self.switch_min_frac = float(switch_min_frac)
        self.switch_max_frac = float(switch_max_frac)
        self.teammate_modes = tuple(teammate_modes)
        self.reveal_teammate_modes = bool(reveal_teammate_modes)
        self.include_teammate_last_actions = bool(include_teammate_last_actions)
        self.initial_mode_ids = (
            tuple(int(v) for v in initial_mode_ids)
            if initial_mode_ids is not None
            else None
        )
        self.switch_mode_ids = (
            tuple(int(v) for v in switch_mode_ids)
            if switch_mode_ids is not None
            else None
        )
        self.fixed_switch_step = (
            int(fixed_switch_step) if fixed_switch_step is not None else None
        )
        if len(self.teammate_modes) < 2:
            raise ValueError("SwitchingLBFEnv needs at least two teammate modes")
        for name, mode_ids in (
            ("initial_mode_ids", self.initial_mode_ids),
            ("switch_mode_ids", self.switch_mode_ids),
        ):
            if mode_ids is not None and len(mode_ids) != 2:
                raise ValueError(f"{name} must contain two teammate mode IDs")
            if mode_ids is not None and any(
                mode < 0 or mode >= len(self.teammate_modes) for mode in mode_ids
            ):
                raise ValueError(f"{name} contains an invalid teammate mode ID")

        self.episode_limit = int(max_episode_steps or 50)
        self.env = gym.make(base_key, max_episode_steps=self.episode_limit)
        self.base_env = self.env.unwrapped
        for attr in ("_max_episode_steps", "_max_steps", "max_steps", "_step_limit"):
            if hasattr(self.base_env, attr):
                setattr(self.base_env, attr, self.episode_limit)
                break

        self.base_n_agents = int(self.base_env.n_agents)
        if self.base_n_agents != 3:
            raise ValueError("The prototype expects exactly 3 base LBF players")

        # EPyMARL sees one learnable ego. The wrapped LBF still contains three
        # players; players 1 and 2 are controlled below by scripted policies.
        self.n_agents = 1
        self.action_space = spaces.Tuple((self.env.action_space[0],))
        self._base_obs_size = int(np.prod(self.env.observation_space[0].shape))
        self._action_size = int(self.env.action_space[0].n)
        extra_obs_size = 0
        if self.reveal_teammate_modes:
            extra_obs_size += 2 * len(self.teammate_modes)
        if self.include_teammate_last_actions:
            extra_obs_size += 2 * self._action_size
        self._extra_obs_size = extra_obs_size
        obs_size = self._base_obs_size + extra_obs_size
        self.observation_space = spaces.Tuple(
            (spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32),)
        )
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._switch_step = 0
        self._initial_modes = (0, 1)
        self._current_modes = (0, 1)
        self._switched = False
        self._last_teammate_actions = (0, 0)

    def _mode_name(self, mode_id):
        return self.teammate_modes[int(mode_id) % len(self.teammate_modes)]

    def _food_positions(self):
        return [
            tuple(int(v) for v in p)
            for p in np.argwhere(np.asarray(self.base_env.field) > 0)
        ]

    @staticmethod
    def _manhattan(a, b):
        return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

    def _choose_target(self, player_id, mode):
        player = self.base_env.players[player_id]
        if player.position is None:
            return None
        foods = self._food_positions()
        if not foods:
            return None
        pos = tuple(int(v) for v in player.position)
        distances = {food: self._manhattan(pos, food) for food in foods}
        name = self._mode_name(mode)
        if name == "left_priority":
            return min(foods, key=lambda f: (f[1], distances[f], f[0]))
        if name == "wait":
            ego = self.base_env.players[0]
            if ego.position is None or self._manhattan(pos, ego.position) > 2:
                return None
        if name == "far_priority":
            return max(foods, key=lambda f: (distances[f], f[0], f[1]))
        return min(foods, key=lambda f: (distances[f], f[0], f[1]))

    def _move_action(self, player_id, target):
        player = self.base_env.players[player_id]
        if player.position is None or target is None:
            return 0  # NOOP
        row, col = (int(v) for v in player.position)
        target_row, target_col = target
        if (row, col) == target:
            return self.env.action_space[player_id].n - 1  # LOAD
        if abs(target_row - row) >= abs(target_col - col) and target_row != row:
            return 1 if target_row < row else 2  # UP / DOWN
        if target_col != col:
            return 3 if target_col < col else 4  # LEFT / RIGHT
        return 0

    def _teammate_action(self, player_id, mode):
        return self._move_action(player_id, self._choose_target(player_id, mode))

    def _info(self, teammate_actions):
        return {
            "switching_lbf_step": float(self._t),
            "switching_lbf_switch_step": float(self._switch_step),
            "switching_lbf_switched": float(self._switched),
            "switching_lbf_teammate_0_mode": float(self._current_modes[0]),
            "switching_lbf_teammate_1_mode": float(self._current_modes[1]),
            "switching_lbf_teammate_actions": float(sum(teammate_actions)),
        }

    def _augment_obs(self, obs):
        features = [np.asarray(obs[0], dtype=np.float32).reshape(-1)]
        if self.reveal_teammate_modes:
            mode_features = np.zeros((2, len(self.teammate_modes)), dtype=np.float32)
            for i, mode in enumerate(self._current_modes):
                mode_features[i, int(mode)] = 1.0
            features.append(mode_features.reshape(-1))
        if self.include_teammate_last_actions:
            action_features = np.zeros((2, self._action_size), dtype=np.float32)
            for i, action in enumerate(self._last_teammate_actions):
                action_features[i, int(action)] = 1.0
            features.append(action_features.reshape(-1))
        return (np.concatenate(features).astype(np.float32),)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._t = 0
        self._switched = False
        self._last_teammate_actions = (0, 0)
        lo = max(1, int(self.episode_limit * self.switch_min_frac))
        hi = max(lo + 1, int(self.episode_limit * self.switch_max_frac) + 1)
        self._switch_step = (
            min(max(self.fixed_switch_step, 1), self.episode_limit - 1)
            if self.fixed_switch_step is not None
            else int(self._rng.integers(lo, hi))
        )
        n_modes = len(self.teammate_modes)
        self._initial_modes = (
            self.initial_mode_ids
            if self.initial_mode_ids is not None
            else tuple(int(v) for v in self._rng.choice(n_modes, size=2, replace=False))
        )
        self._current_modes = self._initial_modes
        info = dict(info or {})
        info.update(self._info((0, 0)))
        # Hidden mode and switch timing are deliberately absent from obs.
        return self._augment_obs(obs), info

    def step(self, actions):
        if self._t >= self._switch_step and not self._switched:
            n_modes = len(self.teammate_modes)
            self._current_modes = (
                self.switch_mode_ids
                if self.switch_mode_ids is not None
                else tuple(
                    (mode + 1 + int(self._rng.integers(n_modes - 1))) % n_modes
                    for mode in self._initial_modes
                )
            )
            self._switched = True

        ego_action = int(np.asarray(actions).reshape(-1)[0])
        teammate_actions = [
            self._teammate_action(1, self._current_modes[0]),
            self._teammate_action(2, self._current_modes[1]),
        ]
        obs, reward, terminated, truncated, info = self.env.step(
            [ego_action, *teammate_actions]
        )
        self._t += 1
        self._last_teammate_actions = tuple(teammate_actions)
        info = dict(info or {})
        info.update(self._info(teammate_actions))
        # The base LBF returns one reward per player; this interface has one
        # learnable ego, so expose the team reward as a Gymnasium scalar.
        reward = float(np.asarray(reward, dtype=np.float32).sum())
        return self._augment_obs(obs), reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self._rng = np.random.default_rng(seed)
        return self.env.unwrapped.seed(seed)


def register_switching_lbf():
    """Register the prototype once without touching external environments."""
    from gymnasium.envs.registration import register, registry

    registrations = {
        "epymarl/Switching-LBF-v0": {},
        "epymarl/Switching-LBF-TypeOracle-v0": {"reveal_teammate_modes": True},
        "epymarl/Switching-LBF-LastAction-v0": {
            "include_teammate_last_actions": True
        },
        "epymarl/Switching-LBF-Belief-v0": {
            "include_teammate_last_actions": True
        },
    }
    for env_id, kwargs in registrations.items():
        if env_id not in registry:
            register(
                id=env_id,
                entry_point="envs.switching_lbf:SwitchingLBFEnv",
                kwargs=kwargs,
            )
