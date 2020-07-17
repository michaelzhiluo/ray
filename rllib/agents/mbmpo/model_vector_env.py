import logging
import numpy as np

from ray.rllib.utils.annotations import override
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.evaluation.rollout_worker import get_global_worker

logger = logging.getLogger(__name__)


def custom_model_vector_env(**kwargs):
    worker_index = kwargs["env_context"].worker_index
    env = kwargs["env"]
    if worker_index:
        return _VectorizedModelGymEnv(
            make_env=kwargs["make_env"],
            existing_envs=kwargs["existing_envs"],
            num_envs=kwargs["num_worker_envs"],
            observation_space=env.observation_space,
            action_space=env.action_space,
            env_config=kwargs["env_config"])
    else:
        # This env will become vectorized later
        return env


class _VectorizedModelGymEnv(VectorEnv):
    """Internal wrapper to translate model-based gym envs into a VectorEnv object.
    """

    def __init__(self,
                 make_env=None,
                 existing_envs=None,
                 num_envs=1,
                 *,
                 observation_space=None,
                 action_space=None,
                 env_config=None):
        self.make_env = make_env
        self.envs = existing_envs
        self.num_envs = num_envs
        while len(self.envs) < num_envs:
            self.envs.append(self.make_env(len(self.envs)))

        super().__init__(
            observation_space=observation_space
            or self.envs[0].observation_space,
            action_space=action_space or self.envs[0].action_space,
            num_envs=num_envs)
        worker = get_global_worker()
        self.model, self.device = worker.foreach_policy(
            lambda x, y: (x.dynamics_model, x.device))[0]

    @override(VectorEnv)
    def vector_reset(self):
        self.cur_obs = [e.reset() for e in self.envs]
        return self.cur_obs

    @override(VectorEnv)
    def reset_at(self, index):
        return self.envs[index].reset()

    @override(VectorEnv)
    def vector_step(self, actions):
        if self.cur_obs is None:
            raise ValueError("Need to reset env first")

        obs_batch = np.stack(self.cur_obs, axis=0)
        action_batch = np.stack(actions, axis=0)

        next_obs_batch = self.model.predict_model_batches(
            obs_batch, action_batch, device=self.device)

        next_obs_batch = np.clip(next_obs_batch, -1000, 1000)

        rew_batch = self.envs[0].reward(obs_batch, action_batch,
                                        next_obs_batch)

        if hasattr(self.envs[0], "done"):
            dones_batch = self.envs[0].done(next_obs_batch)
        else:
            dones_batch = np.asarray([False for _ in range(self.num_envs)])

        info_batch = [{} for _ in range(self.num_envs)]

        self.cur_obs = next_obs_batch

        return list(obs_batch), list(rew_batch), list(dones_batch), info_batch

    @override(VectorEnv)
    def get_unwrapped(self):
        return self.envs
