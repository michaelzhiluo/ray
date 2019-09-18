from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.external_env import ExternalEnv
from ray.rllib.env.serving_env import ServingEnv
from ray.rllib.env.vector_env import VectorEnv
from ray.rllib.env.env_context import EnvContext
from gym.envs.registration import registry, register, make, spec

print("Registering Custom Environments")
register(
    id='HalfCheetahRandDirec-v2',
    entry_point='ray.rllib.env.half_cheetah_rand_direc:NormalizedEnv',
    max_episode_steps=1000,
)

__all__ = [
    "BaseEnv", "MultiAgentEnv", "ExternalEnv", "VectorEnv", "ServingEnv",
    "EnvContext"
]
