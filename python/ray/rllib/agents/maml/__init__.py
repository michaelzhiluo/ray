from ray.rllib.agents.maml.maml import MAMLTrainer, DEFAULT_CONFIG
from ray.rllib.utils import renamed_agent

MAMLAgent = renamed_agent(MAMLTrainer)

__all__ = ["MAMLAgent", "MAMLTrainer", "DEFAULT_CONFIG"]
