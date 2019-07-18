from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
import logging
from ray.rllib.evaluation.metrics import get_learner_stats
from ray.rllib.optimizers.policy_optimizer import PolicyOptimizer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.filter import RunningStat
from ray.rllib.utils.timer import TimerStat
from ray.rllib.utils.memory import ray_get_and_free

logger = logging.getLogger(__name__)

class MAMLOptimizer(PolicyOptimizer):
    """ MAML Optimizer

    Each worker represents a different task (num_workers = num_tasks)
    Environments can be vectorized within each worker
    """

    def __init__(self, workers, inner_adaptation_steps=1, train_batch_size=1):
        PolicyOptimizer.__init__(self, workers)
        # Each worker represents a different task
        self.num_tasks = len(self.workers.remote_workers())
        self.update_weights_timer = TimerStat()
        self.set_tasks_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.meta_grad_timer = TimerStat()
        self.inner_adaptation_steps = inner_adaptation_steps
        self.train_batch_size = train_batch_size
        self.learner_stats = {}

    @override(PolicyOptimizer)
    def step(self):
        # Initialize Workers to have the same weights
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        # Set Tasks for each Worker
        with self.set_tasks_timer:
            env_configs = self.workers.local_worker().sample_tasks(self.num_tasks)
            for i,e in enumerate(self.workers.remote_workers()):
                e.set_task.remote(env_configs[i])

        # Collecting Data from Pre and Post Adaptations
        with self.sample_timer:

            # Pre Adaptation Sampling from Workers
            samples_id = [e.sample.remote() for e in self.workers.remote_workers()]
            samples = ray_get_and_free(samples_id)
            all_samples = SampleBatch.concat_samples(samples)

            # Data Collection for Meta-Update Step (which will be done on Master Learner)
            for step in range(self.inner_adaptation_steps):
                # Inner Adaptation Gradient Steps
                for i, e in enumerate(self.workers.remote_workers()):
                    e.learn_on_batch.remote(samples_id[i])
                # Post Adaptation Sampling from Workers
                samples_id = [e.sample.remote() for e in self.workers.remote_workers()]
                samples = ray_get_and_free(samples_id)
                all_samples = all_samples.concat(SampleBatch.concat_samples(samples))

        # Meta gradient Update
        # All Samples should be a list of list of dicts where the dims are (inner_adaptation_steps+1,num_workers,SamplesDict)
        # Should the whole computation graph be in master?
        with self.meta_grad_timer:
            fetches = self.workers.local_worker().learn_on_batch(all_samples)
            #self.learner_stats = get_learner_stats(fetches)

        return self.learner_stats
