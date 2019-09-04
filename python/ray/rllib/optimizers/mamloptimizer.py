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
    """ MAML Optimizer: Workers are different tasks while 
    Every time MAML Optimizer steps...
    1) Workers are set to the same weights as master...
    2) Tasks are randomly sampled and assigned to each worker...
    3) Inner Adaptation Steps
        -Workers collect their own data, update themselves, and collect more data...
        -All data from all workers from all steps gets aggregated to all_samples
    4) Using the aggregated data, update the meta-objective
    """

    def __init__(self, workers, inner_adaptation_steps=1, train_batch_size=1, maml_optimizer_steps=5):
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
        self.maml_optimizer_steps = maml_optimizer_steps

    @override(PolicyOptimizer)
    def step(self):
        # Moved this step to _train() in agents/maml/maml.py
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
            samples = ray_get_and_free([e.sample.remote("pre") for e in self.workers.remote_workers()])
            all_samples = SampleBatch.concat_samples(samples)

            # Data Collection for Meta-Update Step (which will be done on Master Learner)
            for step in range(self.inner_adaptation_steps):
                # Inner Adaptation Gradient Steps
                for i, e in enumerate(self.workers.remote_workers()):
                    e.learn_on_batch.remote(samples[i])
                # Post Adaptation Sampling from Workers
                samples = ray_get_and_free([e.sample.remote("post") for e in self.workers.remote_workers()])
                all_samples = all_samples.concat(SampleBatch.concat_samples(samples))

        #import pdb; pdb.set_trace()
        # Meta gradient Update
        # All Samples should be a list of list of dicts where the dims are (inner_adaptation_steps+1,num_workers,SamplesDict)
        # Should the whole computation graph be in master?

        with self.meta_grad_timer:
            for i in range(self.maml_optimizer_steps):
                fetches = self.workers.local_worker().learn_on_batch(all_samples)
            self.learner_stats = get_learner_stats(fetches)

        self.num_steps_sampled += all_samples.count
        self.num_steps_trained += all_samples.count

        return self.learner_stats
