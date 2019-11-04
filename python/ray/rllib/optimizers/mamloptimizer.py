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
from ray.rllib.optimizers.LinearBaseline import LinearFeatureBaseline
import numpy as np
import scipy

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

    def __init__(self, workers, config, inner_adaptation_steps=1, train_batch_size=1, maml_optimizer_steps=5):
        PolicyOptimizer.__init__(self, workers)
        # Each worker represents a different task
        self.baseline = LinearFeatureBaseline()
        self.discount = config["gamma"]
        self.gae_lambda = config["lambda"]
        self.num_tasks = len(self.workers.remote_workers())
        self.update_weights_timer = TimerStat()
        self.set_tasks_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.meta_grad_timer = TimerStat()
        self.inner_adaptation_steps = inner_adaptation_steps
        self.train_batch_size = train_batch_size
        self.learner_stats = {}
        self.maml_optimizer_steps = maml_optimizer_steps
        self.config = config
        self.context_input_size = self.config["model"]["concat_input_size"]

    @override(PolicyOptimizer)
    def step(self):
        # Initialize Workers to have the same weights
        print("Start of Optimizer Loop: Setting Weights")
        with self.update_weights_timer:
            if self.workers.remote_workers():
                weights = ray.put(self.workers.local_worker().get_weights())
                for e in self.workers.remote_workers():
                    e.set_weights.remote(weights)

        print("Setting Tasks for each Worker")
        # Set Tasks for each Worker
        with self.set_tasks_timer:
            env_configs = self.workers.local_worker().sample_tasks(self.num_tasks)
            for i,e in enumerate(self.workers.remote_workers()):
                e.set_task.remote(env_configs[i])

        env_configs = self.preprocess_context(env_configs)

        #import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        # Collecting Data from Pre and Post Adaptations
        print("Sampling Data")
        with self.sample_timer:
            meta_split = []
            # Pre Adaptation Sampling from Workers
            samples = ray_get_and_free([e.sample.remote("pre") for i,e in enumerate(self.workers.remote_workers())])
            samples, split_list = self.post_processing(samples, self.config["num_envs_per_worker"])
            all_samples = SampleBatch.concat_samples(samples)
            meta_split.append(split_list)

            # Data Collection for Meta-Update Step (which will be done on Master Learner)
            for step in range(self.inner_adaptation_steps):
                # Inner Adaptation Gradient Steps
                print("Inner Adaptation")
                for i, e in enumerate(self.workers.remote_workers()):
                    if self.config["use_context"] == "dynamic":
                        context_inp = np.atleast_1d(env_configs[i])
                        assert self.context_input_size%context_inp.shape[0]==0
                        samples[i]["context"] = np.tile(context_inp, int(self.context_input_size/context_inp.shape[0]))
                    elif self.config["use_context"] == "static":
                        samples[i]["context"] = np.ones(self.context_input_size)
                    e.learn_on_batch.remote(samples[i])
                
                print("Sampling Data")
                samples = ray_get_and_free([e.sample.remote("post") for e in self.workers.remote_workers()])
                samples, split_list = self.post_processing(samples, self.config["num_envs_per_worker"])
                all_samples = all_samples.concat(SampleBatch.concat_samples(samples))
                meta_split.append(split_list)
        context_list = []
        for i in env_configs:
            if self.config["use_context"] == "dynamic":
                context_inp = np.atleast_1d(i)
                context_list += [np.tile(context_inp, int(self.context_input_size/context_inp.shape[0]))]
            elif self.config["use_context"] == "static":
                context_list += [np.ones(self.context_input_size)]
        
        context_list = np.array(context_list).reshape([-1])

        if not self.config["use_context"] == "none":
            all_samples['context'] = context_list

        # Meta gradient Update
        # All Samples should be a list of list of dicts where the dims are (inner_adaptation_steps+1,num_workers,SamplesDict)
        # Should the whole computation graph be in master?
        print("Meta Update")
        with self.meta_grad_timer:
            all_samples["split"] = np.array(meta_split)
            for i in range(self.maml_optimizer_steps):
                fetches = self.workers.local_worker().learn_on_batch(all_samples)
                print(meta_split)
                #import pdb; pdb.set_trace()
            self.learner_stats = get_learner_stats(fetches)

        self.num_steps_sampled += all_samples.count
        self.num_steps_trained += all_samples.count

        return self.learner_stats

    def preprocess_context(self, env_configs):
        if isinstance(env_configs[0], dict):
            for i, config in enumerate(env_configs):
                env_configs[i] = np.concatenate([v.flatten() for v in config.values()])
        return env_configs

    def post_processing(self, samples, num_envs_per_worker):
        split_list = []
        for sample in samples:
            split_list.append(sample['obs'].shape[0])
            indexes = np.where(sample['dones']==True)[0]
            indexes = indexes+1

            reward_list = np.split(sample['rewards'], indexes)[:-1]
            observation_list = np.split(sample['obs'], indexes)[:-1]

            temp_list = []
            for i in range(0, len(reward_list)):
                temp_list.append({"rewards": reward_list[i], "observations": observation_list[i]})
            
            advantages = self._compute_samples_data(temp_list)
            sample["advantages"] = advantages
        return samples, split_list


    def _compute_samples_data(self, paths):
        assert type(paths) == list

        # 1) compute discounted rewards (returns)
        for idx, path in enumerate(paths):
            path["returns"] = self.discount_cumsum(path["rewards"], self.discount)

        # 2) fit baseline estimator using the path returns and predict the return baselines
        self.baseline.fit(paths, target_key="returns")
        all_path_baselines = [self.baseline.predict(path) for path in paths]

        # 3) compute advantages and adjusted rewards
        paths = self._compute_advantages(paths, all_path_baselines)

        # 4) stack path data
        advantages = np.concatenate([path["advantages"] for path in paths])

        advantages = self.normalize_advantages(advantages)
        return advantages

    def _compute_advantages(self, paths, all_path_baselines):
        assert len(paths) == len(all_path_baselines)

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = self.discount_cumsum(
                deltas, self.discount * self.gae_lambda)

        return paths

    def discount_cumsum(self, x, discount):
        """
        See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering

        Returns:
            (float) : y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
        """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def normalize_advantages(self, advantages):
        """
        Args:
            advantages (np.ndarray): np array with the advantages

        Returns:
            (np.ndarray): np array with the advantages normalized
        """
        return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)
            


