from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging 
import ray
from ray.rllib.agents.maml.maml_policy import MAMLTFPolicy
from ray.rllib.agents.trainer import Trainer, with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncSamplesOptimizer, LocalMultiGPUOptimizer, MAMLOptimizer
from ray.rllib.utils.annotations import override
from ray.tune.trainable import Trainable
from ray.tune.trial import Resources

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # If true, use the Generalized Advantage Estimator (GAE)
    # with a value function, see https://arxiv.org/pdf/1506.02438.pdf.
    "use_gae": True,
    # GAE(lambda) parameter
    "lambda": 1.0,
    # Initial coefficient for KL divergence
    "kl_coeff": 0.2,
    # Size of batches collected from each worker
    "sample_batch_size": 200,
    # Number of timesteps collected for each SGD round
    "train_batch_size": 4000,
    # Total SGD batch size across all devices for SGD
    "sgd_minibatch_size": 128,
    # Number of SGD iterations in each outer loop
    "num_sgd_iter": 30,
    # Stepsize of SGD
    "lr": 5e-5,
    # Learning rate schedule
    "lr_schedule": None,
    # Share layers for value function
    "vf_share_layers": False,
    # Coefficient of the value function loss
    "vf_loss_coeff": 1.0,
    # Coefficient of the entropy regularizer
    "entropy_coeff": 0.0,
    # PPO clip parameter
    "clip_param": 0.3,
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    "vf_clip_param": 10.0,
    # If specified, clip the global norm of gradients by this amount
    "grad_clip": None,
    # Target value for KL divergence
    "kl_target": 0.01,
    # Whether to rollout "complete_episodes" or "truncate_episodes"
    "batch_mode": "truncate_episodes",
    # Which observation filter to apply to the observation
    "observation_filter": "NoFilter",
    # (Deprecated) Use the sampling behavior as of 0.6, which launches extra
    # sampling tasks for performance but can waste a large portion of samples.
    "straggler_mitigation": False,

    "inner_adaptation_steps": 1,

    "maml_optimizer_steps": 5,
})
# __sphinx_doc_end__
# yapf: enable

class MAMLTrainer(Trainer):

    _name = "MAML"
    _default_config = DEFAULT_CONFIG
    _policy_graph = MAMLTFPolicy

    @override(Trainer)
    def _init(self, config, env_creator):
        self._validate_config()
        self.workers = self._make_workers(
            env_creator, self._policy_graph, config, config["num_workers"])
        self.optimizer = MAMLOptimizer(
            self.workers,
            inner_adaptation_steps=config["inner_adaptation_steps"],
            train_batch_size=config["train_batch_size"],
            maml_optimizer_steps=config["maml_optimizer_steps"])

    def update_pre_post_stats(self, pre_res, post_res):
        pre_reward_max = pre_res['episode_reward_max']
        pre_reward_mean = pre_res['episode_reward_mean']
        pre_reward_min = pre_res['episode_reward_min']

        pre_res['episode_reward_max(post)'] = post_res['episode_reward_max']
        pre_res['episode_reward_mean(post)'] = post_res['episode_reward_mean']
        pre_res['episode_reward_min(post)'] = post_res['episode_reward_min']

        return pre_res

    @override(Trainer)
    def _train(self):
        weights = ray.put(self.workers.local_worker().get_weights())

        if "observation_filter" not in self.raw_user_config:
            # TODO(ekl) remove this message after a few releases
            logger.info(
                "Important! Since 0.7.0, observation normalization is no "
                "longer enabled by default. To enable running-mean "
                "normalization, set 'observation_filter': 'MeanStdFilter'. "
                "You can ignore this message if your environment doesn't "
                "require observation normalization.")
        prev_steps = self.optimizer.num_steps_sampled
        fetches = self.optimizer.step()
        if "kl" in fetches:
            # single-agent
            self.workers.local_worker().for_policy(
                lambda pi: pi.update_kl(fetches["kl"]))

        # Half of the workers collect data for pre-adaptation metrics, other half collect data for post-adaptation metrics
        half_index = int(len(self.workers.remote_workers())/2)
        first_half  = self.workers.remote_workers()[:half_index]
        second_half = self.workers.remote_workers()[half_index:]
        for e in first_half:
            e.set_weights.remote(weights)

        # Pre adaptation metrics
        res = self.optimizer.collect_metrics_pre(
            self.config["collect_metrics_timeout"],
            min_history=self.config["metrics_smoothing_episodes"],
            selected_workers=first_half)
        res.update(
            timesteps_this_iter=self.optimizer.num_steps_sampled - prev_steps,
            info=res.get("info", {}))
        print("Pre adaption stats", res)

        # Post adaptation metrics
        res1 = self.optimizer.collect_metrics_post(
            self.config["collect_metrics_timeout"],
            min_history=self.config["metrics_smoothing_episodes"],
            selected_workers=second_half)
        print("Post adaptation stats", res1)

        res = self.update_pre_post_stats(res, res1)
        
        # Warn about bad clipping configs
        if self.config["vf_clip_param"] <= 0:
            rew_scale = float("inf")
        elif res["policy_reward_mean"]:
            rew_scale = 0  # punt on handling multiagent case
        else:
            rew_scale = round(
                abs(res["episode_reward_mean"]) / self.config["vf_clip_param"],
                0)
        if rew_scale > 200:
            logger.warning(
                "The magnitude of your environment rewards are more than "
                "{}x the scale of `vf_clip_param`. ".format(rew_scale) +
                "This means that it will take more than "
                "{} iterations for your value ".format(rew_scale) +
                "function to converge. If this is not intended, consider "
                "increasing `vf_clip_param`.")
        #import pdb; pdb.set_trace()
        return res

    def _validate_config(self):
        if self.config["entropy_coeff"] < 0:
            raise DeprecationWarning("entropy_coeff must be >= 0")
        if self.config["sgd_minibatch_size"] > self.config["train_batch_size"]:
            raise ValueError(
                "Minibatch size {} must be <= train batch size {}.".format(
                    self.config["sgd_minibatch_size"],
                    self.config["train_batch_size"]))
        if (self.config["batch_mode"] == "truncate_episodes"
                and not self.config["use_gae"]):
            raise ValueError(
                "Episode truncation is not supported without a value "
                "function. Consider setting batch_mode=complete_episodes.")
        if (self.config["multiagent"]["policies"]
                and not self.config["simple_optimizer"]):
            logger.info(
                "In multi-agent mode, policies will be optimized sequentially "
                "by the multi-GPU optimizer. Consider setting "
                "simple_optimizer=True if this doesn't work for you.")
        if not self.config["vf_share_layers"]:
            logger.warning(
                "FYI: By default, the value function will not share layers "
                "with the policy model ('vf_share_layers': False).")
