"""Adapted from A3CTFPolicy to add V-trace.

Keep in sync with changes to A3CTFPolicy and VtraceSurrogatePolicy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import ray
import numpy as np
from ray.rllib.agents.impala import vtrace
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy, \
    LearningRateSchedule
from ray.rllib.models.action_dist import Categorical
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.explained_variance import explained_variance
from ray.rllib.utils import try_import_tf
from ray.rllib.evaluation.postprocessing import compute_advantages, \
    Postprocessing
from ray.rllib.models.misc import normc_initializer, get_activation_fn

tf = try_import_tf()

# Frozen logits of the policy that computed the action
BEHAVIOUR_LOGITS = "behaviour_logits"
INNER_LR = 0.1


class PPOLoss(object):
    def __init__(self,
                 obs,
                 model,
                 var_list,
                 action_space,
                 value_targets,
                 advantages,
                 actions,
                 logits,
                 vf_preds,
                 curr_action_dist,
                 value_fn,
                 cur_kl_coeff,
                 valid_mask,
                 entropy_coeff=0,
                 clip_param=0.1,
                 vf_clip_param=0.1,
                 vf_loss_coeff=1.0,
                 context = None
                 ):
        """Constructs the loss for Proximal Policy Objective.

        Arguments:
            action_space: Environment observation space specification.
            value_targets (Placeholder): Placeholder for target values; used
                for GAE.
            actions (Placeholder): Placeholder for actions taken
                from previous model evaluation.
            advantages (Placeholder): Placeholder for calculated advantages
                from previous model evaluation.
            logits (Placeholder): Placeholder for logits output from
                previous model evaluation.
            vf_preds (Placeholder): Placeholder for value function output
                from previous model evaluation.
            curr_action_dist (ActionDistribution): ActionDistribution
                of the current model.
            value_fn (Tensor): Current value function output Tensor.
            cur_kl_coeff (Variable): Variable holding the current PPO KL
                coefficient.
            valid_mask (Tensor): A bool mask of valid input elements (#2992).
            entropy_coeff (float): Coefficient of the entropy regularizer.
            clip_param (float): Clip parameter
            vf_clip_param (float): Clip parameter for the value function
            vf_loss_coeff (float): Coefficient of the value function loss
            use_gae (bool): If true, use the Generalized Advantage Estimator.
        """
        self.policy_vars = {}
        for var in var_list:
            self.policy_vars[var.name] = var


        def reduce_mean_valid(t):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))
        print("PPO LOSS")

        dist_cls, _ = ModelCatalog.get_action_dist(action_space, {})
        prev_dist = dist_cls(logits)
        logp_ratio = tf.exp(
            curr_action_dist.logp(actions) - prev_dist.logp(actions))

        action_kl = prev_dist.kl(curr_action_dist)
        self.mean_kl = reduce_mean_valid(action_kl)

        self.kl_loss = cur_kl_coeff[0] * self.mean_kl

        surrogate_loss = advantages * logp_ratio 
        self.mean_policy_loss = reduce_mean_valid(-surrogate_loss)

        loss = reduce_mean_valid(
            -surrogate_loss)
        loss = tf.Print(loss, ["Worker Loss", loss, self.mean_kl])
        self.loss = loss + 0.0*self.kl_loss

class MAMLLoss(object):

    def __init__(self,
            model,
            config,
            dist_class, 
            value_targets,
            advantages,
            actions,
            behaviour_logits,
            vf_preds,
            cur_kl_coeff,
            valid_mask, 
            policy_vars, 
            obs, 
            num_tasks,
            split,
            inner_adaptation_steps=1,
            entropy_coeff=0,
            clip_param=0.3,
            vf_clip_param=0.1,
            vf_loss_coeff=1.0,
            context=None):

        print("MAML Loss")
        self.config = config
        self.num_tasks = num_tasks
        self.inner_adaptation_steps = inner_adaptation_steps
        self.clip_param = clip_param
        self.dist_class = dist_class
        self.cur_kl_coeff = cur_kl_coeff

        # Split episode tensors into [inner_adaptation_steps+1, num_tasks, -1]
        self.obs = self.split_placeholders(obs, split)
        self.actions = self.split_placeholders(actions, split)
        self.behaviour_logits = self.split_placeholders(behaviour_logits, split)
        self.advantages = self.split_placeholders(advantages, split)
        self.value_targets = self.split_placeholders(value_targets, split)
        self.vf_preds = self.split_placeholders(vf_preds, split)
        self.valid_mask = self.split_placeholders(valid_mask, split)
        pi_new_init = self.split_placeholders(model.outputs, split)
        contexts = [None]*num_tasks
        if context is not None:
            contexts = tf.split(context, self.num_tasks)
        
        #  Construct name to tensor dictionary
        self.policy_vars = {}
        for var in policy_vars:
            self.policy_vars[var.name] = var
        print(self.policy_vars)

        # Calculate pi_new for PPO
        pi_new_logits, current_policy_vars, value_fns = [], [], []
        for i in range(self.num_tasks):
            pi_new, value_fn = self.feed_forward(self.obs[0][i], self.policy_vars, policy_config = config["model"], context = contexts[i])
            pi_new_logits.append(pi_new)
            value_fns.append(value_fn)
            current_policy_vars.append(self.policy_vars)

        self.pi_new_logits = pi_new_logits

        inner_kls = []
        inner_ppo_loss = []
        # Recompute weights for inner-adaptation (since this is also incoporated in meta objective loss function)
        for step in range(self.inner_adaptation_steps):
            kls = []
            for i in range(self.num_tasks):
                # Loss Function Shenanigans
                ppo_loss, _, kl_loss, _, _ = self.PPOLoss(
                    actions = self.actions[step][i],
                    curr_logits = pi_new_logits[i],
                    behaviour_logits = self.behaviour_logits[step][i],
                    advantages = self.advantages[step][i],
                    value_fn = value_fns[i],
                    value_targets = self.value_targets[step][i],
                    vf_preds = self.vf_preds[step][i],
                    cur_kl_coeff = cur_kl_coeff,
                    valid_mask = self.valid_mask[step][i],
                    entropy_coeff = entropy_coeff,
                    clip_param = clip_param,
                    vf_clip_param = vf_clip_param,
                    vf_loss_coeff = vf_loss_coeff,
                    clip_loss = False
                    )

                adapted_policy_vars = self.compute_updated_variables(ppo_loss, current_policy_vars[i])
                pi_new_logits[i], value_fns[i] = self.feed_forward(self.obs[step+1][i], adapted_policy_vars, policy_config=config["model"], context = contexts[i])
                current_policy_vars[i] = adapted_policy_vars
                kls.append(kl_loss)
                inner_ppo_loss.append(ppo_loss)
            self.kls = kls
            inner_kls.append(kls)

        mean_inner_kl = tf.stack([tf.reduce_mean(tf.stack(inner_kl)) for inner_kl in inner_kls])
        self.mean_inner_kl = mean_inner_kl
        ppo_obj = []
        for i in range(self.num_tasks):
            ppo_loss, _, kl_loss, _, _ = self.PPOLoss(
                    actions = self.actions[self.inner_adaptation_steps][i],
                    curr_logits = pi_new_logits[i],
                    behaviour_logits = self.behaviour_logits[self.inner_adaptation_steps][i],
                    advantages = self.advantages[self.inner_adaptation_steps][i],
                    value_fn = value_fns[i],
                    value_targets = self.value_targets[self.inner_adaptation_steps][i],
                    vf_preds = self.vf_preds[self.inner_adaptation_steps][i],
                    cur_kl_coeff = cur_kl_coeff,
                    valid_mask = self.valid_mask[self.inner_adaptation_steps][i],
                    entropy_coeff = entropy_coeff,
                    clip_param = clip_param,
                    vf_clip_param = vf_clip_param,
                    vf_loss_coeff = vf_loss_coeff,
                    clip_loss = True
                    )
            ppo_obj.append(ppo_loss)
        self.kl_loss = tf.reduce_mean(tf.multiply(cur_kl_coeff, mean_inner_kl))
        self.loss = tf.reduce_mean(tf.stack(ppo_obj, axis=0)) + self.kl_loss
        self.loss = tf.Print(self.loss, ["Meta-Loss", self.loss, mean_inner_kl])

    def PPOLoss(self,
         actions,
         curr_logits,
         behaviour_logits,
         advantages,
         value_fn,
         value_targets,
         vf_preds,
         cur_kl_coeff,
         valid_mask,
         entropy_coeff,
         clip_param,
         vf_clip_param,
         vf_loss_coeff,
         clip_loss=False):
    
        def reduce_mean_valid(t, valid_mask):
            return tf.reduce_mean(tf.boolean_mask(t, valid_mask))
        pi_new_dist = self.dist_class(curr_logits)
        pi_old_dist = self.dist_class(behaviour_logits)

        surr_loss = reduce_mean_valid(self.surrogate_loss(actions, pi_new_dist, pi_old_dist, advantages, clip_loss), valid_mask)
        kl_loss = reduce_mean_valid(self.kl_loss(pi_new_dist, pi_old_dist), valid_mask)
        vf_loss = reduce_mean_valid(self.vf_loss(value_fn, value_targets, vf_preds, vf_clip_param), valid_mask)
        entropy_loss = reduce_mean_valid(self.entropy_loss(pi_new_dist), valid_mask)

        total_loss = - surr_loss 
        return total_loss, surr_loss, kl_loss, vf_loss, entropy_loss

    def surrogate_loss(self, actions, curr_dist, prev_dist, advantages, clip_loss):
        pi_new_logp = curr_dist.logp(actions)
        pi_old_logp = prev_dist.logp(actions)

        logp_ratio = tf.exp(pi_new_logp - pi_old_logp)
        if clip_loss:
            return tf.minimum(
            advantages * logp_ratio,
            advantages * tf.clip_by_value(logp_ratio, 1 - self.clip_param,
                                          1 + self.clip_param))
        return advantages*logp_ratio
    
    def kl_loss(self, curr_dist, prev_dist):
        return prev_dist.kl(curr_dist)

    def entropy_loss(self, dist):
        return dist.entropy()

    def vf_loss(self, value_fn, value_targets, vf_preds, vf_clip_param=0.1):
        # GAE Value Function Loss
        vf_loss1 = tf.square(value_fn - value_targets)
        vf_clipped = vf_preds + tf.clip_by_value(
            value_fn - vf_preds, -vf_clip_param, vf_clip_param)
        vf_loss2 = tf.square(vf_clipped - value_targets)
        vf_loss = tf.maximum(vf_loss1, vf_loss2)
        return vf_loss

    def feed_forward(self, obs, policy_vars, policy_config, context = None):
        # Hacky for now, assumes it is Fully connected network in models/fcnet.py, Conv net implemented on a later date
        # Returns pi_new_logits and value_function_prediction
        def fc_network(inp, network_vars, hidden_nonlinearity, output_nonlinearity, policy_config, hyper_vars = None, context=None):
            context_input_size = policy_config["context_input_size"]
            hidden_sizes = policy_config["fcnet_hiddens"]
            bias_added = False
            if context is not None:
                c = context

            if context is not None and policy_config["concat_context"]:
                app = c[0][0:context_input_size]
                app = tf.tile(app, tf.shape(inp)[0:1])
                app = tf.reshape(app, [-1, context_input_size])
                inp = tf.concat([inp, app], axis =1)

            if context is not None and not policy_config["concat_context"]:
                for name, param in hyper_vars.items():
                    if "kernel" in name:
                        c = tf.matmul(c, param)
                    elif "bias" in name:
                        c= tf.add(c, param)
                        bias_added = True
                    else:
                        raise NameError

                    if bias_added:
                        if "hyper_out" not in name:
                            c = hidden_nonlinearity(c)
                        elif "hyper_out" in name:
                            c = output_nonlinearity(c)
                        else:
                            raise NameError
                        bias_added = False
                film_params = tf.split(c, [val for val in hidden_sizes for _ in (0, 1)],axis=1)

            x = inp
            for name, param in network_vars.items():
                if "kernel" in name:
                    x = tf.matmul(x, param)
                elif "bias" in name:
                    x = tf.add(x, param)
                    bias_added = True
                else:
                    raise NameError

                if bias_added:
                    if "fc_out" not in name:
                        x = hidden_nonlinearity(x)
                        if context is not None and not policy_config["concat_context"]:
                            x =  tf.einsum('ij,kj->ij', x, film_params.pop(0))+ film_params.pop(0)
                    elif "fc_out" in name:
                        x = output_nonlinearity(x)
                    else:
                        raise NameError
                    bias_added = False
            return x

        if context is not None:
            context = tf.reshape(context, (-1,policy_config["concat_input_size"]))
        policyn_vars = {}
        valuen_vars = {}
        hyper_vars = {}
        log_std = None
        for name, param in policy_vars.items():
            if "hyper_film" in name:
                hyper_vars[name] = param
            elif 'value_function' in name:
                valuen_vars[name] = param
            elif "log_std" in name:
                log_std = param
            else:
                policyn_vars[name] = param

        output_nonlinearity = tf.identity
        hidden_nonlinearity = get_activation_fn(policy_config["fcnet_activation"])
        
        pi_new_logits = fc_network(obs, policyn_vars, hidden_nonlinearity, output_nonlinearity, policy_config, hyper_vars = hyper_vars, context=context)
        if log_std is not None:
            pi_new_logits = tf.concat(
                [pi_new_logits, 0.0 * pi_new_logits + log_std], 1)
        value_fn = fc_network(obs, valuen_vars, hidden_nonlinearity, output_nonlinearity, policy_config)

        return pi_new_logits, tf.reshape(value_fn, [-1])

    def compute_updated_variables(self, loss, network_vars):
        grad = tf.gradients(loss, list(network_vars.values()))
        adapted_vars = {}
        counter =0
        for i, tup in enumerate(network_vars.items()):
            name, var = tup
            if grad[i] is None:
                adapted_vars[name] = var
            else:
                adapted_vars[name] = var - 0.1*grad[i] #tf.multiply(self.step_sizes[name], grad[i])
        return adapted_vars

    def split_placeholders(self, placeholder, split):
        inner_placeholder_list = tf.split(placeholder, tf.math.reduce_sum(split, axis=1), axis=0)
        placeholder_list = []
        for index, split_placeholder in enumerate(inner_placeholder_list):
            placeholder_list.append(tf.split(split_placeholder, split[index], axis=0))
        '''
        inner_placeholder_list = tf.split(placeholder, self.inner_adaptation_steps+1, axis=0)
        placeholder_list = []
        for split_placeholder in inner_placeholder_list:
            placeholder_list.append(tf.split(split_placeholder, self.num_tasks, axis=0))
        '''
        return placeholder_list

    def _create_step_size_vars(self):
        # Step sizes
        step_sizes = dict()
        for key, param in self.policy_vars.items():
            shape = param.get_shape().as_list()
            init_stepsize = np.ones(shape, dtype=np.float32) * self.config["inner_lr"]
            step_sizes[key] = tf.Variable(initial_value=init_stepsize,
                                          dtype=tf.float32, trainable=False)
        self.step_sizes = step_sizes


class MAMLPostprocessing(object):
    """Adds the policy logits, VF preds, and advantages to the trajectory."""

    @override(TFPolicy)
    def extra_compute_action_fetches(self):
        return dict(
            TFPolicy.extra_compute_action_fetches(self), **{
                SampleBatch.VF_PREDS: self.value_function,
                BEHAVIOUR_LOGITS: self.logits
            })

    @override(Policy)
    def postprocess_trajectory(self,
                               sample_batch,
                               other_agent_batches=None,
                               episode=None):
        completed = sample_batch["dones"][-1]
        if completed:
            last_r = 0.0
        else:
            next_state = []
            for i in range(len(self.model.state_in)):
                next_state.append([sample_batch["state_out_{}".format(i)][-1]])
            last_r = self._value(sample_batch[SampleBatch.NEXT_OBS][-1],
                                 sample_batch[SampleBatch.ACTIONS][-1],
                                 sample_batch[SampleBatch.REWARDS][-1],
                                 *next_state)
        batch = compute_advantages(
            sample_batch,
            last_r,
            self.config["gamma"],
            self.config["lambda"],
            use_gae=self.config["use_gae"])
        return batch


class MAMLTFPolicy(LearningRateSchedule, MAMLPostprocessing, TFPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 config,
                 existing_inputs=None):
        """
        Arguments:
            observation_space: Environment observation space specification.
            action_space: Environment action space specification.
            config (dict): Configuration values for PPO graph.
            existing_inputs (list): Optional list of tuples that specify the
                placeholders upon which the graph should be built upon.
        """
        config = dict(ray.rllib.agents.maml.maml.DEFAULT_CONFIG, **config)
        self.sess = tf.get_default_session()
        self.action_space = action_space
        self.config = config
        self.kl_coeff_val = [self.config["kl_coeff"]]*self.config["inner_adaptation_steps"]
        self.kl_target = self.config["kl_target"]
        dist_cls, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"])

        if existing_inputs:
            obs_ph, value_targets_ph, adv_ph, act_ph, \
                logits_ph, vf_preds_ph, prev_actions_ph, prev_rewards_ph = \
                existing_inputs[:8]
            existing_state_in = existing_inputs[8:-1]
            existing_seq_lens = existing_inputs[-1]
        else:
            obs_ph = tf.placeholder(
                tf.float32,
                name="obs",
                shape=(None, ) + observation_space.shape)
            adv_ph = tf.placeholder(
                tf.float32, name="advantages", shape=(None, ))
            act_ph = ModelCatalog.get_action_placeholder(action_space)
            logits_ph = tf.placeholder(
                tf.float32, name="logits", shape=(None, logit_dim))
            vf_preds_ph = tf.placeholder(
                tf.float32, name="vf_preds", shape=(None, ))
            value_targets_ph = tf.placeholder(
                tf.float32, name="value_targets", shape=(None, ))
            prev_actions_ph = ModelCatalog.get_action_placeholder(action_space)
            prev_rewards_ph = tf.placeholder(
                tf.float32, [None], name="prev_reward")
            split_ph = tf.placeholder(tf.int32, name="Meta-Update-Splitting", shape=(self.config["inner_adaptation_steps"]+1, self.config["num_workers"]))
            if not self.config["use_context"] == "none":
                context_ph = tf.placeholder(tf.float32, [None], name="context")
            else:
                context_ph = None
            existing_state_in = None
            existing_seq_lens = None
        self.observations = obs_ph
        self.prev_actions = prev_actions_ph
        self.prev_rewards = prev_rewards_ph

        self.loss_in = [
            (SampleBatch.CUR_OBS, obs_ph),
            (Postprocessing.VALUE_TARGETS, value_targets_ph),
            (Postprocessing.ADVANTAGES, adv_ph),
            (SampleBatch.ACTIONS, act_ph),
            (BEHAVIOUR_LOGITS, logits_ph),
            (SampleBatch.VF_PREDS, vf_preds_ph),
            (SampleBatch.PREV_ACTIONS, prev_actions_ph),
            (SampleBatch.PREV_REWARDS, prev_rewards_ph),
        ]

        input_dict = {
                "obs": obs_ph,
                "prev_actions": prev_actions_ph,
                "prev_rewards": prev_rewards_ph,
                "is_training": self._get_is_training_placeholder(),
            }

        self.loss_in.append(("split", split_ph))
        if not self.config["use_context"] == "none":
            self.loss_in.append(("context", context_ph))
            input_dict["context"] = context_ph

        self.model = ModelCatalog.get_model(
            input_dict,
            observation_space,
            action_space,
            logit_dim,
            self.config["model"],
            state_in=existing_state_in,
            seq_lens=existing_seq_lens)

        # KL Coefficient
        self.kl_coeff = tf.get_variable(
            initializer=tf.constant_initializer(self.kl_coeff_val),
            name="kl_coeff",
            shape=(self.config["inner_adaptation_steps"]),
            trainable=False,
            dtype=tf.float32)

        self.logits = self.model.outputs
        curr_action_dist = dist_cls(self.logits)
        self.sampler = curr_action_dist.sample()
        if self.config["use_gae"]:
            vf_config = self.config["model"].copy()
            # Do not split the last layer of the value function into
            # mean parameters and standard deviation parameters and
            # do not make the standard deviations free variables.
            vf_config["free_log_std"] = False
            if vf_config["use_lstm"]:
                vf_config["use_lstm"] = False
                logger.warning(
                    "It is not recommended to use a LSTM model with "
                    "vf_share_layers=False (consider setting it to True). "
                    "If you want to not share layers, you can implement "
                    "a custom LSTM model that overrides the "
                    "value_function() method.")
            with tf.variable_scope("value_function"):
                self.value_function = ModelCatalog.get_model({
                    "obs": obs_ph,
                    "prev_actions": prev_actions_ph,
                    "prev_rewards": prev_rewards_ph,
                    "is_training": self._get_is_training_placeholder(),
                }, observation_space, action_space, 1, vf_config).outputs
                self.value_function = tf.reshape(self.value_function, [-1])
        else:
            self.value_function = tf.zeros(shape=tf.shape(obs_ph)[:1])

        if self.model.state_in:
            max_seq_len = tf.reduce_max(self.model.seq_lens)
            mask = tf.sequence_mask(self.model.seq_lens, max_seq_len)
            mask = tf.reshape(mask, [-1])
        else:
            mask = tf.ones_like(adv_ph, dtype=tf.bool)

        self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              tf.get_variable_scope().name)
        # If you are a task or worker, compute normal PPO Loss for inner adaptation update
        self.worker_index = self.config["worker_index"]
        if self.config["worker_index"]:
            self.loss_obj = PPOLoss(
                obs_ph,
                self.model,
                self.var_list,
                action_space,
                value_targets_ph,
                adv_ph,
                act_ph,
                logits_ph,
                vf_preds_ph,
                curr_action_dist,
                self.value_function,
                self.kl_coeff,
                mask,
                entropy_coeff=self.config["entropy_coeff"],
                clip_param=self.config["clip_param"],
                vf_clip_param=self.config["vf_clip_param"],
                vf_loss_coeff=self.config["vf_loss_coeff"])
        else:
            # If you are the master learner, compute meta-gradient objective
            self.loss_obj = MAMLLoss(
                model = self.model,
                config = self.config,
                dist_class = dist_cls, 
                value_targets = value_targets_ph,
                advantages = adv_ph,
                actions = act_ph,
                behaviour_logits = logits_ph,
                vf_preds = vf_preds_ph,
                cur_kl_coeff = self.kl_coeff,
                valid_mask = mask, 
                policy_vars = self.var_list, 
                obs = obs_ph,  
                num_tasks = self.config["num_workers"],
                split = split_ph,
                inner_adaptation_steps=1,
                entropy_coeff=self.config["entropy_coeff"],
                clip_param=self.config["clip_param"],
                vf_clip_param=self.config["vf_clip_param"],
                vf_loss_coeff=self.config["vf_loss_coeff"],
                context = context_ph,
                )

        if self.config["worker_index"]:
            LearningRateSchedule.__init__(self, self.config["inner_lr"], self.config["lr_schedule"], self.config["worker_index"])
        else:
            LearningRateSchedule.__init__(self, self.config["lr"],
                                          self.config["lr_schedule"], self.config["worker_index"])
        TFPolicy.__init__(
            self,
            observation_space,
            action_space,
            self.sess,
            obs_input=obs_ph,
            action_sampler=self.sampler,
            action_prob=curr_action_dist.sampled_action_prob(),
            loss=self.loss_obj.loss,
            model=self.model,
            loss_inputs=self.loss_in,
            state_inputs=self.model.state_in,
            state_outputs=self.model.state_out,
            prev_action_input=prev_actions_ph,
            prev_reward_input=prev_rewards_ph,
            seq_lens=self.model.seq_lens,
            max_seq_len=config["model"]["max_seq_len"],
            context = context_ph)


        self.sess.run(tf.global_variables_initializer())
        self.explained_variance = explained_variance(value_targets_ph,
                                                     self.value_function)
        self.stats_fetches = {
            "cur_kl_coeff": self.kl_coeff,
            "cur_lr": tf.cast(self.cur_lr, tf.float64),
            "total_loss": self.loss_obj.loss,
            "kl_loss": self.loss_obj.kl_loss
            #"policy_loss": self.loss_obj.mean_policy_loss,
            #"vf_loss": self.loss_obj.mean_vf_loss,
            #"vf_explained_var": self.explained_variance,
            #"kl": self.loss_obj.mean_kl,
            #"entropy": self.loss_obj.mean_entropy
        }
        if not self.config["worker_index"]:
            self.stats_fetches["kl"] =  self.loss_obj.mean_inner_kl

    @override(TFPolicy)
    def copy(self, existing_inputs):
        """Creates a copy of self using existing input placeholders."""
        return MAMLTFPolicy(
            self.observation_space,
            self.action_space,
            self.config,
            existing_inputs=existing_inputs)

    @override(TFPolicy)
    def gradients(self, optimizer, loss):
        grads = optimizer.compute_gradients(
            loss, colocate_gradients_with_ops=True)
        return grads

    @override(Policy)
    def get_initial_state(self):
        return self.model.state_init

    @override(TFPolicy)
    def extra_compute_grad_fetches(self):
        return {LEARNER_STATS_KEY: self.stats_fetches}

    def update_kls(self, sampled_kls):
        for i, kl in enumerate(sampled_kls): 
            if kl < self.kl_target/1.5:
                self.kl_coeff_val[i] *= 0.5
            elif kl > 1.5 * self.kl_target:
                self.kl_coeff_val[i] *= 2.0
            '''     
            if kl > 2.0 * self.kl_target:
                self.kl_coeff_val[i] *= 1.5
            elif kl < 0.5 * self.kl_target:
                self.kl_coeff_val[i] *= 0.5
            '''
        self.kl_coeff.load(self.kl_coeff_val, session=self.sess)
        return self.kl_coeff_val

    def _value(self, ob, prev_action, prev_reward, *args):
        feed_dict = {
            self.observations: [ob],
            self.prev_actions: [prev_action],
            self.prev_rewards: [prev_reward],
            self.model.seq_lens: [1]
        }
        assert len(args) == len(self.model.state_in), \
            (args, self.model.state_in)
        for k, v in zip(self.model.state_in, args):
            feed_dict[k] = v
        vf = self.sess.run(self.value_function, feed_dict)
        return vf[0]
