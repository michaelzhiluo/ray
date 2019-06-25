import gym
import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.tune.trainable import Trainable
from ray import tune

from easydict import EasyDict

from ray.rllib.env.multi_agent_env import MultiAgentEnv

def make_multiagent(args):
    class MultiEnv(MultiAgentEnv):
        def __init__(self):
            self.agents = [gym.make(args.env) for _ in range(args.num_agents)]
            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space

        def reset(self):
            self.dones = set()
            return {i: a.reset() for i, a in enumerate(self.agents)}

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                if done[i]:
                    self.dones.add(i)
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

    return MultiEnv   

def make_fed_env(args):   
    FedEnv = make_multiagent(args)
    env_name = "multienv_FedRL"
    register_env(env_name, lambda _: FedEnv())
    return env_name

def gen_policy_graphs(args):
    single_env = gym.make(args.env)
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    policy_graphs = {f'agent_{i}': (args.graph_type, obs_space, act_space, {}) 
         for i in range(args.num_agents)}
    return policy_graphs

def policy_mapping_fn(agent_id):
    return f'agent_{agent_id}'

def change_weights(weights, i):
    """
    Helper function for FedQ-Learning
    """
    dct = {}
    for key, val in weights.items():
        # new_key = key
        still_here = key[:6]
        there_after = key[7:]
        # new_key[6] = i
        new_key = still_here + str(i) + there_after
        dct[new_key] = val
    # print(dct.keys())
    return dct

def synchronize(agent, weights, num_agents):
    """
    Helper function to synchronize weights of the multiagent
    """
    weights_to_set = {f'agent_{i}': weights 
         for i in range(num_agents)}
    # weights_to_set = {f'agent_{i}': change_weights(weights, i) 
    #    for i in range(num_agents)}
    # print(weights_to_set)
    agent.set_weights(weights_to_set)

def uniform_initialize(agent, num_agents):
    """
    Helper function for uniform initialization
    """
    new_weights = agent.get_weights(["agent_0"]).get("agent_0")
    # print(new_weights.keys())
    synchronize(agent, new_weights, num_agents)

def compute_softmax_weighted_avg(weights, alphas, num_agents, temperature=1):
    """
    Helper function to compute weighted avg of weights weighted by alphas
    Weights and alphas must have same keys. Uses softmax.
    params:
        weights - dictionary
        alphas - dictionary
    returns:
        new_weights - array
    """
    def softmax(x, beta=temperature, length=num_agents):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(beta * (x - np.max(x)))
        return (e_x / e_x.sum()).reshape(length, 1)
    
    alpha_vals = np.array(list(alphas.values()))
    soft = softmax(alpha_vals)
    weight_vals = np.array(list(weights.values()))
    new_weights = sum(np.multiply(weight_vals, soft))
    return new_weights

def reward_weighted_update(agent, result, num_agents):
    """
    Helper function to synchronize weights of multiagent via
    reward-weighted avg of weights
    """
    return softmax_reward_weighted_update(agent, result, num_agents, temperature=0)

def softmax_reward_weighted_update(agent, result, num_agents, temperature=1):
    """
    Helper function to synchronize weights of multiagent via
    softmax reward-weighted avg of weights with specific temperature
    """
    all_weights = agent.get_weights()
    policy_reward_mean = result['policy_reward_mean']
    episode_reward_mean = result['episode_reward_mean']
    # try:
    if policy_reward_mean:
        new_weights = compute_softmax_weighted_avg(all_weights, policy_reward_mean, num_agents, temperature=temperature)
        synchronize(agent, new_weights, num_agents)
    # except Exception as e:
        # print(e)
        # print(f"Couldn't update, probably because episode_reward_mean is {episode_reward_mean}")
        

class FedRLActor(Trainable):
    def __init__(self, config=None, logger_creator=None):
        super().__init__(config, logger_creator)
        agent_config = config["for_agent"]
        algo_config = config["for_algo"]
        self.temperature = algo_config["temperature"]
        self.agent = algo_config["agent_type"](config=agent_config)
        self.num_agents = len(agent_config["multiagent"]["policy_graphs"].keys())
        uniform_initialize(self.agent, self.num_agents)
    def _train(self):
        result = self.agent.train()
        # modify reporting for multiagent a bit ONLY when same MDP
        result['episode_reward_mean'] = result['episode_reward_mean']/self.num_agents
        result['episode_reward_max'] = result['episode_reward_max']/self.num_agents
        result['episode_reward_min'] = result['episode_reward_min']/self.num_agents
        print(pretty_print(result))
        # Do update
#         if result['episodes_total'] > 5:
        #     if strategy == REWARD:
        softmax_reward_weighted_update(self.agent, result, self.num_agents, temperature=self.temperature)

        # reward_weighted_update(self.agent, result, self.num_agents)
        # print("finished reward weighted update")
        return result
    def _save(self, checkpoint_dir=None):
        return self.agent.save(checkpoint_dir)
    def _restore(self, checkpoint):
        return self.agent.restore(checkpoint)
    def stop(self):
        self.agent._stop()

slow_start = True

def manage_curriculum(info):
    global slow_start
    print("Manage Curriculum callback called on phase {}".format(slow_start))
    result = info["result"]
    if slow_start and result["training_iteration"] % 100 == 0 and result["training_iteration"] != 0:
        slow_start = False
        agent = info["agent"]
        agent.optimizer.train_batch_size *= 5
def clear_buffer(info):
    agent = info["trainer"]
    optimizer = agent.optimizer
    print(f"Clearing buffer of len {len(optimizer.episode_history)}")
    optimizer.episode_history = []
    print(f"Cleared buffer to len {len(optimizer.episode_history)}")
          
def fed_learning(args):
    ray.init(ignore_reinit_error=True)
    policy_graphs = gen_policy_graphs(args)
    multienv_name = make_fed_env(args)
    tune.run(
        FedRLActor,
        name=f"{args.env}-{args.agent_type}-{args.num_agents}",
        # stop={"timesteps_total": args.timesteps},
        # stop={"episodes_total": args.episodes},
        stop={"episode_reward_mean": 9800},
        config={
            "for_agent": {
                "multiagent": {
                    "policy_graphs": policy_graphs,
                    "policy_mapping_fn": tune.function(lambda agent_id: f'agent_{agent_id}'),
                    # "policies_to_train":
                },
#                 "train_batch_size": 128,
                "env": multienv_name,
                "gamma": 0.99,
                "lambda": 0.95,
                "kl_coeff": 1.0,
                "num_sgd_iter": 32,
                "lr": .0003 * args.num_agents,
                "vf_loss_coeff": 0.5,
                "clip_param": 0.2,
                "sgd_minibatch_size": 4096,
                "train_batch_size": 65536,
                "grad_clip": 0.5,
                "batch_mode": "truncate_episodes",
                "observation_filter": "MeanStdFilter",
                # "lr": tune.grid_search(args.lrs),
#                 "simple_optimizer": True,
                "callbacks":{
                    "on_train_result": tune.function(clear_buffer),
                },
                "num_workers": args.num_workers,
                "num_gpus": 1,
                # "num_gpus_per_worker": 1.0/args.num_workers,
                # "num_cpus_per_worker": 0.5,
            },
            "for_algo": {
               # "num_iters": args.num_iters,
                # "temperature": 1.0,
                "temperature": tune.grid_search(args.temperatures),
                "agent_type": args.agent_type
            },
        },
        resources_per_trial={
            "gpu": 1.0,
            "cpu": 1.0,
            # "extra_gpu": 1.0,
            # "cpu": 0.5,
            "extra_cpu": 7.0,
        },
        checkpoint_at_end=True
    )


# print(e)

args = EasyDict({
    'num_agents': 5,
    'num_workers': 16,
    'temperatures': [0, 8, 0.5, 4, 2, 1, 16],
    'timesteps': 1e7,
    # 'lr': 5e-4,
    'lrs': [5e-5, 5e-4, 5e-3],
    'episodes': 150,
#     'num_iters': 100,
    'env': 'HalfCheetah-v2',
    'name': 'fed_experiment',
    # 'agent_type': ray.rllib.agents.ddpg.ddpg.DDPGAgent,
    # 'graph_type': ray.rllib.agents.ddpg.ddpg_policy_graph.DDPGPolicyGraph,
    'agent_type': ray.rllib.agents.ppo.ppo.PPOTrainer,
    'graph_type': ray.rllib.agents.ppo.ppo_policy_graph.PPOPolicyGraph,
#     'graph_type': ray.rllib.agents.pg.pg_policy_graph.PGPolicyGraph,
#     'graph_type': ray.rllib.agents.a3c.a3c_tf_policy_graph.A3CPolicyGraph,
#     'agent_type': ray.rllib.agents.a3c.a3c.A3CAgent,
#     'agent_type': ray.rllib.agents.pg.pg.PGTrainer,
})
# train
fed_learning(args)
# eval