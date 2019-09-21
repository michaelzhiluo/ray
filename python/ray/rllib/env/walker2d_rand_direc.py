import numpy as np
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from gym.spaces import Box
from rand_param_envs.gym.spaces import Box as OldBox
import inspect
import sys

class Serializable(object):

    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        try:
            if object.__getattribute__(self, "_serializable_initialized"):
                return
        except AttributeError:
            pass
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw]
            else:
                kwargs = dict()
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        out = type(self)(*d["__args"], **d["__kwargs"])
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()

        # Split the entries in kwargs between positional and keyword arguments
        # and update d['__args'] and d['__kwargs'], respectively.
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(obj.__init__)
        else:
            spec = inspect.getargspec(obj.__init__)
        in_order_args = spec.args[1:]

        d["__args"] = list(d["__args"])
        for kw, val in kwargs.items():
            if kw in in_order_args:
                d["__args"][in_order_args.index(kw)] = val
            else:
                d["__kwargs"][kw] = val

        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out


class Walker2DRandDirecEnv(gym.utils.EzPickle, MujocoEnv):
    def __init__(self):
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'walker2d.xml', 8)
        gym.utils.EzPickle.__init__(self)
    
    def sample_tasks(self, n_tasks):
        return np.random.choice((-1.0, 1.0), (n_tasks, ))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_direction = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_direction

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (self.goal_direction * (posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = False
        #done = not (height > 0.8 and height < 2.0 and
                    #ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5


class NormalizedEnv(Serializable):
    """
    Normalizes the environment class.

    Args:
        Env (gym.Env): class of the unnormalized gym environment
        scale_reward (float): scale of the reward
        normalize_obs (bool): whether normalize the observations or not
        normalize_reward (bool): whether normalize the reward or not
        obs_alpha (float): step size of the running mean and variance for the observations
        reward_alpha (float): step size of the running mean and variance for the observations
        normalization_scale (float): rescaled action magnitude

    """
    def __init__(self,
                 env=Walker2DRandDirecEnv(),
                 scale_reward=1.,
                 normalize_obs=False,
                 normalize_reward=False,
                 obs_alpha=0.001,
                 reward_alpha=0.001,
                 normalization_scale=10.,
                 ):
        Serializable.quick_init(self, locals())

        self._scale_reward = 1
        self._wrapped_env = env

        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha
        self._obs_mean = np.zeros(self.observation_space.shape)
        self._obs_var = np.ones(self.observation_space.shape)
        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.
        self._normalization_scale = normalization_scale


    @property
    def action_space(self):
        if isinstance(self._wrapped_env.action_space, Box):
            ub = np.ones(self._wrapped_env.action_space.shape) * self._normalization_scale
            return Box(-1 * ub, ub, dtype=np.float32)
        return self._wrapped_env.action_space

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

    def _update_obs_estimate(self, obs):
        o_a = self._obs_alpha
        self._obs_mean = (1 - o_a) * self._obs_mean + o_a * obs
        self._obs_var = (1 - o_a) * self._obs_var + o_a * np.square(obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        r_a = self._reward_alpha
        self._reward_mean = (1 - r_a) * self._reward_mean + r_a * reward
        self._reward_var = (1 - r_a) * self._reward_var + r_a * np.square(reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    def reset(self):
        obs = self._wrapped_env.reset()
        if self._normalize_obs:
            return self._apply_normalize_obs(obs)
        else:
            return obs

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["_obs_mean"] = self._obs_mean
        d["_obs_var"] = self._obs_var
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_var = d["_obs_var"]

    def step(self, action):
        if isinstance(self._wrapped_env.action_space, Box) or isinstance(self._wrapped_env.action_space, OldBox):
            # rescale the action
            lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
            scaled_action = lb + (action + self._normalization_scale) * (ub - lb) / (2 * self._normalization_scale)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if getattr(self, "_normalize_obs", False):
            next_obs = self._apply_normalize_obs(next_obs)
        if getattr(self, "_normalize_reward", False):
            reward = self._apply_normalize_reward(reward)
        return next_obs, reward * self._scale_reward, done, info