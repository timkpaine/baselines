import numpy as np
from future.utils import with_metaclass
from abc import ABCMeta, abstractmethod
from baselines.common import raise_if_none


class AbstractEnvRunner(with_metaclass(ABCMeta)):
    def __init__(self, _named_only=object(), env=None, model=None, nsteps=None):
        raise_if_none(env=env, model=model, nsteps=nsteps)
        self.env = env
        self.model = model
        self.nenv = nenv = env.num_envs if hasattr(env, 'num_envs') else 1
        self.batch_ob_shape = (nenv*nsteps,) + env.observation_space.shape
        self.obs = np.zeros((nenv,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    @abstractmethod
    def run(self):
        raise NotImplementedError

