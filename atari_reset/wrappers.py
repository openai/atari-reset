import pickle
import gym
from collections import deque
from PIL import Image
from gym import spaces
import imageio
import numpy as np
from multiprocessing import Process, Pipe
import mpi4py.rc
import horovod.tensorflow as hvd
mpi4py.rc.initialize = False
from mpi4py import MPI

reset_for_batch = False

class MyWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MyWrapper, self).__init__(env)
    def decrement_starting_point(self, nr_steps):
        return self.env.decrement_starting_point(nr_steps)
    def recursive_getattr(self, name):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            return self.env.recursive_getattr(name)
    def batch_reset(self):
        global reset_for_batch
        reset_for_batch = True
        obs = self.env.reset()
        reset_for_batch = False
        return obs
    def reset(self):
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

    def step_async(self, actions):
        return self.env.step_async(actions)

    def step_wait(self):
        return self.env.step_wait()

    def reset_task(self):
        return self.env.reset_task()

    @property
    def num_envs(self):
        return self.env.num_envs

class VecFrameStack(MyWrapper):
    """
    Vectorized environment base class
    """
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack
        wos = venv.observation_space # wrapped ob space
        low = np.repeat(wos.low, self.nstack, axis=-1)
        high = np.repeat(wos.high, self.nstack, axis=-1)
        self.stackedobs = np.zeros((venv.num_envs,)+low.shape, low.dtype)
        self._observation_space = spaces.Box(low=low, high=high)
        self._action_space = venv.action_space
    def step(self, vac):
        """
        Apply sequence of actions to sequence of environments
        actions -> (observations, rewards, news)
        where 'news' is a boolean vector indicating whether each element is new.
        """
        obs, rews, news, infos = self.venv.step(vac)
        self.stackedobs = np.roll(self.stackedobs, shift=-1, axis=-1)
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs, rews, news, infos
    def reset(self):
        """
        Reset all environments
        """
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[..., -obs.shape[-1]:] = obs
        return self.stackedobs
    @property
    def action_space(self):
        return self._action_space
    @property
    def observation_space(self):
        return self._observation_space
    def close(self):
        self.venv.close()
    @property
    def num_envs(self):
        return self.venv.num_envs

class ReplayResetEnv(MyWrapper):
    """
        Randomly resets to states from a replay
    """

    def __init__(self, env, demo_file_name, seed, reset_steps_ignored=512, workers_per_sp=4, frac_sample=0.2, game_over_on_life_loss=True):
        super(ReplayResetEnv, self).__init__(env)
        with open(demo_file_name, "rb") as f:
            dat = pickle.load(f)
        self.actions = dat['actions']
        rewards = dat['rewards']
        assert len(rewards) == len(self.actions)
        self.returns = np.cumsum(rewards)
        self.checkpoints = dat['checkpoints']
        self.checkpoint_action_nr = dat['checkpoint_action_nr']
        self.rng = np.random.RandomState(seed)
        self.reset_steps_ignored = reset_steps_ignored
        self.actions_to_overwrite = []
        self.starting_point = len(self.actions) - 1 - seed//workers_per_sp
        self.starting_point_current_ep = None
        self.frac_sample = frac_sample
        self.game_over_on_life_loss = game_over_on_life_loss

    def step(self, action):
        if len(self.actions_to_overwrite) > 0:
            action = self.actions_to_overwrite.pop(0)
            valid = False
        else:
            valid = True
        prev_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, info = self.env.step(action)
        self.action_nr += 1
        self.score += reward

        # game over on loss of life, to speed up learning
        if self.game_over_on_life_loss:
            lives = self.env.unwrapped.ale.lives()
            if lives < prev_lives and lives > 0:
                done = True

        # kill if we have achieved the final score, or if we're laggging the demo too much
        if self.score >= self.returns[-1]:
            self.extra_frames_counter -= 1
            if self.extra_frames_counter <= 0:
                done = True
                info['replay_reset.random_reset'] = True # to distinguish from actual game over
        elif self.action_nr>50 and self.score<self.returns[np.minimum(len(self.returns)-1,self.action_nr-50)]:
            done = True

        # output flag to increase entropy if near the starting point of this episode
        if self.action_nr < self.starting_point + 100:
            info['increase_entropy'] = True

        if done:
            ep_info = {'l':self.action_nr, 'as_good_as_demo':(self.score >= self.returns[-1]),
                       'r':self.score, 'starting_point': self.starting_point_current_ep}
            info['episode'] = ep_info

        if not valid:
            info['replay_reset.invalid_transition'] = True

        return obs, reward, done, info

    def decrement_starting_point(self, nr_steps):
        if self.starting_point>0:
            self.starting_point = int(np.maximum(self.starting_point - nr_steps, 0))

    def reset(self):
        obs = self.env.reset()
        self.extra_frames_counter = int(np.exp(self.rng.rand()*7))

        if reset_for_batch:
            self.starting_point_current_ep = 0
            self.actions_to_overwrite = self.actions[:]
            self.action_nr = 0
            self.score = self.returns[0]
        else:

            if self.rng.rand() <= 1.-self.frac_sample:
                self.starting_point_current_ep = self.starting_point
            else:
                self.starting_point_current_ep = self.rng.randint(low=self.starting_point, high=len(self.actions))

            start_action_nr = 0
            start_ckpt = None
            for nr, ckpt in zip(self.checkpoint_action_nr[::-1], self.checkpoints[::-1]):
                if nr <= (self.starting_point_current_ep - self.reset_steps_ignored):
                    start_action_nr = nr
                    start_ckpt = ckpt
                    break
            if start_action_nr > 0:
                self.env.unwrapped.restore_state(start_ckpt)
            nr_to_start_lstm = np.maximum(self.starting_point_current_ep - self.reset_steps_ignored, start_action_nr)
            if nr_to_start_lstm>start_action_nr:
                for a in self.actions[start_action_nr:nr_to_start_lstm]:
                    action = self.env.unwrapped._action_set[a]
                    self.env.unwrapped.ale.act(action)
            self.actions_to_overwrite = self.actions[nr_to_start_lstm:self.starting_point_current_ep]
            if nr_to_start_lstm>0:
                obs = self.env.unwrapped._get_image()
            self.action_nr = nr_to_start_lstm
            self.score = self.returns[nr_to_start_lstm]

        return obs

class MaxAndSkipEnv(MyWrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        MyWrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        combined_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            combined_info.update(info)
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, combined_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(MyWrapper):
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.sign(reward)
        return obs, reward, done, info

class EpsGreedyEnv(MyWrapper):
    def __init__(self, env, eps=0.01):
        MyWrapper.__init__(self, env)
        self.eps = eps

    def step(self, action):
        if np.random.uniform()<self.eps:
            action = np.random.randint(self.env.action_space.n)
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class StickyActionEnv(MyWrapper):
    def __init__(self, env, p=0.25):
        MyWrapper.__init__(self, env)
        self.p = p
        self.last_action = 0

    def step(self, action):
        if np.random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class Box(gym.Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    Example usage:
    self.action_space = spaces.Box(low=-10, high=10, shape=(1,))
    """
    def __init__(self, low, high, shape=None, dtype=np.uint8):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are arrays of the same shape
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)
        self.dtype = dtype
    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (x <= self.high).all()
    def to_jsonable(self, sample_n):
        return np.array(sample_n).tolist()
    def from_jsonable(self, sample_n):
        return [np.asarray(sample) for sample in sample_n]
    @property
    def shape(self):
        return self.low.shape
    @property
    def size(self):
        return self.low.shape
    def __repr__(self):
        return "Box" + str(self.shape)
    def __eq__(self, other):
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

class WarpFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        MyWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = Box(low=0, high=255, shape=(self.res, self.res, 1), dtype = np.uint8)

    def reshape_obs(self, obs):
        obs = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        obs = np.array(Image.fromarray(obs).resize((self.res, self.res),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape((self.res, self.res, 1))

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info

class MyResizeFrame(MyWrapper):
    def __init__(self, env):
        """Warp frames to 105x80"""
        MyWrapper.__init__(self, env)
        self.res = (105, 80, 3)
        self.observation_space = Box(low=0, high=255, shape=self.res, dtype = np.uint8)

    def reshape_obs(self, obs):
        obs = np.array(Image.fromarray(obs).resize((self.res[0],self.res[1]),
                                                   resample=Image.BILINEAR), dtype=np.uint8)
        return obs.reshape(self.res)

    def reset(self):
        return self.reshape_obs(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.reshape_obs(obs), reward, done, info

class FireResetEnv(MyWrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        MyWrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class VideoWriter(MyWrapper):
    def __init__(self, env, file_prefix):
        MyWrapper.__init__(self, env)
        self.file_prefix = file_prefix
        self.video_writer = None
        self.counter = 0

    def process_frame(self, frame):
        f_out = np.zeros((224, 160, 3), dtype=np.uint8)
        f_out[7:-7, :] = np.cast[np.uint8](frame)
        return f_out

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.video_writer.append_data(self.process_frame(obs))
        return obs, reward, done, info

    def reset(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.counter += 1
        self.video_writer = imageio.get_writer(self.file_prefix + str(self.counter) + '.mp4', mode='I', fps=120)
        return self.env.reset()

def my_wrapper(env, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id
    if clip_rewards:
        env = ClipRewardEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if 'Pong' in env.spec.id:
        env = FireResetEnv(env)
    env = MyResizeFrame(env)
    return env

class ResetManager(MyWrapper):
    def __init__(self, env):
        super(ResetManager, self).__init__(env)
        starting_points = self.env.recursive_getattr('starting_point')
        all_starting_points = flatten_lists(MPI.COMM_WORLD.allgather(starting_points))
        self.min_starting_point = min(all_starting_points)
        self.max_starting_point = max(all_starting_points)
        self.nrstartsteps = self.max_starting_point - self.min_starting_point
        assert(self.nrstartsteps > 10)
        self.max_max_starting_point = self.max_starting_point
        self.starting_point_success = np.zeros(self.max_starting_point+10000)
        self.counter = 0
        self.infos = []

    def proc_infos(self):
        epinfos = [info['episode'] for info in self.infos if 'episode' in info]

        if hvd.size()>1:
            epinfos = flatten_lists(MPI.COMM_WORLD.allgather(epinfos))

        new_sp_wins = {}
        new_sp_counts = {}
        for epinfo in epinfos:
            sp = epinfo['starting_point']
            if sp in new_sp_counts:
                new_sp_counts[sp] += 1
                if epinfo['as_good_as_demo']:
                    new_sp_wins[sp] += 1
            else:
                new_sp_counts[sp] = 1
                if epinfo['as_good_as_demo']:
                    new_sp_wins[sp] = 1
                else:
                    new_sp_wins[sp] = 0

        for sp,wins in new_sp_wins.items():
            self.starting_point_success[sp] = np.cast[np.float32](wins)/new_sp_counts[sp]

        # move starting point, ensuring at least 20% of workers are able to complete the demo
        csd = np.argwhere(np.cumsum(self.starting_point_success) / self.nrstartsteps >= 0.2)
        if len(csd) > 0:
            new_max_start = csd[0][0]
        else:
            new_max_start = np.minimum(self.max_starting_point + 100, self.max_max_starting_point)
        n_points_to_shift = self.max_starting_point - new_max_start
        self.decrement_starting_point(n_points_to_shift)
        self.infos = []

    def decrement_starting_point(self, n_points_to_shift):
        self.env.decrement_starting_point(n_points_to_shift)
        starting_points = self.env.recursive_getattr('starting_point')
        all_starting_points = flatten_lists(MPI.COMM_WORLD.allgather(starting_points))
        self.max_starting_point = max(all_starting_points)

    def set_max_starting_point(self, starting_point):
        n_points_to_shift = self.max_starting_point - starting_point
        self.decrement_starting_point(n_points_to_shift)

    def step(self, action):
        obs, rews, news, infos = self.env.step(action)
        self.infos += infos
        self.counter += 1
        if self.counter > (self.max_max_starting_point - self.max_starting_point) / 2 and self.counter % 1024 == 0:
            self.proc_infos()
        return obs, rews, news, infos

    def step_wait(self):
        obs, rews, news, infos = self.env.step_wait()
        self.infos += infos
        self.counter += 1
        if self.counter > (self.max_max_starting_point - self.max_starting_point) / 2 and self.counter % 1024 == 0:
            self.proc_infos()
        return obs, rews, news, infos

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.action_space, env.observation_space))
        elif cmd == 'get_history':
            senv = env
            while not hasattr(senv, 'get_history'):
                senv = senv.env
            remote.send(senv.get_history(data))
        elif cmd == 'recursive_getattr':
            remote.send(env.recursive_getattr(data))
        elif cmd == 'decrement_starting_point':
            env.decrement_starting_point(data)
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class SubprocVecEnv(MyWrapper):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)

        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def get_history(self, nsteps):
        for remote in self.remotes:
            remote.send(('get_history', nsteps))
        results = [remote.recv() for remote in self.remotes]
        obs, acts, dones = zip(*results)
        obs = np.stack(obs)
        acts = np.stack(acts)
        dones = np.stack(dones)
        return obs, acts, dones

    def recursive_getattr(self, name):
        for remote in self.remotes:
            remote.send(('recursive_getattr',name))
        return [remote.recv() for remote in self.remotes]

    def decrement_starting_point(self, n):
        for remote in self.remotes:
            remote.send(('decrement_starting_point', n))

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    @property
    def num_envs(self):
        return len(self.remotes)

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

