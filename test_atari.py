#!/usr/bin/env python
import argparse

def test(game_name, num_timesteps, load_path, save_path, noops=False, sticky=False, epsgreedy=False):
    import os
    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    print('initialized worker %d' % hvd.rank(), flush=True)
    from atari_reset.ppo import learn
    from atari_reset.policies import make_wavenet_policy
    import gym
    from atari_reset.wrappers import SubprocVecEnv, VideoWriter, MaxAndSkipEnv, GrayscaleDownsample,\
        EpsGreedyEnv, StickyActionEnv, NoopResetEnv
    from baselines import logger
    from baselines.bench import Monitor

    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    nenvs = 16

    def make_env(rank):
        def env_fn():
            env = gym.make(game_name + 'NoFrameskip-v4')
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            # write videos for every run
            dir = os.path.join(save_path, game_name)
            os.makedirs(dir, exist_ok=True)
            dir = os.path.join(dir, str(rank))
            os.makedirs(dir, exist_ok=True)
            videofile_prefix = os.path.join(dir, 'episode')
            env = VideoWriter(env, videofile_prefix, fps=120)
            if sticky:
                env = StickyActionEnv(env)
            env = MaxAndSkipEnv(env, 4)
            if noops:
                env = NoopResetEnv(env)
            if epsgreedy:
                env = EpsGreedyEnv(env)
            env = GrayscaleDownsample(env)
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i + nenvs * hvd.rank()) for i in range(nenvs)])

    policy = make_wavenet_policy(env.action_space.n)

    learn(policy=policy, env=env, nsteps=128, log_interval=1, total_timesteps=num_timesteps,
          load_path=load_path, save_path=save_path, game_name=game_name, test_mode=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e8)
    parser.add_argument('--load_path', type=str, default=None, help='Path to load existing model from')
    parser.add_argument('--save_path', type=str, default='results', help='Where to save results to')
    parser.add_argument("--noops", help="Use 0 to 30 random noops at the start of each episode", action="store_true")
    parser.add_argument("--sticky", help="Use sticky actions", action="store_true")
    parser.add_argument("--epsgreedy", help="Take random action with probability 0.01", action="store_true")
    args = parser.parse_args()

    test(args.game, args.num_timesteps, args.load_path, args.save_path, args.noops, args.sticky, args.epsgreedy)

