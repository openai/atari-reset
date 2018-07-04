#!/usr/bin/env python
import argparse

def train(game_name, num_timesteps, lr, entropy_coef, load_path, starting_point, save_path):
    import os
    import numpy as np
    import tensorflow as tf
    import horovod.tensorflow as hvd
    hvd.init()
    print('initialized worker %d' % hvd.rank(), flush=True)
    from atari_reset.ppo import learn
    from atari_reset.policies import make_wavenet_policy
    import gym
    from atari_reset.wrappers import ReplayResetEnv, ResetManager, SubprocVecEnv, VideoWriter, \
        MaxAndSkipEnv, GrayscaleDownsample, EpisodicLifeEnv

    ncpu = 2
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    tf.Session(config=config).__enter__()

    nrstartsteps = 320  # number of non frameskipped steps to divide workers over
    nenvs = 16
    nrworkers = hvd.size() * nenvs
    workers_per_sp = int(np.ceil(nrworkers / nrstartsteps))

    def make_env(rank):
        def env_fn():
            env = gym.make(game_name + 'NoFrameskip-v4')
            env = ReplayResetEnv(env, demo_file_name='demos/' + game_name + '.demo', seed=rank,
                                 workers_per_sp=workers_per_sp, clip_and_scale=True, gamma=.999)
            if rank%16==0 and hvd.rank()%8==0: # write videos for monitoring progress during training
                dir = os.path.join(save_path, game_name)
                os.makedirs(dir, exist_ok=True)
                videofile_prefix = os.path.join(dir, 'episode')
                env = VideoWriter(env, videofile_prefix, fps=120)
            env = EpisodicLifeEnv(env)
            env = MaxAndSkipEnv(env, 4)
            env = GrayscaleDownsample(env)
            return env
        return env_fn

    env = SubprocVecEnv([make_env(i + nenvs * hvd.rank()) for i in range(nenvs)])
    env = ResetManager(env)

    if starting_point is not None:
        env.set_max_starting_point(starting_point)

    policy = make_wavenet_policy(env.action_space.n)

    learn(policy=policy, env=env, nsteps=128, nlag=60, lam=.95, ngae=20, gamma=.999, noptepochs=2, nminibatches=4, log_interval=1,
          save_interval=100, ent_coef=entropy_coef, lr=lr, cliprange=0.1, total_timesteps=num_timesteps,
          norm_adv=False, subtract_rew_avg=True, load_path=load_path, save_path=save_path, game_name=game_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='MontezumaRevenge')
    parser.add_argument('--num_timesteps', type=int, default=1e12)
    parser.add_argument('--load_path', type=str, default=None, help='Path to load existing model from')
    parser.add_argument('--starting_point', type=int, default=None, help='Demo-step to start training from, if not the last')
    parser.add_argument('--save_path', type=str, default='results', help='Where to save results to')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--entropy_coef', type=float, default=1e-4)
    args = parser.parse_args()

    train(args.game, args.num_timesteps, args.learning_rate, args.entropy_coef,
          args.load_path, args.starting_point, args.save_path)

