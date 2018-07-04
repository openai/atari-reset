'''
Proximal policy optimization with a few tricks. Adapted from the implementation in baselines.
'''

import os.path as osp
import time
import joblib
import numpy as np
from baselines import logger
from collections import deque
import tensorflow as tf
import horovod.tensorflow as hvd
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from baselines.common import explained_variance
from baselines.common.mpi_moments import mpi_moments

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenv, nsteps, ent_coef, vf_coef, l2_coef,
                 cliprange, adam_epsilon=1e-6, load_path=None, test_mode=False):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_space, ac_space, nenv, 1, test_mode=test_mode, reuse=False)
        train_model = policy(sess, ob_space, ac_space, nenv, nsteps, test_mode=test_mode, reuse=True)

        A = train_model.pdtype.sample_placeholder([nenv*nsteps], name='action')
        ADV = tf.placeholder(tf.float32, [nenv*nsteps], name='advantage')
        VALID = tf.placeholder(tf.float32, [nenv*nsteps], name='valid')
        R = tf.placeholder(tf.float32, [nenv*nsteps], name='return')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [nenv*nsteps], name='neglogprob')
        OLDVPRED = tf.placeholder(tf.float32, [nenv*nsteps], name='valuepred')
        LR = tf.placeholder(tf.float32, [], name='lr')

        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(VALID * train_model.pd.entropy())
        vpred = train_model.vf

        vpredclipped = OLDVPRED + tf.clip_by_value(vpred - OLDVPRED, - cliprange, cliprange)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(VALID * tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = tf.reduce_mean(VALID * tf.maximum(pg_losses, pg_losses2))
        mv = tf.reduce_mean(VALID)
        approxkl = .5 * tf.reduce_mean(VALID * tf.square(neglogpac - OLDNEGLOGPAC)) / mv
        clipfrac = tf.reduce_mean(VALID * tf.to_float(tf.greater(tf.abs(ratio - 1.0), cliprange))) / mv
        params = tf.trainable_variables()
        l2_loss = .5 * sum([tf.reduce_sum(tf.square(p)) for p in params])
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef + l2_coef*l2_loss

        opt = tf.train.AdamOptimizer(LR, epsilon=adam_epsilon)
        opt = hvd.DistributedOptimizer(opt)
        train_op = opt.minimize(loss)

        def train(lr, obs, returns, advs, masks, actions, values, neglogpacs, valids, increase_ent, states=None):
            td_map = {LR: lr, train_model.X: obs, A: actions, ADV: advs, VALID: valids, R: returns,
                      OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, train_model.E: increase_ent}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run([pg_loss, vf_loss, l2_loss, entropy, approxkl, clipfrac, train_op], feed_dict=td_map)[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'l2_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        sess.run(tf.global_variables_initializer())
        if load_path and hvd.rank()==0:
            self.load(load_path)
        sess.run(hvd.broadcast_global_variables(0))
        tf.get_default_graph().finalize()


class Runner(object):

    def __init__(self, env, model, nsteps, gamma, lam, norm_adv, subtract_rew_avg):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.gamma = gamma
        self.lam = lam
        self.norm_adv = norm_adv
        self.subtract_rew_avg = subtract_rew_avg
        self.nsteps = nsteps
        self.num_steps_to_cut_left = nsteps//2
        self.num_steps_to_cut_right = 0
        obs = [np.cast[model.train_model.X.dtype.name](env.reset())]
        states = [model.initial_state]
        dones = [np.array([False for _ in range(nenv)])]
        random_res = [np.array([False for _ in range(nenv)])]
        # mb_obs, mb_increase_ent, mb_rewards, mb_reward_avg, mb_actions, mb_values, mb_valids, mb_random_resets, mb_dones, mb_neglogpacs, mb_states
        self.mb_stuff = [obs, [np.zeros(obs[0].shape[0], dtype=np.uint8)], [], [], [], [], [], [random_res], dones, [], states]

    def run(self):
        # shift forward
        if len(self.mb_stuff[2]) >= self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            self.mb_stuff = [l[self.nsteps:] for l in self.mb_stuff]

        mb_obs, mb_increase_ent, mb_rewards, mb_reward_avg, mb_actions, mb_values, mb_valids, mb_random_resets, \
            mb_dones, mb_neglogpacs, mb_states = self.mb_stuff
        epinfos = []
        while len(mb_rewards) < self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            actions, values, states, neglogpacs = self.model.step(mb_obs[-1], mb_states[-1], mb_dones[-1], mb_increase_ent[-1])
            mb_actions.append(actions)
            mb_values.append(values)
            mb_states.append(states)
            mb_neglogpacs.append(neglogpacs)

            obs, rewards, dones, infos = self.env.step(actions)
            mb_obs.append(np.cast[self.model.train_model.X.dtype.name](obs))
            mb_increase_ent.append(np.asarray([info.get('increase_entropy', False) for info in infos], dtype=np.uint8))
            mb_rewards.append(rewards)
            mb_dones.append(dones)
            mb_valids.append([(not info.get('replay_reset.invalid_transition', False)) for info in infos])
            mb_random_resets.append(np.array([info.get('replay_reset.random_reset', False) for info in infos]))

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

        # GAE
        mb_advs = [np.zeros_like(mb_values[0])] * (len(mb_rewards) + 1)
        for t in reversed(range(len(mb_rewards))):
            if t < self.num_steps_to_cut_left:
                mb_valids[t] = np.zeros_like(mb_valids[t])
            else:
                if t == len(mb_values)-1:
                    next_value = self.model.value(mb_obs[-1], mb_states[-1], mb_dones[-1])
                else:
                    next_value = mb_values[t+1]
                use_next = np.logical_not(mb_dones[t+1])
                adv_mask = np.logical_not(mb_random_resets[t+1])
                delta = mb_rewards[t] + self.gamma * use_next * next_value - mb_values[t]
                mb_advs[t] = adv_mask * (delta + self.gamma * self.lam * use_next * mb_advs[t + 1])

        # extract arrays
        end = self.nsteps + self.num_steps_to_cut_left
        ar_mb_obs = np.asarray(mb_obs[:end], dtype=self.model.train_model.X.dtype.name)
        ar_mb_ent = np.stack(mb_increase_ent[:end], axis=0)
        ar_mb_valids = np.asarray(mb_valids[:end], dtype=np.float32)
        ar_mb_actions = np.asarray(mb_actions[:end])
        ar_mb_values = np.asarray(mb_values[:end], dtype=np.float32)
        ar_mb_neglogpacs = np.asarray(mb_neglogpacs[:end], dtype=np.float32)
        ar_mb_dones = np.asarray(mb_dones[:end], dtype=np.bool)
        ar_mb_advs = np.asarray(mb_advs[:end], dtype=np.float32)
        ar_mb_rets = ar_mb_values + ar_mb_advs

        if self.norm_adv:
            adv_mean, adv_std, _ = mpi_moments(ar_mb_advs.ravel())
            ar_mb_advs = (ar_mb_advs - adv_mean) / (adv_std + 1e-7)

        # obs, increase_ent, advantages, masks, actions, values, neglogpacs, valids, returns, states, epinfos = runner.run()
        return (*map(sf01, (ar_mb_obs, ar_mb_ent, ar_mb_advs, ar_mb_dones, ar_mb_actions, ar_mb_values, ar_mb_neglogpacs, ar_mb_valids, ar_mb_rets)),
            mb_states[0], epinfos)

def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


def learn(policy, env, nsteps, total_timesteps, ent_coef=1e-4, lr=1e-4,
            vf_coef=0.5, l2_coef=1e-5, gamma=0.99, lam=0.95, log_interval=10,
            noptepochs=4, cliprange=0.2, save_interval=0, norm_adv=True, subtract_rew_avg=False,
            load_path=None, save_path='results', game_name='', test_mode=False):
    nenvs = env.num_envs
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nsteps_train = nsteps + nsteps // 2

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenv=nenvs, nsteps=nsteps_train, ent_coef=ent_coef,
                  vf_coef=vf_coef, l2_coef=l2_coef, cliprange=cliprange, load_path=load_path, test_mode=test_mode)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam,
                    norm_adv=norm_adv, subtract_rew_avg=subtract_rew_avg)

    tfirststart = time.time()
    nupdates = total_timesteps // (nbatch*hvd.size())
    update = 0
    epinfobuf = deque(maxlen=100)
    while update < nupdates:
        tstart = time.time()
        update += 1

        obs, increase_ent, advantages, masks, actions, values, neglogpacs, valids, returns, states, epinfos = runner.run()

        if hvd.size()>1:
            epinfos = flatten_lists(MPI.COMM_WORLD.allgather(epinfos))

        if not test_mode:
            mblossvals = []
            for _ in range(noptepochs):
                mblossvals.append(model.train(lr, obs, returns, advantages, masks, actions, values, neglogpacs, valids, increase_ent, states))

        if hvd.rank() == 0:
            tnow = time.time()
            tps = int(nbatch*hvd.size() / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                epinfobuf.extend(epinfos)
                if len(epinfos) >= 100:
                    epinfos_to_report = epinfos
                else:
                    epinfos_to_report = epinfobuf
                ev = explained_variance(values, returns)
                logger.logkv("serial_timesteps", update*nsteps)
                logger.logkv("nupdates", update)
                logger.logkv("total_timesteps", update*nbatch*hvd.size())
                logger.logkv("tps", tps)
                logger.logkv("explained_variance", float(ev))
                logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfos_to_report]))
                logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfos_to_report]))
                if not test_mode:
                    lossvals = np.mean(mblossvals, axis=0)
                    for (lossval, lossname) in zip(lossvals, model.loss_names):
                        logger.logkv(lossname, lossval)
                    if hasattr(env, 'max_starting_point'):
                        logger.logkv('max_starting_point', env.max_starting_point)
                        logger.logkv('as_good_as_demo_start', safemean(
                            [epinfo['as_good_as_demo'] for epinfo in epinfos_to_report if
                             epinfo['starting_point'] <= env.max_starting_point]))
                        logger.logkv('as_good_as_demo_all', safemean(
                            [epinfo['as_good_as_demo'] for epinfo in epinfos_to_report]))
                        logger.logkv('perc_started_below_max_sp', safemean(
                            [epinfo['starting_point'] <= env.max_starting_point for epinfo in epinfos_to_report]))

                logger.logkv('time_elapsed', tnow - tfirststart)
                logger.logkv('perc_valid', np.mean(valids))
                logger.logkv('tcount', update*nbatch*hvd.size())
                logger.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) and not test_mode:
                savepath = osp.join(osp.join(save_path, game_name), '%.6i'%update)
                print('Saving to', savepath)
                model.save(savepath)

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

