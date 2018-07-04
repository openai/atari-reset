'''
Proximal policy optimization with a few tricks. Adapted from the implementation in baselines.
'''

import os
import sys
import time
import joblib
import numpy as np
import os.path as osp
from collections import deque
import tensorflow as tf
import horovod.tensorflow as hvd
import mpi4py.rc
mpi4py.rc.initialize = False
from mpi4py import MPI
from baselines.common.mpi_moments import mpi_moments, mpi_mean
from baselines.common import explained_variance
from baselines import logger

class Model(object):
    def __init__(self, policy, ob_space, ac_space, nenv_step, nenv_train, nsteps, nlag, ent_coef, vf_coef,
                 cliprange, adam_epsilon=1e-12, load_path=None, wdecay=0., test_mode=False):
        sess = tf.get_default_session()

        A = tf.placeholder(tf.int32, [nsteps, nenv_train], name='action')
        ADV = tf.placeholder(tf.float32, [nsteps, nenv_train], name='advantage')
        VALID = tf.placeholder(tf.float32, [nsteps, nenv_train], name='valid')
        R = tf.placeholder(tf.float32, [nsteps, nenv_train], name='return')
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [nsteps, nenv_train], name='neglogprob')
        OLDVPRED = tf.placeholder(tf.float32, [nsteps, nenv_train], name='valuepred')
        DONES_train = tf.placeholder(tf.int32, [nsteps+nlag, nenv_train], name='dones_train')
        LR = tf.placeholder(tf.float32, [], name='lr')
        X_train = tf.placeholder(tf.int16, (nsteps+nlag,nenv_train)+ob_space.shape[-3:], name='X_train')
        E = tf.placeholder(tf.int32, [nsteps, nenv_train], name='increase_ent') # for states that need more exploration

        out_train = policy(tf.to_float(X_train)/255., dones=DONES_train, nenv=nenv_train)[-nsteps:]

        logits = out_train[:,:,:ac_space.n]
        logits -= tf.expand_dims(tf.to_float(E > 0), -1) * 0.5 * logits
        probs = tf.nn.softmax(logits)
        vpred = out_train[:,:,-1]

        mv = tf.reduce_mean(VALID) + 1e-6
        entropy = tf.reduce_mean(VALID * tf.nn.softmax_cross_entropy_with_logits_v2(labels=probs, logits=logits)) / mv

        vf_losses1 = tf.square(vpred - R)
        vf_loss = .5 * tf.reduce_mean(VALID * vf_losses1) / mv

        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=A, logits=logits)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - cliprange, 1.0 + cliprange)
        pg_loss = tf.reduce_mean(VALID * tf.maximum(pg_losses, pg_losses2)) / mv

        approxkl = .5 * tf.reduce_mean(VALID * tf.square(neglogpac - OLDNEGLOGPAC)) / mv
        clipfrac = tf.reduce_mean(VALID * tf.to_float(tf.greater(tf.abs(ratio - 1.0), cliprange))) / mv

        params = tf.trainable_variables()
        l2_loss = .5 * sum([tf.reduce_sum(tf.square(p)) for p in params])
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        opt = tf.train.AdamOptimizer(LR, epsilon=adam_epsilon)
        opt = hvd.DistributedOptimizer(opt)
        train_op = opt.minimize(loss)

        if wdecay > 0:
            wdecay_op = tf.group(*[p.assign((1.-wdecay)*p) for p in params])

        def train(lr, obs, ents, dones, returns, advs, actions, values, neglogpacs, valids):
            td_map = {LR: lr, X_train: obs, E: ents, A: actions, ADV: advs, VALID: valids, R: returns,
                      OLDNEGLOGPAC: neglogpacs, OLDVPRED: values, DONES_train: dones}
            return sess.run([pg_loss, vf_loss, l2_loss, entropy, approxkl, clipfrac, train_op], feed_dict=td_map)[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'l2_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        X_step = tf.placeholder(tf.int16, (nenv_step,) + ob_space.shape[-3:], name='X_step')
        E_step = tf.placeholder(tf.int32, (nenv_step,), name='E_step')
        DONES_step = tf.placeholder(tf.int32, [nenv_step], name='dones_step')
        out_test = tf.squeeze(policy(tf.to_float(X_step)/255., dones=DONES_step, nenv=nenv_step))
        logits = out_test[:, :ac_space.n]
        if test_mode:
            logits *= 1.5
        else:
            logits -= tf.expand_dims(tf.to_float(E_step > 0), -1) * 0.5 * logits
        u = tf.random_uniform([nenv_step, ac_space.n], minval=0., maxval=1, dtype=tf.float32, seed=hvd.rank())
        actions = tf.argmax(logits - tf.log(-tf.log(u)), axis=-1)
        neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        vpred = out_test[:,-1]

        def step(obs, ents, dones):
            fd = {X_step: obs, E_step: ents, DONES_step: dones}
            return sess.run([actions,neglogpac,vpred], feed_dict=fd)

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p,lp in zip(params, loaded_params):
                restores.append(p.assign(lp))
            sess.run(restores)

        def weight_decay():
            if wdecay > 0:
                sess.run(wdecay_op)

        self.train = train
        self.step = step
        self.save = save
        self.load = load
        self.weight_decay = weight_decay
        sess.run(tf.global_variables_initializer())
        if load_path and hvd.rank()==0:
            self.load(load_path)
        sess.run(hvd.broadcast_global_variables(0))
        tf.get_default_graph().finalize()


class Runner(object):

    def __init__(self, env, model, nsteps, gamma, lam, norm_adv, subtract_rew_avg, num_steps_to_cut_left, num_steps_to_cut_right):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.gamma = gamma
        self.lam = lam
        self.norm_adv = norm_adv
        self.subtract_rew_avg = subtract_rew_avg
        self.nsteps = nsteps
        self.num_steps_to_cut_left = num_steps_to_cut_left
        self.num_steps_to_cut_right = num_steps_to_cut_right
        obs = [env.reset()]
        dones = [np.array([False for _ in range(nenv)])]
        random_res = [np.array([False for _ in range(nenv)])]
        increase_ent = [np.array([False for _ in range(nenv)])]
        self.data = {'obs':obs, 'increase_ent':increase_ent, 'rewards':[], 'actions':[],
                     'values':[], 'valids':[], 'random_resets':random_res, 'dones':dones, 'neglogpacs':[]}

    def run(self):
        # shift forward
        if len(self.data['rewards']) >= self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            for k,v in self.data.items():
                self.data[k] = v[self.nsteps:]

        epinfos = []
        while len(self.data['rewards']) < self.nsteps+self.num_steps_to_cut_left+self.num_steps_to_cut_right:
            actions, neglogpacs, values = self.model.step(self.data['obs'][-1], self.data['increase_ent'][-1], self.data['dones'][-1])
            self.data['actions'].append(actions)
            self.data['neglogpacs'].append(neglogpacs)
            self.data['values'].append(values)

            obs, rewards, dones, infos = self.env.step(actions)
            self.data['obs'].append(obs)
            self.data['increase_ent'].append([info.get('replay_reset.increase_entropy', False) for info in infos])
            self.data['rewards'].append(rewards)
            self.data['dones'].append(dones)
            self.data['valids'].append([(not info.get('replay_reset.invalid_transition', False)) for info in infos])
            self.data['random_resets'].append(np.array([info.get('replay_reset.random_reset', False) for info in infos]))

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)

        # generalized advantage estimation
        mb_advs = [np.zeros_like(self.data['values'][0])] * len(self.data['rewards'])
        for t in reversed(range(len(self.data['rewards'])-1)):
            use_next = 1. - np.array(self.data['dones'][t+1])
            adv_mask = 1. - np.array(self.data['random_resets'][t+1])
            delta = self.data['rewards'][t] + self.gamma * use_next * self.data['values'][t+1] - self.data['values'][t]
            mb_advs[t] = adv_mask * (delta + self.gamma * self.lam * use_next * mb_advs[t + 1])

        # extract arrays
        end = self.nsteps + self.num_steps_to_cut_left
        ar_mb_obs = np.stack(self.data['obs'][:end],axis=0)
        ar_mb_dones = np.stack(self.data['dones'][:end], axis=0)
        ar_mb_valids = np.stack(self.data['valids'][self.num_steps_to_cut_left:end],axis=0)
        ar_mb_actions = np.stack(self.data['actions'][self.num_steps_to_cut_left:end],axis=0)
        ar_mb_entropy = np.stack(self.data['increase_ent'][self.num_steps_to_cut_left:end], axis=0)
        ar_mb_values = np.stack(self.data['values'][self.num_steps_to_cut_left:end],axis=0)
        ar_mb_neglogpacs = np.stack(self.data['neglogpacs'][self.num_steps_to_cut_left:end],axis=0)
        ar_mb_advs = np.stack(mb_advs[self.num_steps_to_cut_left:end], axis=0)
        ar_mb_rets = ar_mb_values + ar_mb_advs

        if self.norm_adv:
            adv_mean, adv_std, _ = mpi_moments(ar_mb_advs.ravel())
            ar_mb_advs = (ar_mb_advs - adv_mean) / (adv_std + 1e-7)
        elif self.subtract_rew_avg:
            adv_mean, _ = mpi_mean(ar_mb_advs.ravel())
            ar_mb_advs -= adv_mean

        mb_data = {'obs':ar_mb_obs, 'ents':ar_mb_entropy, 'dones':ar_mb_dones, 'advs':ar_mb_advs,
                   'actions':ar_mb_actions, 'values':ar_mb_values, 'neglogpacs':ar_mb_neglogpacs,
                   'valids':ar_mb_valids, 'returns':ar_mb_rets}
        return mb_data, epinfos

def learn(policy, env, nsteps, nlag, total_timesteps, ent_coef, lr,
            vf_coef=0.5, gamma=0.99, lam=0.95, ngae=60, log_interval=10,
            noptepochs=4, nminibatches=4, cliprange=0.2, save_interval=0,
            norm_adv=False, subtract_rew_avg=True, load_path=None, save_path='', game_name='', wdecay=0.,
            test_mode=False):
    nenvs = env.num_envs
    envs_per_batch = nenvs//nminibatches
    ob_space = env.observation_space
    ac_space = env.action_space
    nbatch = nenvs * nsteps

    model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenv_step=nenvs, nenv_train=envs_per_batch,
                  nsteps=nsteps, nlag=nlag, ent_coef=ent_coef, vf_coef=vf_coef, cliprange=cliprange,
                  load_path=load_path, wdecay=wdecay, test_mode=test_mode)
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, norm_adv=norm_adv,
                    subtract_rew_avg=subtract_rew_avg, num_steps_to_cut_left=nlag, num_steps_to_cut_right=ngae)

    tfirststart = time.time()
    nupdates = total_timesteps // (nbatch*hvd.size())
    update = 0
    epinfobuf = deque(maxlen=100)
    dir = osp.join(save_path, game_name)
    os.makedirs(dir, exist_ok=True)
    lgr = logger.Logger(dir, output_formats=[logger.HumanOutputFormat(sys.stdout), logger.CSVOutputFormat(osp.join(dir, 'results.csv'))])
    rng = np.random.RandomState(hvd.rank())
    while update < nupdates:
        tstart = time.time()
        update += 1

        mb_data, epinfos = runner.run()
        time_stepping = time.time() - tstart

        time_before_train = time.time()
        if not test_mode:
            mblossvals = []
            for _ in range(noptepochs):
                for p in np.split(rng.permutation(nenvs), nminibatches):
                    mblossvals.append(model.train(lr, **{k: v[:,p] for k,v in mb_data.items()}))
            model.weight_decay()
        time_training = time.time() - time_before_train

        epinfos = MPI.COMM_WORLD.gather(epinfos, root=0)
        if hvd.rank() == 0:
            epinfos = flatten_lists(epinfos)
            lossvals = np.mean(mblossvals, axis=0)
            tnow = time.time()
            tps = int(nbatch*hvd.size() / (tnow - tstart))
            if update % log_interval == 0 or update == 1:
                epinfobuf.extend(epinfos)
                if len(epinfos) >= 100:
                    epinfos_to_report = epinfos
                else:
                    epinfos_to_report = epinfobuf
                ev = explained_variance(mb_data['values'].reshape([-1]), mb_data['returns'].reshape([-1]))
                lgr.logkv("serial_timesteps", update*nsteps)
                lgr.logkv("nupdates", update)
                lgr.logkv("total_timesteps", update*nbatch*hvd.size())
                lgr.logkv("tps", tps)
                lgr.logkv("explained_variance", float(ev))
                lgr.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfos_to_report]))
                lgr.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfos_to_report]))
                lgr.logkv('time_elapsed', tnow - tfirststart)
                lgr.logkv('time_stepping', time_stepping)
                lgr.logkv('time_training', time_training)
                lgr.logkv('perc_valid', np.mean(mb_data['valids']))
                lgr.logkv('tcount', update*nbatch*hvd.size())
                lgr.logkv('mean_return', np.mean(mb_data['returns']))
                lgr.logkv('max_return', np.amax(mb_data['returns']))
                lgr.logkv('min_return', np.amin(mb_data['returns']))
                if not test_mode:
                    for (lossval, lossname) in zip(lossvals, model.loss_names):
                        lgr.logkv(lossname, lossval)
                    if hasattr(env, 'max_starting_point'):
                        lgr.logkv('max_starting_point', env.max_starting_point)
                        lgr.logkv('as_good_as_demo_start', safemean(
                            [epinfo['as_good_as_demo'] for epinfo in epinfos_to_report if
                             epinfo['starting_point'] <= env.max_starting_point]))
                        lgr.logkv('as_good_as_demo_all', safemean(
                            [epinfo['as_good_as_demo'] for epinfo in epinfos_to_report]))
                        lgr.logkv('perc_started_below_max_sp', safemean(
                            [epinfo['starting_point'] <= env.max_starting_point for epinfo in epinfos_to_report]))
                lgr.dumpkvs()
            if save_interval and (update % save_interval == 0 or update == 1) and not test_mode:
                savepath = osp.join(dir, '%.6i' % update)
                print('Saving to', savepath)
                model.save(savepath)

    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

