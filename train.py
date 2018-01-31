from headers import *
import common
import utils

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def create_scheduler(type='medium'):
    if type == 'none':
        return utils.ConstantSchedule(1.0)
    if type == 'linear':
        return utils.LinearSchedule(10000, 1.0, 0.0)
    if type == 'medium':
        endpoints = [(0, 0), (2000, 0.1), (5000, 0.25), (10000, 0.5), (20000, 1.0)]
    elif type == 'high':
        endpoints = [(0, 0), (3000, 0.1), (8000, 0.25), (15000, 0.5), (30000, 1.0)]
    elif type == 'low': # low
        endpoints = [(0, 0), (1000, 0.1), (2000, 0.25), (7000, 0.5), (15000, 1.0)]
    elif type == 'exp':
        endpoints = [(0, 0), (1000, 0.01), (5000, 0.1), (10000, 0.5), (13000, 0.75), (17000, 0.9), (22000, 0.95), (30000, 1.0)]
    print('Building PiecewiseScheduler with <endpoints> = {}'.format(endpoints))
    scheduler = utils.PiecewiseSchedule(endpoints, outside_value=1.0)
    return scheduler


def train(args=None,
          houseID=0, reward_type='indicator', success_measure='center',
          multi_target=False,
          algo='pg', model_name='cnn',  # NOTE: optional: model_name='rnn'
          iters=2000000, report_rate=20, save_rate=1000, eval_range=200,
          log_dir='./temp', save_dir='./_model_', warmstart=None,
          log_debug_info=True):

    if 'scheduler' in args:
        scheduler = args['scheduler']
    else:
        scheduler = None

    if args is None:
        args = common.create_default_args(algo)

    hardness = args['hardness']
    if hardness is not None:
        print('>>> Hardness Level = {}'.format(hardness))

    trainer = common.create_trainer(algo, model_name, args)
    env = common.create_env(houseID, reward_type, hardness,
                            success_measure=success_measure,
                            segment_input=args['segment_input'],
                            depth_input=args['depth_input'],
                            render_device=args['render_gpu'],
                            cacheAllTarget=args['multi_target'],
                            use_discrete_action=('dpg' not in algo))
    logger = utils.MyLogger(log_dir, True)
    if multi_target:
        assert hasattr(trainer, 'set_target')

    if warmstart is not None:
        if os.path.exists(warmstart):
            logger.print('Warmstarting from <{}> ...'.format(warmstart))
            trainer.load(warmstart)
        else:
            logger.print('Warmstarting from save_dir <{}> with version <{}> ...'.format(save_dir, warmstart))
            trainer.load(save_dir, warmstart)

    logger.print('Start Training')

    if log_debug_info:
        common.debugger = utils.MyLogger(log_dir, True, 'full_logs.txt')
    else:
        common.debugger = utils.FakeLogger()

    episode_rewards = [0.0]
    episode_success = [0.0]
    episode_length = [0.0]
    episode_targets = ['kitchen']

    trainer.reset_agent()
    if multi_target:
        obs = env.reset()
        target_room = env.info['target_room']
        trainer.set_target(target_room)
        episode_targets[-1] = target_room
    else:
        env.reset(target='kitchen')
    assert not np.any(np.isnan(obs)), 'nan detected in the observation!'
    obs = obs.transpose([1, 0, 2])
    logger.print('Observation Shape = {}'.format(obs.shape))

    episode_step = 0
    t = 0
    best_res = -1e50
    elap = time.time()
    update_times = 0
    print('Starting iterations...')
    try:
        while(len(episode_rewards) <= iters):
            idx = trainer.process_observation(obs)
            # get action
            if scheduler is not None:
                noise_level = scheduler.value(len(episode_rewards) - 1)
                action = trainer.action(noise_level)
            else:
                action = trainer.action()
            #proc_action = [np.exp(a) for a in action]
            # environment step
            obs, rew, done, info = env.step(action)
            assert not np.any(np.isnan(obs)), 'nan detected in the observation!'
            obs = obs.transpose([1, 0, 2])
            episode_step += 1
            episode_length[-1] += 1
            terminal = (episode_step >= args['episode_len'])
            # collect experience
            trainer.process_experience(idx, action, rew, done, terminal, info)
            episode_rewards[-1] += rew
            if rew > 5:  # magic number
                episode_success[-1] = 1.0

            if done or terminal:
                trainer.reset_agent()
                if multi_target:
                    obs = env.reset()
                    target_room = env.info['target_room']
                    trainer.set_target(target_room)
                    episode_targets.append(target_room)
                else:
                    obs = env.reset(target='kitchen')
                assert not np.any(np.isnan(obs)), 'nan detected in the observation!'
                obs = obs.transpose([1, 0, 2])
                episode_step = 0
                episode_rewards.append(0)
                episode_success.append(0)
                episode_length.append(0)


            # update all trainers
            trainer.preupdate()
            stats = trainer.update()
            if stats is not None:
                update_times += 1
                if common.debugger is not None:
                    common.debugger.print('>>>>>> Update#{} Finished!!!'.format(update_times), False)

            # save results
            if ((done or terminal) and (len(episode_rewards) % save_rate == 0)) or\
               (len(episode_rewards) > iters):
                trainer.save(save_dir)
                logger.print('Successfully Saved to <{}>'.format(save_dir + '/' + trainer.name + '.pkl'))
                if np.mean(episode_rewards[-eval_range:]) > best_res:
                    best_res = np.mean(episode_rewards[-eval_range:])
                    trainer.save(save_dir, "best")

            # display training output
            if ((update_times % report_rate == 0) and (algo != 'pg') and (stats is not None)) or \
                ((update_times == 0) and (algo != 'pg') and (len(episode_rewards) % 100 == 0) and (done or terminal)) or \
                ((algo == 'pg') and (stats is not None)):
                logger.print('Episode#%d, Updates=%d, Time Elapsed = %.3f min' % (len(episode_rewards), update_times, (time.time()-elap) / 60))
                logger.print('-> Total Samples: %d' % t)
                logger.print('-> Avg Episode Length: %.4f' % (t / len(episode_rewards)))
                if stats is not None:
                    for k in stats:
                        logger.print('  >> %s = %.4f' % (k, stats[k]))
                logger.print('  >> Reward  = %.4f' % np.mean(episode_rewards[-eval_range:]))
                logger.print('  >> Success Rate  = %.4f' % np.mean(episode_success[-eval_range:]))
                if multi_target:
                    ep_rew = episode_rewards[-eval_range:]
                    ep_suc = episode_success[-eval_range:]
                    ep_tar = episode_targets[-eval_range:]
                    ep_len = episode_length[-eval_range:]
                    total_n = len(ep_rew)
                    tar_stats = dict()
                    for k,r,s,l in zip(ep_tar,ep_rew,ep_suc,ep_len):
                        if k not in tar_stats:
                            tar_stats[k] = [0.0, 0.0, 0.0, 0.0]
                        tar_stats[k][0] += 1
                        tar_stats[k][1] += r
                        tar_stats[k][2] += s
                        tar_stats[k][3] += l
                    for k in tar_stats.keys():
                        n, r, s, l = tar_stats[k]
                        logger.print('  --> Multi-Room<%s> Freq = %.4f, Rew = %.4f, Succ = %.4f (AvgLen = %.3f)' % (k, n / total_n, r / n, s / n, l / n))
                print('----> Data Loading Time = %.4f min' % (time_counter[-1] / 60))
                print('----> GPU Data Transfer Time = %.4f min' % (time_counter[0] / 60))
                print('----> Training Time = %.4f min' % (time_counter[1] / 60))
                print('----> Target Net Update Time = %.4f min' % (time_counter[2] / 60))

            t += 1
    except KeyboardInterrupt:
        print('Keyboard Interrupt!!!!!!')
    trainer.save(save_dir, "final")
    with open(save_dir+'/final_training_stats.pkl', 'wb') as f:
        pickle.dump([episode_rewards, episode_success, episode_targets, episode_length], f)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning for 3D House Navigation")
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test'], default='small')
    parser.add_argument("--house", type=int, default=0,
                        help="house ID (default 0); if < 0, then multi-house environment")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--linear-reward", action='store_true', default=False,
                        help="[Deprecated] whether to use reward according to distance; o.w. indicator reward")
    parser.add_argument("--reward-type", choices=['none','linear','indicator','delta','speed'], default='indicator',
                        help="Reward shaping type")
    parser.add_argument("--action-dim", type=int, help="degree of freedom of agent movement, must be in the range of [2, 4], default=4")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'], default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    parser.add_argument("--history-frame-len", type=int, default=4,
                        help="length of the stacked frames, default=4")
    parser.add_argument("--success-measure", choices=['center', 'stay', 'see'], default='center',
                        help="criteria for a successful episode")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--render-gpu", type=int,
                        help="An integer indicating the gpu_id for render. Default by choosing the first GPU in all the accessible devices.")
    # Core training parameters
    parser.add_argument("--algo", choices=['ddpg','pg', 'rdpg', 'ddpg_joint', 'ddpg_alter', 'ddpg_eagle',
                                           'a2c', 'qac', 'dqn'], default="ddpg", help="algorithm")
    parser.add_argument("--model", choices=['cnn','rnn','attentive_cnn','random'], default="cnn", help="policy neural net")
    parser.add_argument("--lrate", type=float, help="learning rate for policy")
    parser.add_argument("--critic-lrate", type=float, help="learning rate for critic")
    parser.add_argument('--weight-decay', type=float, help="weight decay for policy")
    parser.add_argument('--critic-weight-decay', type=float, help="weight decay for critic")
    parser.add_argument("--gamma", type=float, help="discount")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--max-episode-len", type=int, help="maximum episode length")
    parser.add_argument("--update-freq", type=int, help="update model parameters once every this many samples collected")
    parser.add_argument("--max-iters", type=int, default=int(2e6), help="maximum number of training episodes")
    parser.add_argument("--target-net-update-rate", type=float, help="update rate for target networks")
    parser.add_argument("--target-net-update-freq", type=int, help="[Only For DQN] update (copy) frequency for target network. This will De-effect --target-net-update-rate")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--entropy-penalty", type=float, help="policy entropy regularizer")
    parser.add_argument("--critic-penalty", type=float, default=0.001, help="critic norm regularizer")
    parser.add_argument("--replay-buffer-size", type=int, help="size of replay buffer")
    parser.add_argument("--noise-scheduler", choices=['low','medium','high','none','linear','exp'],
                        dest='scheduler', default='medium',
                        help="Whether to use noise-level scheduler to control the smoothness of action output. default=False.")
    parser.add_argument("--use-action-gating", dest='action_gating', action='store_true',
                        help="whether to use action gating structure in the critic model")
    parser.set_defaults(action_gating=False)
    parser.add_argument("--use-target-gating", dest='target_gating', action='store_true',
                        help="[only affect when --multi-target] whether to use target instruction gating structure in the model")
    parser.set_defaults(target_gating=False)
    parser.add_argument("--use-residual-critic", dest='residual_critic', action='store_true',
                        help="whether to use residual structure for feature extraction in the critic model (N.A. for joint-ac model) ")
    parser.set_defaults(residual_critic=False)
    # Attentive DDPG Parameters
    parser.add_argument("--att-resolution", choices=['normal','tiny','low','high','row','row_low','row_tiny'], default="low",
                        help="[Att-CNN-Only] resolution of attention mask (squared input signal not supported yet)")
    parser.add_argument("--att-shared-cnn", dest="att_shared_cnn", action="store_true",
                        help="[Att-CNN-Only] to shared the CNN part for both manager and actor")
    parser.set_defaults(att_shared_cnn=False)
    parser.add_argument("--att-skip-depth", dest="att_skip_depth", action="store_true",
                        help="[Att-CNN-Only] do not attend on the depth channel. only effect when --depth-input flag is on")
    parser.set_defaults(att_skip_depth=False)
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--batch-length", type=int,
                        help="[RNN-Only] maximum length of an episode in a batch")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'],
                        help="[RNN-Only] RNN cell type")
    # Aux Tasks and Additional Sampling Choice
    parser.add_argument("--dist-sampling", dest='dist_sample', action="store_true")
    parser.set_defaults(dist_sample=False)
    parser.add_argument("--q-loss-coef", type=float,
                        help="For joint model, the coefficient for q_loss")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_model_", help="directory in which training state and model should be saved")
    parser.add_argument("--log-dir", type=str, default="./log", help="directory in which logs training stats")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--report-rate", type=int, default=50, help="report training stats once every time this many training steps are performed")
    parser.add_argument("--warmstart", type=str, help="model to recover from. can be either a directory or a file.")
    parser.add_argument("--debug", action="store_true", dest="debug", help="log all the computation details")
    parser.add_argument("--no-debug", action="store_false", dest="debug", help="turn off debug logs")
    parser.set_defaults(debug=False)
    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()

    common.set_house_IDs(cmd_args.env_set, ensure_kitchen=(not cmd_args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (cmd_args.env_set, len(common.all_houseIDs)))

    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)  #optional

    if cmd_args.action_dim is not None:
        print('Degree of freedom set to be <{}>!'.format(cmd_args.action_dim))
        common.action_shape = (cmd_args.action_dim, 2)

    if cmd_args.linear_reward:
        print('--linearReward option is now *Deprecated*!!! Use --reward-type option instead! Now force <reward_type == \'linear\'>')
        cmd_args.reward_type = 'linear'

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    args = common.create_default_args(cmd_args.algo, cmd_args.model, cmd_args.gamma,
                                      cmd_args.lrate, cmd_args.critic_lrate,
                                      cmd_args.max_episode_len, cmd_args.batch_size,
                                      cmd_args.update_freq,
                                      cmd_args.use_batch_norm,
                                      cmd_args.entropy_penalty,
                                      cmd_args.critic_penalty,
                                      cmd_args.weight_decay,
                                      cmd_args.critic_weight_decay,
                                      cmd_args.replay_buffer_size,
                                      # Att-CNN Parameters
                                      cmd_args.att_resolution,
                                      cmd_args.att_skip_depth,
                                      # RNN Parameters
                                      cmd_args.batch_length, cmd_args.rnn_layers,
                                      cmd_args.rnn_cell, cmd_args.rnn_units,
                                      # input type
                                      cmd_args.segmentation_input,
                                      cmd_args.depth_input,
                                      cmd_args.resolution,
                                      cmd_args.history_frame_len
                                      )

    args['algo'] = cmd_args.algo

    args['target_net_update_rate']=cmd_args.target_net_update_rate
    args['target_net_update_freq']=cmd_args.target_net_update_freq

    if cmd_args.hardness is not None:
        args['hardness'] = cmd_args.hardness

    if cmd_args.scheduler is not None:
        args['scheduler'] = create_scheduler(cmd_args.scheduler)

    if cmd_args.dist_sample:
        args['dist_sample'] = True
        assert not cmd_args.multi_target, 'Dist-Sampling is not supported in Multi-Target Training!'

    if cmd_args.q_loss_coef is not None:
        args['q_loss_coef'] = cmd_args.q_loss_coef

    if cmd_args.render_gpu is not None:
        all_gpus = common.get_gpus_for_rendering()
        assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU for rendering!'
        args['render_gpu'] = all_gpus[cmd_args.render_gpu]
    else:
        args['render_gpu'] = None

    args['action_gating'] = cmd_args.action_gating   # gating in ddpg network
    args['residual_critic'] = cmd_args.residual_critic  # resnet for critic (classical ddpg)
    args['multi_target'] = cmd_args.multi_target  # multi-target learning
    args['target_gating'] = cmd_args.target_gating

    # attentive-cnn related params
    args['att_shared_cnn'] = cmd_args.att_shared_cnn
    if 'attentive' in args['model_name']:
        assert args['algo'] == 'ddpg_joint', 'Attentive-CNN Model only supported by DDPG_Joint Algo!!!'
        assert not cmd_args.multi_target, 'Attentive Model is not supported by Multi-Target Training'

    train(args,
          houseID=cmd_args.house,
          reward_type=cmd_args.reward_type,
          success_measure=cmd_args.success_measure,
          multi_target=cmd_args.multi_target,
          algo=cmd_args.algo, model_name=cmd_args.model, iters=cmd_args.max_iters,
          report_rate=cmd_args.report_rate, save_rate=cmd_args.save_rate,
          log_dir=cmd_args.log_dir, save_dir=cmd_args.save_dir,
          warmstart=cmd_args.warmstart,
          log_debug_info=cmd_args.debug)
