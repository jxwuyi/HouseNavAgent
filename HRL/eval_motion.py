from headers import *
import common
import utils

import sys, os, platform, pickle, json, argparse, time

import numpy as np
import random

from HRL.fake_motion import FakeMotion
from HRL.rnn_motion import RNNMotion
from HRL.random_motion import RandomMotion
from HRL.mixture_motion import MixMotion, create_mixture_motion_trainer_dict


def create_motion(args, task):
    if args['motion'] == 'rnn':
        if (args['warmstart_dict'] is not None) and os.path.isfile(args['warmstart_dict']):
            with open(args['warmstart_dict'], 'r') as f:
                trainer_args = json.load(f)
        else:
            trainer_args = args
        common.process_observation_shape('rnn', trainer_args['resolution_level'],
                                         segmentation_input=trainer_args['segment_input'],
                                         depth_input=trainer_args['depth_input'],
                                         history_frame_len=1,
                                         target_mask_input=trainer_args['target_mask_input'])
        import zmq_train
        trainer = zmq_train.create_zmq_trainer('a3c', 'rnn', trainer_args)
        model_file = args['warmstart']
        if model_file is not None:
            trainer.load(model_file)
        trainer.eval()
        motion = RNNMotion(task, trainer,
                           pass_target=args['multi_target'],
                           term_measure=args['terminate_measure'])
    elif args['motion'] == 'random':
        motion = RandomMotion(task, None, term_measure=args['terminate_measure'])
    elif args['motion'] == 'fake':
        motion = FakeMotion(task, None, term_measure=args['terminate_measure'])
    else: # mixture motion
        mixture_dict_file = args['mixture_motion_dict']
        try:
            with open(mixture_dict_file, 'r') as f:
                arg_dict = json.load(f)
        except Exception as e:
            print('Invalid Mixture Motion Dict!! file = <{}>'.format(mixture_dict_file))
            raise e
        trainer_dict, pass_tar_dict, obs_mode_dict = create_mixture_motion_trainer_dict(arg_dict)
        motion = MixMotion(task, trainer_dict, pass_tar_dict,
                           term_measure=args['terminate_measure'],
                           obs_mode=obs_mode_dict)
        common.ensure_object_targets(args['object_target'])

    return motion


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def proc_info(info):
    return dict(yaw=info['yaw'], loc=info['loc'], grid=info['grid'],
                dist=info['dist'])


def evaluate(args):

    elap = time.time()

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()

    fixed_target = args['fixed_target']
    if (fixed_target is not None) and (fixed_target != 'any-room') and (fixed_target != 'any-object'):
        assert fixed_target in common.n_target_instructions, 'invalid fixed target <{}>'.format(fixed_target)

    __backup_CFG = common.CFG.copy()
    if fixed_target == 'any-room':
        common.ensure_object_targets(False)

    if args['hardness'] is not None:
        print('>>>> Hardness = {}'.format(args['hardness']))
    if args['max_birthplace_steps'] is not None:
        print('>>>> Max BirthPlace Steps = {}'.format(args['max_birthplace_steps']))
    set_seed(args['seed'])
    task = common.create_env(args['house'], task_name=args['task_name'], false_rate=args['false_rate'],
                            hardness=args['hardness'], max_birthplace_steps=args['max_birthplace_steps'],
                            success_measure=args['success_measure'],
                            depth_input=args['depth_input'],
                            target_mask_input=args['target_mask_input'],
                            segment_input=args['segmentation_input'],
                            genRoomTypeMap=False,
                            cacheAllTarget=args['multi_target'],
                            render_device=args['render_gpu'],
                            use_discrete_action=True,
                            include_object_target=args['object_target'] and (fixed_target != 'any-room'),
                            include_outdoor_target=args['outdoor_target'],
                            discrete_angle=True)

    if (fixed_target is not None) and (fixed_target != 'any-room') and (fixed_target != 'any-object'):
        task.reset_target(fixed_target)

    if fixed_target == 'any-room':
        common.CFG = __backup_CFG
        common.ensure_object_targets(True)

    # create motion
    motion = create_motion(args, task)

    logger = utils.MyLogger(args['log_dir'], True)
    logger.print('Start Evaluating ...')

    episode_success = []
    episode_good = []
    episode_stats = []
    t = 0
    seed = args['seed']
    max_episode_len = args['max_episode_len']
    for it in range(args['max_iters']):
        cur_infos = []
        motion.reset()
        set_seed(seed + it + 1)  # reset seed
        task.reset(target=fixed_target)
        info = task.info

        episode_success.append(0)
        episode_good.append(0)
        cur_stats = dict(best_dist=info['dist'],
                         success=0, good=0, reward=0, target=task.get_current_target(),
                         meters=task.info['meters'], optstep=task.info['optsteps'], length=max_episode_len, images=None)
        if hasattr(task.house, "_id"):
            cur_stats['world_id'] = task.house._id

        store_history = args['store_history']
        if store_history:
            cur_infos.append(proc_info(task.info))

        if args['temperature'] is not None:
            ep_data = motion.run(task.get_current_target(), max_episode_len,
                                 temperature=args['temperature'])
        else:
            ep_data = motion.run(task.get_current_target(), max_episode_len)

        for dat in ep_data:
            info = dat[4]
            if store_history:
                cur_infos.append(proc_info(info))
            cur_dist = info['dist']
            if cur_dist == 0:
                cur_stats['good'] += 1
                episode_good[-1] = 1
            if cur_dist < cur_stats['best_dist']:
                cur_stats['best_dist'] = cur_dist

        episode_step = len(ep_data)
        if ep_data[-1][3]:  # done
            if ep_data[-1][2] > 5:  # magic number:
                episode_success[-1] = 1
                cur_stats['success'] = 1
        cur_stats['length'] = episode_step   # store length

        if store_history:
            cur_stats['infos'] = cur_infos
        episode_stats.append(cur_stats)

        dur = time.time() - elap
        logger.print('Episode#%d, Elapsed = %.3f min' % (it+1, dur/60))
        if args['multi_target']:
            logger.print('  ---> Target Room = {}'.format(cur_stats['target']))
        logger.print('  ---> Total Samples = {}'.format(t))
        logger.print('  ---> Success = %d  (rate = %.3f)'
                     % (cur_stats['success'], np.mean(episode_success)))
        logger.print('  ---> Times of Reaching Target Room = %d  (rate = %.3f)'
                     % (cur_stats['good'], np.mean(episode_good)))
        logger.print('  ---> Best Distance = %d' % cur_stats['best_dist'])
        logger.print('  ---> Birth-place Distance = %d' % cur_stats['optstep'])

    logger.print('######## Final Stats ###########')
    logger.print('Success Rate = %.3f' % np.mean(episode_success))
    logger.print('> Avg Ep-Length per Success = %.3f' % np.mean([s['length'] for s in episode_stats if s['success'] > 0]))
    logger.print('> Avg Birth-Meters per Success = %.3f' % np.mean([s['meters'] for s in episode_stats if s['success'] > 0]))
    logger.print('Reaching Target Rate = %.3f' % np.mean(episode_good))
    logger.print('> Avg Ep-Length per Target Reach = %.3f' % np.mean([s['length'] for s in episode_stats if s['good'] > 0]))
    logger.print('> Avg Birth-Meters per Target Reach = %.3f' % np.mean([s['meters'] for s in episode_stats if s['good'] > 0]))
    if args['multi_target']:
        all_targets = list(set([s['target'] for s in episode_stats]))
        for tar in all_targets:
            n = sum([1.0 for s in episode_stats if s['target'] == tar])
            succ = [float(s['success'] > 0) for s in episode_stats if s['target'] == tar]
            good = [float(s['good'] > 0) for s in episode_stats if s['target'] == tar]
            length = [s['length'] for s in episode_stats if s['target'] == tar]
            meters = [s['meters'] for s in episode_stats if s['target'] == tar]
            good_len = np.mean([l for l,g in zip(length, good) if g > 0.5])
            succ_len = np.mean([l for l,s in zip(length, succ) if s > 0.5])
            good_mts = np.mean([l for l, g in zip(meters, good) if g > 0.5])
            succ_mts = np.mean([l for l, s in zip(meters, succ) if s > 0.5])
            logger.print(
                '>>>>> Multi-Target <%s>: Rate = %.3f (n=%d), Good = %.3f (AvgLen=%.3f; Mts=%.3f), Succ = %.3f (AvgLen=%.3f; Mts=%.3f)'
                % (tar, n/len(episode_stats), n, np.mean(good), good_len, good_mts, np.mean(succ), succ_len, succ_mts))

    return episode_stats


def parse_args():
    parser = argparse.ArgumentParser("Evaluation Locomotion for 3D House Navigation")
    # Select Task
    parser.add_argument("--task-name", choices=['roomnav', 'objnav'], default='roomnav')
    parser.add_argument("--false-rate", type=float, default=0, help='The Rate of Impossible Targets')
    # Environment
    parser.add_argument("--env-set", choices=['small', 'train', 'test', 'color'], default='small')
    parser.add_argument("--house", type=int, default=0, help="house ID")
    parser.add_argument("--render-gpu", type=int, help="gpu id for rendering the environment")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--hardness", type=float, help="real number from 0 to 1, indicating the hardness of the environment")
    parser.add_argument("--max-birthplace-steps", type=int, help="int, the maximum steps required from birthplace to target")
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'], default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--target-mask-input", dest='target_mask_input', action='store_true',
                        help="whether to include target mask 0/1 signal as part of the input signal")
    parser.set_defaults(target_mask_input=False)
    parser.add_argument("--success-measure", choices=['stop', 'stay', 'see'], default='see',
                        help="criteria for a successful episode")
    parser.add_argument("--terminate-measure", choices=['mask', 'stay', 'see'], default='mask',
                        help="criteria for terminating a motion execution")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, a new target room will be selected per episode")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also a target. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--no-outdoor-target", dest='outdoor_target', action='store_false',
                        help="when this flag is set, we will exclude <outdoor> target")
    parser.set_defaults(outdoor_target=True)
    parser.add_argument("--only-eval-room-target", dest='only_eval_room', action='store_true',
                        help="when this flag is set, only evaluate room targets. only effective when --include-object-target")
    parser.set_defaults(only_eval_room=False)
    parser.add_argument("--only-eval-object-target", dest='only_eval_object', action='store_true',
                        help="when this flag is set, only evaluate object targets. only effective when --include-object-target")
    parser.set_defaults(only_eval_object=False)
    parser.add_argument("--fixed-target", choices=common.ALLOWED_TARGET_ROOM_TYPES + common.ALLOWED_OBJECT_TARGET_TYPES + ['any-room', 'any-object'],
                        help="once set, all the episode will be fixed to a specific target.")
    # Core parameters
    parser.add_argument("--motion", choices=['rnn', 'fake', 'random', 'mixture'], default="fake", help="type of the locomotion")
    parser.add_argument("--mixture-motion-dict", type=str, help="dict for mixture-motion, only effective when --motion mixture")
    parser.add_argument("--max-episode-len", type=int, default=2000, help="maximum episode length")
    parser.add_argument("--max-iters", type=int, default=1000, help="maximum number of eval episodes")
    parser.add_argument("--store-history", action='store_true', default=False, help="whether to store all the episode frames")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--use-target-gating", dest='target_gating', action='store_true',
                        help="[only affect when --multi-target] whether to use target instruction gating structure in the model")
    parser.set_defaults(target_gating=False)
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'],
                        help="[RNN-Only] RNN cell type")
    # Auxiliary Task Options
    parser.add_argument("--auxiliary-task", dest='aux_task', action='store_true',
                        help="Whether to perform auxiliary task of predicting room types")
    parser.set_defaults(aux_task=False)
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="./log/eval", help="directory in which logs eval stats")
    parser.add_argument("--warmstart", type=str, help="file to load the policy model")
    parser.add_argument("--warmstart-dict", type=str, help="arg dict the policy model, only effective when --motion rnn")
    # Other
    parser.add_argument("--temperature", type=float, help="temperature for executing motion; only effective when --motion rnn/mixture")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    assert (args.warmstart is None) or (os.path.exists(args.warmstart)), 'Model File Not Exists!'

    assert not args.aux_task, 'Currently do not support Aux-Task!'

    common.set_house_IDs(args.env_set, ensure_kitchen=(not args.multi_target))
    print('>> Environment Set = <%s>, Total %d Houses!' % (args.env_set, len(common.all_houseIDs)))

    if args.object_target:
        common.ensure_object_targets()

    if not os.path.exists(args.log_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(args.log_dir))
        os.makedirs(args.log_dir)

    if args.motion == 'rnn':
        assert args.warmstart is not None
    if args.motion == 'mixture':
        assert args.mixture_motion_dict is not None

    if args.fixed_target is None:
        if args.only_eval_room:
            args.fixed_target = 'any-room'
        elif args.only_eval_object:
            args.fixed_target = 'any-object'

    if args.seed is None:
        args.seed = 0

    if args.temperature is not None:
        assert args.motion in ['rnn', 'mixture']

    dict_args = args.__dict__

    episode_stats = evaluate(dict_args)

    if args.store_history:
        filename = args.log_dir
        if filename[-1] != '/':
            filename += '/'
        filename += args.motion+'_full_eval_history.pkl'
        print('Saving all stats to <{}> ...'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump([episode_stats, args], f)
        print('  >> Done!')
