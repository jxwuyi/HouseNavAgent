from headers import *
import common
import utils

import sys, os, platform, pickle, json, argparse, time

import numpy as np
import random

from HRL.fake_motion import FakeMotion
from HRL.rnn_motion import RNNMotion
from HRL.BayesGraph import GraphPlanner


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def proc_info(info):
    return dict(yaw=info['yaw'], loc=info['loc'], grid=info['grid'],
                dist=info['dist'])


def evaluate(args):

    args['segment_input'] = args['segmentation_input']

    elap = time.time()

    # Do not need to log detailed computation stats
    common.debugger = utils.FakeLogger()

    # ensure observation shape
    common.process_observation_shape('rnn', args['resolution'],
                                     args['segmentation_input'],
                                     args['depth_input'],
                                     target_mask_input=args['target_mask_input'])

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
    if args['motion'] == 'rnn':
        import zmq_train
        trainer = zmq_train.create_zmq_trainer('a3c', 'rnn', args)
        model_file = args['warmstart']
        if model_file is not None:
            trainer.load(model_file)
        trainer.eval()
        motion = RNNMotion(task, trainer)
    else:
        motion = FakeMotion(task, None)

    # create planner
    graph = None
    max_motion_steps = args['n_exp_steps']
    if args['planner'] == 'rnn':
        assert False, 'Currently only support Graph-planner'
    else:
        graph = GraphPlanner(motion)
        if not args['outdoor_target']:
            graph.add_excluded_target('outdoor')
        # hack
        #graph.set_param(-1, 0.85)

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
        task_target = task.get_current_target()
        cur_stats = dict(best_dist=info['dist'],
                         success=0, good=0, reward=0, target=task_target,
                         plan=[],
                         optstep=task.info['optsteps'], length=max_episode_len, images=None)
        if hasattr(task.house, "_id"):
            cur_stats['world_id'] = task.house._id

        store_history = args['store_history']
        if store_history:
            cur_infos.append(proc_info(task.info))

        episode_step = 0

        # reset planner
        if graph is not None:
            graph.reset()

        while episode_step < max_episode_len:
            graph_target = graph.plan(task.get_feature_mask(), task_target)
            graph_target_id = graph.get_target_index(graph_target)
            allowed_steps = min(max_episode_len - episode_step, max_motion_steps)

            motion_data = motion.run(graph_target, allowed_steps)

            cur_stats['plan'].append((graph_target, len(motion_data), (motion_data[-1][0][graph_target_id] > 0)))

            # store stats
            for dat in motion_data:
                info = dat[4]
                if store_history:
                    cur_infos.append(proc_info(info))
                cur_dist = info['dist']
                if cur_dist == 0:
                    cur_stats['good'] += 1
                    episode_good[-1] = 1
                if cur_dist < cur_stats['best_dist']:
                    cur_stats['best_dist'] = cur_dist

            # update graph
            graph.observe(motion_data, graph_target)

            episode_step += len(motion_data)

            # check done
            if motion_data[-1][3]:
                if motion_data[-1][2] > 5: # magic number
                    episode_success[-1] = 1
                    cur_stats['success'] = 1
                break

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
        logger.print('  ---> Planner Results = {}'.format(cur_stats['plan']))

    logger.print('######## Final Stats ###########')
    logger.print('Success Rate = %.3f' % np.mean(episode_success))
    logger.print('Avg Length per Success = %.3f' % np.mean([s['length'] for s in episode_stats if s['success'] > 0]))
    logger.print('Reaching Target Rate = %.3f' % np.mean(episode_good))
    logger.print('Avg Length per Target Reach = %.3f' % np.mean([s['length'] for s in episode_stats if s['good'] > 0]))
    if args['multi_target']:
        all_targets = list(set([s['target'] for s in episode_stats]))
        for tar in all_targets:
            n = sum([1.0 for s in episode_stats if s['target'] == tar])
            succ = [float(s['success'] > 0) for s in episode_stats if s['target'] == tar]
            good = [float(s['good'] > 0) for s in episode_stats if s['target'] == tar]
            length = [s['length'] for s in episode_stats if s['target'] == tar]
            good_len = np.mean([l for l,g in zip(length, good) if g > 0.5])
            succ_len = np.mean([l for l,s in zip(length, succ) if s > 0.5])
            logger.print(
                '>>>>> Multi-Target <%s>: Rate = %.3f (n=%d), Good = %.3f (AvgLen=%.3f), Succ = %.3f (AvgLen=%.3f)'
                % (tar, n/len(episode_stats), n, np.mean(good), good_len, np.mean(succ), succ_len))

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
    parser.add_argument("--fixed-target", choices=common.ALLOWED_TARGET_ROOM_TYPES + common.ALLOWED_OBJECT_TARGET_TYPES,
                        help="once set, all the episode will be fixed to a specific target.")
    # Core parameters
    parser.add_argument("--motion", choices=['rnn', 'fake'], default="fake", help="type of the locomotion")
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
    # Planner Parameters
    parser.add_argument("--planner", choices=['rnn', 'graph'], default='graph', help='type of the planner')
    parser.add_argument("--planner-filename", type=str, help='parameters for the planners')
    parser.add_argument("--n-exp-steps", type=int, default=40, help='maximum number of steps for exploring a sub-policy')
    # Auxiliary Task Options
    parser.add_argument("--auxiliary-task", dest='aux_task', action='store_true',
                        help="Whether to perform auxiliary task of predicting room types")
    parser.set_defaults(aux_task=False)
    # Checkpointing
    parser.add_argument("--log-dir", type=str, default="./log/eval", help="directory in which logs eval stats")
    parser.add_argument("--warmstart", type=str, help="file to load the policy model")
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

    if args.motion != 'fake':
        assert args.warmstart is not None

    if args.fixed_target is None:
        if args.only_eval_room:
            args.fixed_target = 'any-room'
        elif args.only_eval_object:
            args.fixed_target = 'any-object'

    if args.seed is None:
        args.seed = 0

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
