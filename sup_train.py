from headers import *
import common
import utils

import threading

from trainer.supervise import SUPTrainer

from policy.rnn_discrete_actor_critic import DiscreteRNNPolicy

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


mask_feat_dim = None

batch_frames = batch_len_mask = batch_actions = batch_mask_feat = None

test_batch_frames = test_batch_len_mask = test_batch_actions = test_batch_mask_feat = None


def create_policy(model_name, args, observation_shape, n_action):
    assert model_name == 'rnn', 'currently only support rnn policy!'
    model = DiscreteRNNPolicy(observation_shape, n_action,
                              conv_hiddens=[64, 64, 128, 128],
                              kernel_sizes=5, strides=2,
                              linear_hiddens=[256],
                              policy_hiddens=[128, 64],
                              critic_hiddens=[64, 32],
                              rnn_cell=args['rnn_cell'],
                              rnn_layers=args['rnn_layers'],
                              rnn_units=args['rnn_units'],
                              multi_target=args['multi_target'],
                              use_target_gating=args['target_gating'],
                              aux_prediction=None,
                              no_skip_connect=(args['no_skip_connect'] if 'no_skip_connect' in args else False),
                              pure_feed_forward=(args['feed_forward'] if 'feed_forward' in args else False),
                              extra_feature_dim=(len(common.all_target_instructions) if ('mask_feature' in args) and args['mask_feature'] else None)
                              )
    if common.use_cuda:
        if 'train_gpu' in args:
            train_gpus = args['train_gpu']
            if isinstance(train_gpus, list):  # multi-gpu
                #model = torch.nn.DataParallel(model, device_ids=train_gpus)
                model.cuda(device_id=train_gpus[0])
            else:  # single gpu
                model.cuda(device_id=train_gpus)  # TODO: Actually we only support training on gpu_id=0
        else:
            model.cuda()
    return model


def create_trainer(args):
    observation_shape = common.observation_shape
    n_action = common.n_discrete_actions
    model_gen = lambda: create_policy('rnn', args, observation_shape, n_action)
    trainer = SUPTrainer(model_gen, observation_shape, n_action, args)
    return trainer


def data_loader(data_dir, n_part, t_max=None, fixed_target=None, mask_feature=False, logger=None):
    # data_dir: directory of data partitions
    # n_part: the number of partitions
    # t_max: maximum allowed steps
    # fixed_target: required target
    # mask_feature: whether to use mask feature
    # @return: np_frames, np_length, np_actions, np_target, (optional) np_mask_feature

    def myprint(s):
        if logger is None:
            print(s)
        else:
            logger.print(s)

    dur = time.time()
    myprint('Data-Loading: dir = <{}>, partition = <{}> ...'.format(data_dir, n_part))

    if not os.path.exists(data_dir):
        myprint('[ERROR] data dir <{}> does not exist!'.format(data_dir))
        assert False
    accu_infos = []
    accu_data = []
    total_samples = 0
    max_t_step = 0
    frame_shape = None
    mask_feat_dim = None

    def check_allowed_target(target, fixed_target):
        if fixed_target is None:
            return True
        if 'any' not in fixed_target:
            return target == fixed_target
        if 'room' in fixed_target:
            return target in common.ALLOWED_TARGET_ROOM_TYPES
        # any-object
        return target in common.ALLOWED_OBJECT_TARGET_TYPES

    for p in range(n_part):
        part_file = os.path.join(data_dir, 'partition%d.pkl' % p)
        if not os.path.exists(part_file):
            myprint('[WARNING] data partition <{}> does not exist!'.format(part_file))
            continue
        with open(part_file, 'rb') as f:
            args, birth_infos, data = pickle.load(f)
        if mask_feature:
            assert (args['mask_feature_dim'] is not None), '[ERROR] data partition <{}> does not have mask_feature, which is required!'.format(part_file)
            assert (mask_feat_dim is None) or (mask_feat_dim == args['mask_feat_dim']), '[ERROR] inconsistent <mask_feat_dim> in data partition <{}>!'.format(part_file)
            assert (len(data[0]) > 2) and not isinstance(data[0][2], dict), '[ERROR] missing np_mask_feature in data partition <{}>'.format(part_file)
            mask_feat_dim = args['mask_feat_dim']
        accu_infos.append(birth_infos)
        accu_data.append(data)
        cur_len = [len(dat[1]) for info, dat in zip(birth_infos, data) if check_allowed_target(info['target_room'], fixed_target)]
        total_samples += len(cur_len)
        max_t_step = max(max_t_step, max(cur_len))
        if frame_shape is None:
            frame_shape = data[0][0].shape
    assert frame_shape is not None
    assert total_samples > 0, '[ERROR] No data found!'
    myprint('[Data_Loader] Max-T in data is {}'.format(max_t_step))
    if t_max is None:
        t_max = max_t_step
    myprint('  --> Selected T-Max = {}'.format(t_max))
    #np_frames = np.zeros((total_samples, t_max) + frame_shape, dtype=np.uint8)
    #np_actions = np.zeros((total_samples, t_max), dtype=np.int32)
    lis_frames = []
    lis_actions = []
    np_length = np.zeros((total_samples, ), dtype=np.int32)
    #np_mask_feat = np.zeros((total_samples, t_max, mask_feat_dim), dtype=np.uint8) if mask_feature else None
    lis_mask_feat = []
    np_target = np.zeros((total_samples, ), dtype=np.uint8)
    ptr = 0
    for infos, data in zip(accu_infos, accu_data):
        for info, dat in zip(infos, data):
            # dat: np_frame, np_action, (np_mask_feat), (logs)
            if not check_allowed_target(info['target_room'], fixed_target): continue
            l = len(dat[1])
            np_length[ptr] = min(t_max, l)
            np_target[ptr] = common.target_instruction_dict[info['target_room']]
            if l > t_max: # shrink
                lis_frames.append(dat[0][l - t_max:, ...])
                lis_actions.append(dat[1][l - t_max :])
                if mask_feature: lis_mask_feat.append(dat[2][l - t_max:, ...])
            else:
                lis_frames.append(dat[0])
                lis_actions.append(dat[1])
                if mask_feature: lis_mask_feat.append(dat[3])
            ptr += 1
    dur = time.time() - dur
    myprint(' >> Done! Total Samples = %d, Max-T = %d, Time Elapsed = %.5s (avg = %.4fs)' % (total_samples, t_max, dur, dur / total_samples))
    if not mask_feature:
        lis_mask_feat = None
    return lis_frames, np_length, lis_actions, np_target, lis_mask_feat, t_max


def eval_model(test_data, test_size, batch_size, trainer, logger):
    # test_data: lis_frames, np_length, lis_actions, np_target, lis_mask_feat, t_max
    global test_batch_frames, test_batch_len_mask, test_batch_actions, test_batch_mask_feat, mask_feat_dim
    report_rate = 1.0 / 4
    accu_report_rate = report_rate
    dur = time.time()
    test_epochs = test_size // batch_size

    # Set to eval mode
    trainer.eval()

    # Evaluation
    logger.print('+++++++++++++ Evaluation +++++++++++++')
    total_correct = 0
    total_samples = 0
    for ep in range(test_epochs):
        L, R = ep * batch_size, (ep+1) * batch_size
        cur_indices = list(range(L, R))
        test_batch_len_mask[...] = 0
        for i, k in enumerate(cur_indices):
            l = test_data[1][k]
            total_samples += l
            test_batch_frames[i, :l, ...] = test_data[0][k]
            test_batch_len_mask[i, :l] = 1
            test_batch_actions[i, :l] = test_data[2][k]
            if mask_feat_dim is not None:
                test_batch_mask_feat[i, :l, ...] = test_data[4][k]
        # run forward pass
        act_idx = trainer.action(test_batch_frames, target=test_data[3][cur_indices],
                                return_numpy=True, mask_input=test_batch_mask_feat, greedy_act=True)  # [batch, seq_len]
        flag_actions = (act_idx == test_batch_actions) * test_batch_len_mask
        total_correct += np.sum(flag_actions)
        # output log
        cur_rate = (ep + 1) / test_epochs
        if cur_rate > accu_report_rate:
            accu_report_rate += report_rate
            logger.print(' -->> Eval Samples = %d (Traj = %d), Percent = %.3f (Elapsed = %.3f): Accuracy = %.4f' % 
                        (total_samples, ep * batch_size, cur_rate, time.time() - dur, total_correct / total_samples))
    logger.print('>> DONE! Time Elapsed = %.3f' % (time.time() - dur))
    logger.print(' > Total Samples = %d, Correct = %d, Accuracy = %.4f' % (total_samples, total_correct, total_correct / total_samples))
    logger.print('++++++++++++++++++++++++++++++++++++++')


def train(args=None, warmstart=None):

    # Process Observation Shape
    common.process_observation_shape(model='rnn',
                                     resolution_level=args['resolution_level'],
                                     segmentation_input=args['segment_input'],
                                     depth_input=args['depth_input'],
                                     target_mask_input=args['target_mask_input'],
                                     history_frame_len=1)

    args['logger'] = logger = utils.MyLogger(args['log_dir'], True, keep_file_handler=not args['append_file'])

    tstart = time.time()
    logger.print('Data Loading ...')

    batch_size = args['batch_size']

    ############################
    # Training Data
    train_data = data_loader(args['data_dir'], args['n_part'], args['t_max'], mask_feature=args['target_mask_input'], logger=logger)
    train_size = len(train_data[0])
    # cache training batch memory
    global mask_feat_dim, batch_frames, batch_len_mask, batch_actions, batch_mask_feat
    batch_frames = np.zeros((batch_size, train_data[-1], ) + train_data[0][0].shape[1:], dtype=np.uint8)
    batch_len_mask = np.zeros((batch_size, train_data[-1]), dtype=np.uint8)
    batch_actions = np.zeros((batch_size, train_data[-1]), dtype=np.int32)
    mask_feat_dim = train_data[4][0].shape[-1] if train_data[4] is not None else None
    batch_mask_feat = np.zeros((batch_size, train_data[-1], mask_feat_dim)) if mask_feat_dim is not None else None
    ############################
    # Training Data
    test_data = None
    test_size = 0
    if args['eval_dir'] and args['eval_n_part']:
        test_data = data_loader(args['eval_dir'], args['eval_n_part'], args['t_max'], mask_feature=args['target_mask_input'], logger=logger)
        test_size = len(test_data[0])
        global test_batch_frames, test_batch_len_mask, test_batch_actions, test_batch_mask_feat
        test_batch_frames = np.zeros((batch_size, test_data[-1], ) + train_data[0][0].shape[1:], dtype=np.uint8)
        test_batch_len_mask = np.zeros((batch_size, test_data[-1]), dtype=np.uint8)
        test_batch_actions = np.zeros((batch_size, test_data[-1]), dtype=np.int32)
        test_batch_mask_feat = np.zeros((batch_size, test_data[-1], mask_feat_dim)) if mask_feat_dim is not None else None

    logger.print(' --> Loading Finished! Elapsed = %.4fs' % (time.time() - tstart))

    # create trainer
    trainer = create_trainer(args)
    if warmstart is not None:
        if os.path.exists(warmstart):
            logger.print('Warmstarting from <{}> ...'.format(warmstart))
            trainer.load(warmstart)
        else:
            assert False, '[Error] Warmstart model <{}> not exists!'.format(warmstart)


    #######################
    # Training
    try:
        # both loops must be running
        logger.print('Start Iterations ....')
        indices = list(range(train_size))
        epoch_updates = (train_size + batch_size - 1) // batch_size
        
        for ep in range(args['epochs']):
            dur = time.time()
            # set to train mode
            trainer.train()
            # start training current epoch
            logger.print('Training Epoch#{}/{} ....'.format(ep, args['epochs']))
            random.shuffle(indices)
            for up in range(epoch_updates):
                L, R = up * batch_size, (up + 1) * batch_size
                if R <= train_size:
                    cur_indices = indices[L: R]
                else:
                    cur_indices = indices[L: ] + indices[: R - train_size]
                # lis_frames, np_length, lis_actions, np_target, lis_mask_feat, t_max = data
                seq_len = max(train_data[1][cur_indices])
                batch_len_mask[...] = 0
                for i, k in enumerate(cur_indices):
                    l = train_data[1][k]
                    batch_frames[i, :l, ...] = train_data[0][k]
                    batch_actions[i, :l] = train_data[2][k]
                    batch_len_mask[i, :l] = 1
                    if mask_feat_dim is not None:
                        batch_mask_feat[i, :l, ...] = train_data[4][k]
                # train
                #stats =\
                #trainer.update(batch_frames[:, :seq_len, ...], batch_actions[:, :seq_len], batch_len_mask[:, :seq_len],
                #               target=train_data[3][cur_indices],
                #               mask_input=batch_mask_feat[:, :seq_len, ...] if mask_feat_dim is not None else None)
                stats = \
                trainer.update(batch_frames, batch_actions, batch_len_mask, 
                               target=train_data[3][cur_indices], mask_input=batch_mask_feat if mask_feat_dim is not None else None)
                if (up + 1) % args['report_rate'] == 0:
                    ep_dur = time.time() - dur
                    logger.print('--> Epoch#%d / %d, Update#%d / %d, Percent = %.4f, Total Elapsed = %.4fs, Epoch Elapsed = %.4fs (Avg: %.4fs)' % 
                        (ep + 1, args['epochs'], up + 1, epoch_updates, (up + 1) / epoch_updates, time.time()-tstart, ep_dur, ep_dur / (up + 1)))
                    for k in sorted(stats.keys()):
                        logger.print('   >> %s = %.4f' % (k, stats[k]))
            # Epoch Finished
            logger.print('>> Epoch#{} Done!'.format(ep + 1))
            if (test_data is not None) and (args['eval_rate'] is not None) and ((ep + 1) % args['eval_rate'] == 0):
                eval_model(test_data, test_size, args['eval_batch_size'], trainer, logger)
            if (ep + 1) % args['save_rate'] == 0:
                trainer.save(args['save_dir'])

        logger.print('Done!')
        trainer.save(args['save_dir'], version='final')
    except KeyboardInterrupt:
        trainer.save(args['save_dir'], version='interrupt')
        raise


def parse_args():
    parser = argparse.ArgumentParser("Supervised Learning for 3D House Navigation")
    ########################################################
    # Data Setting
    parser.add_argument("--data-dir", type=str,
                        help="the directory containing data partitions")
    parser.add_argument("--n-part", type=int,
                        help="number of partitions")
    parser.add_argument("--seed", type=int, help="random seed")
    ########################################################
    # Environment Setting
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none', dest='segment_input',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--target-mask-input", dest='target_mask_input', action='store_true',
                        help="whether to include target mask 0/1 signal as part of the input signal")
    parser.set_defaults(target_mask_input=False)
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'],
                        dest='resolution_level', default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    #parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--multi-target", dest='multi_target', action='store_true',
                        help="when this flag is set, the policy will be a multi-target policy")
    parser.set_defaults(multi_target=False)
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also an object. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--include-mask-feature", dest='mask_feature', action='store_true',
                        help="when this flag is set, mast_feature will be fed to the neural network.")
    parser.set_defaults(mask_feature=False)
    parser.add_argument("--fixed-target", type=str, help="fixed training target: candidate values room, object or any-room/object")
    ########################################################
    # sup training parameters
    parser.add_argument("--train-gpu", type=str,
                        help="[SUP] an integer or a ','-split list of integers, indicating the gpu-id for training")
    parser.add_argument("--t-max", type=int,
                        help="[SUP] number of time steps in each batch")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="[SUP] batch size, should be no greather than --num-proc")
    parser.add_argument("--grad-batch", type=int, default=1,
                        help="[SUP] the actual gradient descent batch-size will be <grad-batch> * <batch-size>")

    ###########################################################
    # Core training parameters
    parser.add_argument("--lrate", type=float, help="learning rate for policy")
    parser.add_argument('--weight-decay', type=float, help="weight decay for policy")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="gradient clipping")
    parser.add_argument("--epochs", type=int, default=int(1e6), help="maximum number of training episodes")
    parser.add_argument("--batch-norm", action='store_true', dest='use_batch_norm',
                        help="Whether to use batch normalization in the policy network. default=False.")
    parser.set_defaults(use_batch_norm=False)
    parser.add_argument("--entropy-penalty", type=float, help="policy entropy regularizer")
    parser.add_argument("--logits-penalty", type=float, help="policy logits regularizer")
    parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam', help="optimizer")
    parser.add_argument("--use-target-gating", dest='target_gating', action='store_true',
                        help="[only affect when --multi-target] whether to use target instruction gating structure in the model")
    parser.set_defaults(target_gating=False)

    ####################################################
    # RNN Parameters
    parser.add_argument("--rnn-units", type=int, default=256,
                        help="[RNN-Only] number of units in an RNN cell")
    parser.add_argument("--rnn-layers", type=int, default=1,
                        help="[RNN-Only] number of layers in RNN")
    parser.add_argument("--rnn-cell", choices=['lstm', 'gru'], default='lstm',
                        help="[RNN-Only] RNN cell type")

    ###################################################
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_model_/supervised", help="directory in which training state and model should be saved")
    parser.add_argument("--log-dir", type=str, default="./log/supervised", help="directory in which logs training stats")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every epochs completed")
    parser.add_argument("--report-rate", type=int, default=1,
                        help="report training stats once every time this many training steps are performed")
    parser.add_argument("--warmstart", type=str, help="model to recover from. can be either a directory or a file.")

    ###################################################
    # Logging Option
    parser.add_argument("--append-file-handler", dest='append_file', action='store_true',
                        help="[Logging] When set, the logger will be close when a log message is output and reopen in the next time.")
    parser.set_defaults(append_file=False)

    ###################################################
    # Evaluation Parameters
    parser.add_argument("--eval-rate", type=int,
                        help="[EVAL] report evaluation results once every time this many epochs finished. 0 or None means no evaluation")
    parser.add_argument("--eval-dir", type=str,
                        help="[EVAL] the directory containing test data partitions")
    parser.add_argument("--eval-n-part", type=int,
                        help="[EVAL] number of test data partitions")
    parser.add_argument("--eval-batch-size", type=int, default=64,
                        help="[EVAL] number of batch size for evaluation")

    return parser.parse_args()

if __name__ == '__main__':
    cmd_args = parse_args()

    common.ensure_object_targets(cmd_args.object_target)

    if cmd_args.fixed_target is not None:
        allowed_targets = list(common.target_instruction_dict.keys()) + ['any-room']
        if cmd_args.object_target:
            allowed_targets.append('any-object')
        assert cmd_args.fixed_target in allowed_targets, '--fixed-target specified an invalid target <{}>!'.format(cmd_args.fixed_target)

    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)
        random.seed(cmd_args.seed)
        torch.manual_seed(cmd_args.seed)  #optional

    if not os.path.exists(cmd_args.save_dir):
        print('Directory <{}> does not exist! Creating directory ...'.format(cmd_args.save_dir))
        os.makedirs(cmd_args.save_dir)

    if cmd_args.grad_batch < 1:
        print('--grad-batch option must be a positive integer! reset to default value <1>!')
        cmd_args.grad_batch = 1

    args = cmd_args.__dict__

    args['model_name'] = 'rnn'

    args['mask_feature_dim'] = len(common.all_target_instructions) if ('mask_feature' in args) and args['mask_feature'] else None

    # gpu devices
    all_gpus = common.get_gpus_for_rendering()
    assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU.'
    if args['train_gpu'] is None:
        print('Warning: training gpu not specified. Set to the default gpu <{}> (single-gpu)'.format(all_gpus[0]))
        args['train_gpu'] = all_gpus[0]
    else:
        gpu_ids = args['train_gpu'].split(',')
        train_gpus = [all_gpus[int(k)] for k in gpu_ids]
        if len(gpu_ids) > 1:
            print('>>> Multi-GPU training specified!')
            args['train_gpu'] = train_gpus
        else:
            args['train_gpu'] = train_gpus[0]

    # store training args
    config_file = args['save_dir']
    if config_file[-1] != '/': config_file = config_file + '/'
    config_file = config_file + 'train_args.json'
    with open(config_file, 'w') as f:
        json.dump(args, f)

    train(args, warmstart=cmd_args.warmstart)
