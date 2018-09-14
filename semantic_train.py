from headers import *
import common
import utils

import threading

from trainer.semantic import SemanticTrainer

from policy.cnn_classifier import CNNClassifier

import os, sys, time, pickle, json, argparse
import numpy as np
import random
import torch


mask_feat_dim = None

batch_frames = batch_labels = None

test_batch_frames = test_batch_labels = None


def create_policy(args, observation_shape, n_class):
    model = CNNClassifier(observation_shape, n_class,
                          #hiddens=[64, 64, 128, 128],
                          #kernel_sizes=5, strides=2,
                          #linear_hiddens=[128, 64],
                          hiddens=[4, 8, 16, 16, 32, 32, 64, 64, 128, 256],
                          kernel_sizes=[3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                          strides=[1, 1, 1,  2,  1, 2, 1, 2, 1, 2],
                          linear_hiddens=[32],
                          use_batch_norm=args['batch_norm'],
                          multi_label=False,
                          dropout_rate=0.05)
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


def create_trainer(args, n_class):
    observation_shape = common.observation_shape
    model_gen = lambda: create_policy(args, observation_shape, n_class)
    trainer = SemanticTrainer(model_gen, observation_shape, n_class, args)
    return trainer


def data_loader(data_dir, n_part, fixed_target=None, logger=None, neg_rate=1, stack_frame=None):
    # data_dir: directory of data partitions
    # n_part: the number of partitions
    # fixed_target: required target, if not None, use binary classification; else softmax classification
    # @return: np_frames, np_label

    #################################################
    # TODO: support general training with softmax
    assert fixed_target and ('any' not in fixed_target), '[ERROR] Only support single semantic label prediction'
    #################################################

    label_index = dict()
    label_names = []
    if fixed_target == 'any-object':
        for i, o in enumerate(common.ALLOWED_OBJECT_TARGET_TYPES):
            label_index[o] = i
            label_names.append(o)
    else:
        if not fixed_target or ('any' in fixed_target):
            for i, r in enumerate(common.ALLOWED_TARGET_ROOM_TYPES):
                label_index[r] = i
                label_names.append(r)
        else:
            label_index[fixed_target] = 0
            label_names.append(fixed_target)
    n_NA = len(label_index)
    label_index['NA'] = n_NA
    label_names.append('NA')
    n_class = n_NA + 1
    label_stats = dict()
    for n in label_names:
        label_stats[n] = 0

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

    accu_data = []
    accu_label = []
    frame_shape = None

    def check_allowed_target(target, fixed_target):
        if fixed_target is None:
            return True
        if 'any' not in fixed_target:
            return target == fixed_target
        if 'room' in fixed_target:
            return target in common.ALLOWED_TARGET_ROOM_TYPES
        # any-object
        return target in common.ALLOWED_OBJECT_TARGET_TYPES


    n_pos_samples = 0
    n_neg_samples = 0
    neg_samples_need = 0
    for p in range(n_part):
        part_file = os.path.join(data_dir, 'partition%d.pkl' % p)
        if not os.path.exists(part_file):
            myprint('[WARNING] data partition <{}> does not exist!'.format(part_file))
            continue
        with open(part_file, 'rb') as f:
            _, birth_infos, all_data = pickle.load(f)
        for info, data in zip(birth_infos, all_data):
            if not check_allowed_target(info['target_room'], fixed_target):
                continue
            frames, actions = data
            seq_len = len(actions)
            ####
            # TODO: to support general softmax detection
            ####
            accu_label.append(label_index[info['target_room']])
            label_stats[info['target_room']] += 1
            if stack_frame:
                if seq_len < stack_frame:
                    cur_frame = np.zeros((stack_frame, ) + frames.shape[1:], dtype=np.uint8)
                    cur_frame[-stack_frame:, ...] = frames[:seq_len, ...]
                    accu_data.append(cur_frame)
                else:
                    accu_data.append(frames[seq_len-stack_frame:])
            else:
                accu_data.append(frames[-1])
            n_pos_samples += 1
            neg_samples_need += neg_rate
            if seq_len < 3: continue
            rnd_range = seq_len - 2
            for i in range(neg_samples_need):
                j = random.randint(0, rnd_range)
                accu_label.append(label_index['NA'])
                if stack_frame:
                    if j+1 >= stack_frame:
                        accu_data.append(frames[j-stack_frame+1:j])
                    else:
                        cur_frame = np.zeros((stack_frame, ) + frames.shape[1:], dtype=np.uint8)
                        cur_frame[-j-1:, ...] = frames[:j, ...]
                        accu_data.append(cur_frame)
                else:
                    accu_data.append(frames[j])
                n_neg_samples += 1
                label_stats['NA'] += 1
            neg_samples_need = 0
        if frame_shape is None:
            frame_shape = data[0][0].shape
        myprint('    ----> partition%d: Done!' % p)
    assert frame_shape is not None

    total_samples = len(accu_label)
    assert total_samples > 0, '[ERROR] No data found!'

    myprint('[Data_Loader] Positive Samples = %d, Negative Samples = %d, Total Sample = %d'%(n_pos_samples, n_neg_samples, total_samples))
    np_data = np.array(accu_data)
    np_label = np.array(accu_label, dtype=np.int32)
    for l in label_names:
        n = label_stats[l]
        myprint('  >>> Label <%s>: # = %d (%.4f)' % (l, n, n / total_samples))
    return np_data, np_label, {'labels': label_names, 'shape': frame_shape, 'n_class': n_class, 'n_samples': total_samples}


def eval_model(test_data, test_size, batch_size, trainer, logger, calc_label_stats=False):
    # test_data: lis_frames, np_length, lis_actions, np_target, lis_mask_feat, t_max
    global test_batch_frames, test_batch_labels
    dur = time.time()
    test_epochs = test_size // batch_size
    report_rate = 1.0 / min(4, test_epochs)
    accu_report_rate = report_rate

    label_correct = None
    label_samples = None
    if calc_label_stats:
        labels = test_data[-1]['labels']
        label_correct = dict()
        label_samples = dict()
        for i in range(len(labels)):
            label_correct[i] = 0
            label_samples[i] = 0

    # Set to eval mode
    trainer.eval()

    # Evaluation
    logger.print('+++++++++++++ Evaluation +++++++++++++')
    total_correct = 0
    total_samples = 0
    for ep in range(test_epochs):
        L, R = ep * batch_size, (ep+1) * batch_size
        cur_indices = list(range(L, R))
        total_samples += batch_size
        test_batch_frames[...] = test_data[0][cur_indices]
        test_batch_labels[...] = test_data[1][cur_indices]
        
        # run forward pass
        predict = trainer.action(test_batch_frames, return_numpy=True, greedy_act=True, return_argmax=True)  # [batch]

        # accuracy
        total_correct += np.sum(predict == test_batch_labels)
        if calc_label_stats:
            for i in range(batch_size):
                l = test_batch_labels[i]
                label_samples[l] += 1
                if l == predict[i]:
                    label_correct[l] += 1

        # output log
        cur_rate = (ep + 1) / test_epochs
        if cur_rate > accu_report_rate:
            accu_report_rate += report_rate
            logger.print(' -->> Eval Samples = %d, Percent = %.3f (Elapsed = %.3fs): Accuracy = %.4f' % 
                        (total_samples, cur_rate, time.time() - dur, total_correct / total_samples))
    logger.print('>> DONE! Time Elapsed = %.3f' % (time.time() - dur))
    logger.print(' > Total Samples = %d, Correct = %d, Accuracy = %.4f' % (total_samples, total_correct, total_correct / total_samples))
    if calc_label_stats:
        for i, l in enumerate(labels):
            logger.print('  --> Label<%s>: correct = %d / %d, percent = %.4f' % (l, label_correct[i], label_samples[i], label_correct[i] / label_samples[i]))
    logger.print('++++++++++++++++++++++++++++++++++++++')
    return total_correct / total_samples


def train(args=None, warmstart=None):
    # Process Observation Shape
    common.process_observation_shape(model='cnn',
                                     resolution_level=args['resolution_level'],
                                     segmentation_input=args['segment_input'],
                                     depth_input=args['depth_input'],
                                     history_frame_len=1)

    args['logger'] = logger = utils.MyLogger(args['log_dir'], True, keep_file_handler=not args['append_file'])

    tstart = time.time()
    logger.print('Data Loading ...')

    batch_size = args['batch_size']

    ############################
    # Training Data
    train_data = data_loader(args['data_dir'], args['n_part'], fixed_target=args['fixed_target'], 
                             logger=logger, neg_rate=args['neg_rate'], stack_frame=args['stack_frame'])
    train_size = train_data[-1]['n_samples']
    # cache training batch memory
    global batch_frames, batch_labels
    batch_frames = np.zeros((batch_size, ) + train_data[0][0].shape, dtype=np.uint8)
    batch_labels = np.zeros((batch_size, ), dtype=np.int32)
    ############################
    # Training Data
    test_data = None
    test_size = 0
    if args['eval_dir'] and args['eval_n_part']:
        test_data = data_loader(args['eval_dir'], args['eval_n_part'], fixed_target=args['fixed_target'],
                                logger=logger, neg_rate=1, stack_frame=args['stack_frame'])
        test_size = test_data[-1]['n_samples']
        test_batch_size = args['eval_batch_size']
        global test_batch_frames, test_batch_labels
        test_batch_frames = np.zeros((test_batch_size, ) + train_data[0][0].shape, dtype=np.uint8)
        test_batch_labels = np.zeros((test_batch_size, ), dtype=np.int32)

    logger.print(' --> Loading Finished! Elapsed = %.4fs' % (time.time() - tstart))

    if args['only_load_data']:
        if args['data_dump_dir'] is not None:
            filename = os.path.join(args['data_dump_dir'], "dump_data.pkl")
            print('>>>> Dumping data to <{}>...'.format(filename))
            with open(filename, "wb") as f:
                pickle.dump([train_data, test_data], f)
        return 

    # create trainer
    trainer = create_trainer(args, n_class=test_data[-1]['n_class'])
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

        train_stats = []
        test_stats = []
        train_accuracy = 0
        best_eval_rate = 0
        beg_time = time.time()
        for ep in range(args['epochs']):
            dur = time.time()
            # set to train mode
            trainer.train()
            # start training current epoch
            logger.print('Training Epoch#{}/{} ....'.format(ep+1, args['epochs']))
            random.shuffle(indices)
            prev_ep_dur = time.time() - beg_time
            for up in range(epoch_updates):
                L, R = up * batch_size, (up + 1) * batch_size
                if R <= train_size:
                    cur_indices = indices[L: R]
                else:
                    cur_indices = indices[L: ] + indices[: R - train_size]
                # lis_frames, np_length, lis_actions, np_target, lis_mask_feat, t_max = data
                batch_frames[...] = train_data[0][cur_indices]
                batch_labels[...] = train_data[1][cur_indices]

                stats = \
                trainer.update(batch_frames, batch_labels)
                train_accuracy += stats['accuracy']
                if (up + 1) % args['report_rate'] == 0:
                    ep_dur = time.time() - dur
                    total_dur = time.time() - tstart
                    avg_up_dur = ep_dur / (up + 1)
                    avg_ep_dur = (prev_ep_dur + avg_up_dur * epoch_updates) / (ep + 1)
                    logger.print('--> Epoch#%d / %d, Update#%d / %d, Percent = %.4f, Total Elapsed = %.4fs, Epoch Elapsed = %.4fs (Avg: %.4fs)' %
                        (ep + 1, args['epochs'], up + 1, epoch_updates, (up + 1) / epoch_updates, total_dur, ep_dur, avg_ep_dur))
                    for k in sorted(stats.keys()):
                        logger.print('   >> %s = %.4f' % (k, stats[k]))
                    if args['keep_stats']:
                        stats['epoch'] = ep
                        stats['updates'] = up
                        train_stats.append(stats)
            # Epoch Finished
            train_accuracy /= epoch_updates
            logger.print('>> Epoch#%d Done!!!! Train Accuracy = %.4f'%(ep + 1, train_accuracy))
            if (test_data is not None) and (args['eval_rate'] is not None) and ((ep + 1) % args['eval_rate'] == 0):
                accu = eval_model(test_data, test_size, args['eval_batch_size'], trainer, logger)
                if accu > best_eval_rate:
                    best_eval_rate = accu
                    trainer.save(args['save_dir'], 'best')
                    logger.print(' ------> Best Model Saved! Best Accu = {}'.format(accu))
                if args['keep_stats']:
                    test_stats.append((ep, accu))
            if (ep + 1) % args['save_rate'] == 0:
                trainer.save(args['save_dir'])

        logger.print('Done!')
        trainer.save(args['save_dir'], version='final')
        if args['keep_stats']:
            with open(os.path.join(args['log_dir'], 'train_stats.pkl'), 'wb') as f:
                pickle.dump([train_stats, test_stats], f)
    except KeyboardInterrupt:
        trainer.save(args['save_dir'], version='interrupt')
        if args['keep_stats']:
            with open(os.path.join(args['log_dir'], 'train_stats.pkl'), 'wb') as f:
                pickle.dump([train_stats, test_stats], f)
        raise


def parse_args():
    parser = argparse.ArgumentParser("Supervised Learning for 3D House Navigation")
    ########################################################
    # Data Setting
    parser.add_argument("--data-dir", type=str,
                        help="the directory containing data partitions")
    parser.add_argument("--n-part", type=int,
                        help="number of partitions")
    parser.add_argument("--neg-rate", type=int, default=1,
                        help="negative sample per positive sample")
    parser.add_argument("--only-data-loading", dest="only_load_data", action="store_true",
                        help="When true, only loading data. No training will be performed");
    parser.set_defaults(only_load_data=False)
    parser.add_argument("--data-dump-dir", type=str,
                        help="Only Effect when --only-data-loading")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--stack-frame", type=int, help="When set, will stack frames for a joint prediction")
    parser.add_argument("--self-attention-dim", type=int,
                        help="When set, classifier will use self attention when --stack-frame > 1")
    ########################################################
    # Environment Setting
    parser.add_argument("--segmentation-input", choices=['none', 'index', 'color', 'joint'], default='none', dest='segment_input',
                        help="whether to use segmentation mask as input; default=none; <joint>: use both pixel input and color segment input")
    parser.add_argument("--depth-input", dest='depth_input', action='store_true',
                        help="whether to include depth information as part of the input signal")
    parser.set_defaults(depth_input=False)
    parser.add_argument("--resolution", choices=['normal', 'low', 'tiny', 'high', 'square', 'square_low'],
                        dest='resolution_level', default='normal',
                        help="resolution of visual input, default normal=[120 * 90]")
    parser.add_argument("--include-object-target", dest='object_target', action='store_true',
                        help="when this flag is set, target can be also an object. Only effective when --multi-target")
    parser.set_defaults(object_target=False)
    parser.add_argument("--fixed-target", type=str, default='kitchen', help="fixed training target: candidate values room, object or any-room/object")
    ########################################################
    # sup training parameters
    parser.add_argument("--train-gpu", type=str,
                        help="[SUP] an integer or a ','-split list of integers, indicating the gpu-id for training")
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
    parser.set_defaults(batch_norm=False)
    parser.add_argument("--entropy-penalty", type=float, help="policy entropy regularizer")
    parser.add_argument("--logits-penalty", type=float, help="policy logits regularizer")
    parser.add_argument("--optimizer", choices=['adam', 'rmsprop'], default='adam', help="optimizer")

    ###################################################
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./_model_/supervised", help="directory in which training state and model should be saved")
    parser.add_argument("--log-dir", type=str, default="./log/supervised", help="directory in which logs training stats")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every epochs completed")
    parser.add_argument("--report-rate", type=int, default=1,
                        help="report training stats once every time this many training steps are performed")
    parser.add_argument("--warmstart", type=str, help="model to recover from. can be either a directory or a file.")
    parser.add_argument("--keep-stats", dest="keep_stats", action="store_true")
    parser.set_defaults(keep_stats=False)

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
    parser.add_argument("--eval-batch-size", type=int,
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
    
    if (cmd_args.stack_frame is None) or (cmd_args.stack_frame <= 1):
        cmd_args.stack_frame = None
        cmd_args.self_attention_dim = None

    args = cmd_args.__dict__

    args['model_name'] = 'cnn'

    if args['eval_rate'] is not None:
        if args['eval_batch_size'] is None:
            args['eval_batch_size'] = args['batch_size']

    # gpu devices
    all_gpus = common.get_gpus_for_rendering()
    assert (len(all_gpus) > 0), 'No GPU found! There must be at least 1 GPU.'
    if args['train_gpu'] is None:
        print('Warning: training gpu not specified. Set to the default gpu <{}> (single-gpu)'.format(all_gpus[0]))
        args['train_gpu'] = all_gpus[0]
    else:
        gpu_sids = args['train_gpu'].split(',')
        train_gpus = [int(k) for k in gpu_sids]
        if len(train_gpus) > 1:
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
