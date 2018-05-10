#!/usr/bin/env bash

python3 HRL/eval_motion.py --task-name roomnav --env-set test \
    --house -50 --seed 0 \
    --render-gpu 7 --hardness 0.95 \
    --segmentation-input color --depth-input --target-mask-input \
    --success-measure see --multi-target --include-object-target \
    --motion rnn \
    --max-episode-len 100 --max-iters 5000 \
    --only-eval-object-target \
    --store-history \
    --log-dir ./log/graph/eval/rnn_seg_mask_object_100 \
    --warmstart ./_model_/nips_HRL/seg_mask_large_birth15/any-object/ZMQA3CTrainer.pkl\
    --warmstart-dict ./_model_/nips_HRL/seg_mask_large_birth15/any-object/train_args.json
