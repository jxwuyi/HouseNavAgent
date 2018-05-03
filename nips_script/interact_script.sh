#!/bin/bash

#MODEL_DIR='./_model_/nips/large/birth15_seg_mask_curr/ZMQA3CTrainer_succ.pkl'
MODEL_DIR='./_model_/nips_new/sup_large_no_outdoor/seg_gate_bc_curr/ZMQA3CTrainer.pkl'

# a3c-seg-nog
python3 interact.py --seed 12 --env-set test --house -10 \
    --hardness 0.6 --max-birthplace-steps 15 \
    --success-measure see --multi-target --use-target-gating \
    --include-object-target --eval-target-type only-object \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input --no-outdoor-target \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/interact \
    --warmstart $MODEL_DIR \
    --graph-model ./_graph_/fake/mle_fake_graph_params.pkl --motion-steps 30
