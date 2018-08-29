#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python3 data_gen.py --seed 10 --env-set train \
    --n-house 200 --n-partition 20 --n-proc 10 \
    --sample-size 50000 --t-max 50 \
    --include-object-target --fixed-target any-room \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target --no-outdoor-target \
    --save-dir ./_sup_data_/large --log-rate 500 \
    --max-expansion 100000 \
    --sanity-check
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature

