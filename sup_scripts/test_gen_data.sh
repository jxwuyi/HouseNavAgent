#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python3 data_gen.py --seed 10 --env-set small \
    --n-house 2 --n-partition 2 --n-proc 2 \
    --sample-size 20 --t-max -1 \
    --include-object-target --fixed-target any-room \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --segmentation-input color --depth-input --resolution normal \
    --success-measure see-stop \
    --multi-target --no-outdoor-target \
    --save-dir ./_sup_data_/test --log-rate 10 \
    --max-expansion 200000 \
    --sanity-check
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature

