#!/bin/bash
# a3c-seg-nog
CUDA_VISIBLE_DEVICES=0,2 python3 interact_semantic.py --seed 9 --env-set test --house -10 \
    --hardness 0.6 --max-birthplace-steps 15 --min-birthplace-grids 1 \
    --success-measure see-stop --multi-target \
    --fixed-target any-room \
    --include-object-target \
    --segmentation-input color --depth-input --target-mask-input \
    --semantic-gpu 0 \
    --semantic-threshold 0.9 \
    --semantic-filter-steps 3 \
    --render-gpu 0 \
    --log-dir ./log/interact \
    --semantic-dir ./_model_/semantic/_dict_/semantic_oracle_rooms.json \
    #--semantic-dir ./_model_/semantic/frame4_att32

