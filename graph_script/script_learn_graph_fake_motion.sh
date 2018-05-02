#!/usr/bin/env bash

python3 learn_graph.py --task-name roomnav --env-set train --house 200 --seed 0 \
    --render-gpu 0 \
    --segmentation-input none --depth-input \
    --success-measure see --multi-target --include-object-target \
    --training-mode mle \
    --graph-eps 0.0001 --n-trials 25 --max-eps-steps 30 \
    --motion fake --max-episode-len 200 --max-iters 5000 --store-history \
    --save-dir ./_graph_/fake
