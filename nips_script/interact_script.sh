# a3c-seg-nog
python3 interact.py --seed 0 --env-set test --house -10 \
    --hardness 0.6 --max-birthplace-steps 20 \
    --success-measure see --multi-target --use-target-gating \
    --include-object-target --eval-target-type only-object \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/interact \
    --warmstart ./_model_/nips/large/birth20_seg/a3c_bc64_gate_tmax30/ZMQA3CTrainer_succ.pkl
