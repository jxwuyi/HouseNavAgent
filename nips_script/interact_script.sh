# a3c-seg-nog
python3 interact.py --seed 0 --env-set test --house -3 --hardness 0.6 \
    --success-measure see --multi-target --use-target-gating \
    --include-object-target --only-eval-room-target \
    --algo a3c --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input \
    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm --batch-norm \
    --log-dir ./log/interact \
    --warmstart ./_model_/eval/large/a3c_obj_seg_gate/ZMQA3CTrainer_tmp_cpu.pkl
