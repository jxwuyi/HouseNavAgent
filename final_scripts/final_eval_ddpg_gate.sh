#
# ddpg-vis-gate
CUDA_VISIBLE_DEVICES=2,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating --use-action-gating \
    --algo ddpg_joint --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input --history-frame-len 5 --batch-norm \
    --log-dir ./log/eval/final_full/ddpg_vis_gate --warmstart ./_model_/cluster/full/ddpg_visual_gate_coef10/JointDDPG_JointDDPGTrainer.pkl

#
# ddpg-segment-gate
CUDA_VISIBLE_DEVICES=2,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating --use-action-gating \
    --algo ddpg_joint --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input --history-frame-len 5 --batch-norm \
    --log-dir ./log/eval/final_full/ddpg_seg_gate --warmstart ./_model_/cluster/full/ddpg_segcolor_gate/JointDDPG_JointDDPGTrainer.pkl


#
# ddpg-vis-gate coef100
CUDA_VISIBLE_DEVICES=2,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target --use-target-gating --use-action-gating \
    --algo ddpg_joint --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input --history-frame-len 5 --batch-norm \
    --log-dir ./log/eval/final_full/ddpg_vis_gate_coef100 --warmstart ./_model_/cluster/full/ddpg_visual_gate/JointDDPG_JointDDPGTrainer.pkl


