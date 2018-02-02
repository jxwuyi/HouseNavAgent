#
# ddpg-vis-no-gate (coef100)
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo ddpg_joint --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input --history-frame-len 5 --batch-norm \
    --log-dir ./log/eval/final_full/ddpg_vis_nog --warmstart ./_model_/cluster/full/ddpg_visual_nog/JointDDPG_JointDDPGTrainer.pkl

#
# ddpg-segment-no-gate
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo ddpg_joint --max-episode-len 100 --max-iters 5000 \
    --segmentation-input color --depth-input --history-frame-len 5 --batch-norm \
    --log-dir ./log/eval/final_full/ddpg_seg_nog --warmstart ./_model_/cluster/full/ddpg_segcolor_nog/JointDDPG_JointDDPGTrainer.pkl


#
# ddpg-vis-no-gate coef10
CUDA_VISIBLE_DEVICES=0,1 python3 eval.py --seed 0 --env-set train --house -200 --hardness 0.95 \
    --render-gpu 1 --success-measure see --multi-target \
    --algo ddpg_joint --max-episode-len 100 --max-iters 5000 \
    --segmentation-input none --depth-input --history-frame-len 5 --batch-norm \
    --log-dir ./log/eval/final_full/ddpg_vis_nog_coef10 --warmstart ./_model_/cluster/full/ddpg_visual_nog_coef10/JointDDPG_JointDDPGTrainer.pkl


