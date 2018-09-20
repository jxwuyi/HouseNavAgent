#!/bin/bash
#target="kitchen"
all_targets="outdoor living_room bedroom dining_room bathroom office garage"
data_dir="large"
test_data_dir="small"
train_part=10
test_part=1
stack_frame=4
att_dim=32
repo_name='pano_frame'$stack_frame'_att'$att_dim

for target in $all_targets
do

CUDA_VISIBLE_DEVICES=0 python3 semantic_train.py --seed 0 --panoramic \
    --data-dir ./_sup_data_/panoramic/$data_dir/$target --n-part $train_part \
    --segmentation-input color --depth-input --resolution normal \
    --fixed-target $target \
    --train-gpu 0 \
    --stack-frame $stack_frame --self-attention-dim $att_dim \
    --batch-size 256 --grad-batch 1 --epochs 60 \
    --lrate 0.0005 --weight-decay 0.00001 --grad-clip 5.0 \
    --optimizer adam --batch-norm \
    --save-dir ./_model_/semantic/$repo_name/$target --log-dir ./log/semantic/$repo_name/$target \
    --save-rate 1 --report-rate 20 \
    --eval-rate 1 --eval-dir ./_sup_data_/panoramic/$test_data_dir/$target --eval-n-part $test_part 
    #--batch-norm
    #--eval-batch-size 100
    #--only-data-loading
    # --render-gpu 0,1
    # --include-object-target
    # --include-mask-feature
    # --logits-penalty 0.0001

done
