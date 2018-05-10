#!/bin/bash

all_rooms="kitchen living_room dining_room bedroom garage bathroom office outdoor"

# gen model args for room policies
for obs_signal in none color
do
    save_pref="./_model_/nips_HRL/seg_large_birth15/"
    if [ $obs_signal == "none" ]
    then
        save_pref="./_model_/nips_HRL/visual_large_birth15/"
    fi
    for room in $all_rooms
    do
        save_dir=$save_pref$room
        #kdir $save_dir
        outdoor_flag="--no-outdoor-target"
        if [ $room == "outdoor"  ]
        then
            outdoor_flag=""
        fi
        echo $save_dir
        python3 zmq_train.py --seed 0 --env-set train --rew-clip 3 \
                --n-house 200 --n-proc 200 --batch-size 64 --t-max 30 --grad-batch 1 \
                --max-episode-len 60 $outdoor_flag \
                --hardness 0.95 --max-birthplace-steps 15 --min-birthplace-grids 2 \
                --reward-type new --success-measure see \
                --multi-target --use-target-gating --fixed-target $room \
                --segmentation-input $obs_signal --depth-input --resolution normal \
                --render-gpu 1,2,3,4,5 --max-iters 100000 \
                --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.96 --batch-norm \
                --entropy-penalty 0.1 --logits-penalty 0.0001 --q-loss-coef 1.0 --grad-clip 1.0 --adv-norm \
                --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
                --report-rate 20 --save-rate 1000 --eval-rate 300000 \
                --curriculum-schedule 5,5,20000 \
                --save-dir $save_dir \
                --log-dir $save_dir \
                --warmstart $save_dir"/ZMQA3CTrainer.pkl" \
                --only-fetch-model-dict
    done
    # gen any-object
    for birth_size in small large
    do
        birth="15"
        if [ $birth_size == "small" ]
        then
            birth="10"
        fi
        echo $birth
        for use_mask in yes no
        do
            prefix="./_model_/nips_HRL/seg"
            if [ $obs_signal == "none" ]
            then
                prefix="./_model_/nips_HRL/visual"
            fi
            flag_mask=""
            batch_size="64"
            if [ $use_mask == "no" ]
            then
                flag_mask="--target-mask-input"
                batch_size="50"
                prefix=$prefix"_mask"
            fi
            save_dir=$prefix"_large_birth"$birth"/any-object"
            flag_curr="--curriculum-schedule 5,5,20000"
            ep_len="60"
            if [ $birth == "10" ]
            then
                flag_curr="--curriculum-schedule 5,5,35000"
                ep_len="50"
            fi
            echo $save_dir
            python3 zmq_train.py --seed 0 --env-set train --rew-clip 3 \
                    --n-house 200 --n-proc 200 --batch-size $batch_size --t-max 30 --grad-batch 1 \
                    --max-episode-len $ep_len --no-outdoor-target \
                    --hardness 0.95 --max-birthplace-steps $birth --min-birthplace-grids 2 \
                    --reward-type new --success-measure see --include-object-target --fixed-target any-object \
                    --multi-target --use-target-gating \
                    --segmentation-input $obs_signal --depth-input --resolution normal $flag_mask \
                    --render-gpu 1,2,3,4,5 --max-iters 100000 \
                    --algo a3c --lrate 0.001 --weight-decay 0.00001 --gamma 0.96 --batch-norm \
                    --entropy-penalty 0.1 --logits-penalty 0.0001 --q-loss-coef 1.0 --grad-clip 1.0 --adv-norm \
                    --rnn-units 256 --rnn-layers 1 --rnn-cell lstm \
                    --report-rate 20 --save-rate 1000 --eval-rate 300000 $flag_curr \
                    --save-dir $save_dir \
                    --log-dir $save_dir \
                    --warmstart $save_dir"/ZMQA3CTrainer.pkl" \
                    --only-fetch-model-dict

        done
    done
done
