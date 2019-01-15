#!/bin/bash

python3 fetch_eval_details.py --data /home/jxwuyi/backup/HouseNavAgent/results/obj_nav/old_interrupt_g_300_m_30_new/mixture_full_eval_history.pkl --log-dir ./ --filename _eval_details_obj_nav.pkl --render-gpu 2

#python3 fetch_eval_details.py --data ~/backup/HouseNavAgent/results/pure_policy/room_nav/birth_away_maxbth_40_eplen_30/mixture_full_eval_history.pkl --log-dir ./ --render-gpu 2

#python3 fetch_eval_details.py --data ./results/reproduce/pure_policy/birth_away_maxbth_40_eplen_300/mixture_full_eval_history.pkl --log-dir ./ --filename _eval_details_large_seed7.pkl --render-gpu 0

#required="4:253,5:436"

#python3 fetch_eval_details.py --data ./results/additional/HRL/g_300_m_30_term_mask/mixture_full_eval_history.pkl \
#    --log-dir ./ --filename _eval_details_additional.pkl --render-gpu 0 --plan-dist-iters $required

#python3 fetch_eval_details.py --data ./results/additional/HRL/new_g_300_m_30_term_mask/mixture_full_eval_history.pkl \
#    --log-dir ./ --filename _eval_details_additional_new.pkl --render-gpu 0 --plan-dist-iters $required
