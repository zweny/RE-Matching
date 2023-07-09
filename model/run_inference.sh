# ckpt_name='[dataset]__split_[rel_split_seed]_unseen_[unseen]_f1_[f1_score]' in default setting
CUDA_VISIBLE_DEVICES=0 python -u inference.py --visible_device '0' \
                       --ckpt_name 'ckpt_name'


