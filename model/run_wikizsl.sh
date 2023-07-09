echo 'train REMatching with wikizsl dataset'
# unseen=(15 10 5): testing relation nums
# dataset=(fewrel wikizsl): dataset
# relation_description=(relation_description_addsynonym_processed.json relation_description_processed.json)
# rel_split_seed=(40 41 42 43 44): relation_split_with_random_seed


python -u train.py   --visible_device '0' \
                         --unseen 10 \
                         --rel_split_seed '40' \
                         --dataset 'wikizsl' \
                         --relation_description 'relation_description_addsynonym_processed.json' \






