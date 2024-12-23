#!/bin/bash
valid_precentage=0.00
train_bio_size=900
multi_num=5
test_bio_size=100

# python concat_datas.py \
#     --valid_precentage $valid_precentage\
#     --bio_size $bio_size\
#     --multi_num $multi_num

# python concat_datas_spell_error.py \
#     --valid_precentage $valid_precentage\
#     --bio_size $bio_size\
#     --multi_num $multi_num

# python concat_datas_conterfactual.py \
#     --valid_precentage $valid_precentage\
#     --bio_size $bio_size\
#     --multi_num $multi_num

# python3 concat_data_with_sat_and_score.py \
#     --valid_precentage $valid_precentage\
#     --bio_size $bio_size\
#     --multi_num $multi_num

# python3 concat_data_detail_location.py \
#     --valid_precentage $valid_precentage\
#     --bio_size $bio_size\
#     --multi_num $multi_num

python concat_data_source_names.py \
    --valid_percentage 0.00\
    --train_bio_size 900\
    --test_bio_size 100\
    --multi_num 5\
    --A_number_neutral 5\
    --B_number_neutral 5\
    --A_number 1\
    --B_number 1


# python concat_data_source_time.py\
#     --valid_percentage $valid_precentage\
#     --bio_size $bio_size\
#     --multi_num $multi_num\
#     --A_number_neutral 5\
#     --B_number_neutral 5\
#     --A_number 1\
#     --B_number 1