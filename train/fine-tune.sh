#!/bin/bash
#SBATCH --gpus=4
export PYTHONPATH=.

module load miniforge3/24.1
source activate conf_kno
module load cudnn/8.6.0.163_cuda11.x  
#cudnn/8.8.1.3_cuda11.x
#maybe需要把“--deepspeed conf.json改为--deepspeed_conf conf.json”。

# X_LOG_DIR="log-${SLURM_JOB_ID}"
# X_GPU_LOG="${X_LOG_DIR}/gpu.log"
# mkdir "${X_LOG_DIR}"
# function gpus_collection(){
#     sleep 15
#     process='ps -ef | grep python | grep $USER | grep -v "grep" | wc -l'
#     while [[ "${process}">"0" ]]; do
#         sleep 1
#         nvidia-smi >>"${X_GPU_LOG}"2>&1
#         echo "process num:${process}">>"${X_GPU_LOG}" 2>&1
#         process='ps -ef | grep python | grep $USER | grep -v "grep" | wc-l '
#     done
# }
# gpus_collection &

echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed  --num_gpus=4 \
    fine-tune.py \
        --model_name_or_path /home/bingxing2/home/scx7002/model/Meta-Llama-2-7B-hf \
        --train_file /home/bingxing2/home/scx7002/xinyewang/Extension-of-Conflict-Knowledge-Learning/data/output_text/text_train_bio_data_source_name.json \
        --output_dir /home/bingxing2/home/scx7002/xinyewang/Extension-of-Conflict-Knowledge-Learning/models/trained_models \
        --max_length 128 \
        --train_batch_size 8 \
        --num_train_epochs 8 \
        --learning_rate 1e-5 \
        --weight_decay 0.01 \
        --deepspeed_conf /home/bingxing2/home/scx7002/xinyewang/Extension-of-Conflict-Knowledge-Learning/train/conf.json