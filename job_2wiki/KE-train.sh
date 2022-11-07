#!/bin/bash
#SBATCH --account=<->
#SBATCH --job-name=<->
#SBATCH --time=<->
#SBATCH --nodes=1 --ntasks-per-node=20
#SBATCH --gpus-per-node=1
#SBATCH --output=<->

echo "Job started..."

cd ..

MODEL_TYPE=seq2seq
MODEL_PATH=facebook/bart-large

DATA_PATH=dataset_2wiki_0.1/KE/
OUTPUT_DIR=<path to PLM>

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_dir $DATA_PATH \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length 50 \
--max_length 50 \
--train_batch_size 16 \
--learning_rate 5e-5 \
--gradient_accumulation_steps 4 \
--save_step 10000 \
--overwrite_output_dir \
--num_train_epochs 30 \
--evaluation_metric passage

echo "Job finished!"