#!/bin/bash
#SBATCH --account=<->
#SBATCH --job-name=<->
#SBATCH --time=<->
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --output=<->

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### WORLD_SIZE = gpus/node * num_nodes
export MASTER_PORT=12121
export WORLD_SIZE=6

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

echo "Job started..."

cd ..

MODEL_TYPE=seq2seq
MODEL_PATH=facebook/bart-large

DATA_PATH=dataset_lot/KE/
OUTPUT_DIR=<path to PLM>

srun python main.py \
--data_dir $DATA_PATH \
--model_type $MODEL_TYPE \
--model_name_or_path $MODEL_PATH \
--output_dir $OUTPUT_DIR \
--do_train \
--max_seq_length 128 \
--max_length 128 \
--train_batch_size 8 \
--learning_rate 8e-5 \
--gradient_accumulation_steps 4 \
--save_step 10000 \
--overwrite_output_dir \
--num_train_epochs 150 \
--evaluation_metric passage

echo "Job finished!"