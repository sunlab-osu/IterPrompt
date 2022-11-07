"""
given a directory of saved models, generate evaluating scripts for all checkpoints.
"""
import os
model_path = "current folder name for <save directory>"
path = "parent folder name for <save directory>" + model_path
# first see all available checkpoints
all_checkpoint_steps = []
for path_name in os.listdir(path):
    if "checkpoint" in path_name:
        all_checkpoint_steps.append(path_name.split("-")[1])

# generate scripts for all checkpoints
script = """
#!/bin/bash
#SBATCH --account=<->
#SBATCH --job-name=<->
#SBATCH --time=<->
#SBATCH --nodes=1 --ntasks-per-node=5
#SBATCH --gpus-per-node=1
#SBATCH --output=<->

echo "Job started..."

cd ..

MODEL_TYPE=seq2seq

MODEL_PATH=<path to PLM>
PT_DIR=<parent folder name for <save directory>>/{}
DATA_PATH=dataset_lot/QA/
PREDICTION_DIR=<location for saving prediction files>/{}
MAX_SEQ_LENGTH=512
MAX_LENGTH=128
TRAIN_BATCH_SIZE=16
NUM_EN_PT=80
NUM_DE_PT=80

CUDA_VISIBLE_DEVICES=0 python main.py \
--model_type $MODEL_TYPE \
--data_dir $DATA_PATH \
--model_name_or_path $MODEL_PATH \
--pt_dir $PT_DIR \
--prediction_dir $PREDICTION_DIR \
--predict_on_eval \
--do_predict \
--max_seq_length $MAX_SEQ_LENGTH \
--max_length $MAX_LENGTH \
--train_batch_size $TRAIN_BATCH_SIZE \
--learning_rate 4e-3 \
--save_step 500 \
--num_train_epochs 50 \
--dataloader_num_workers 0 \
--overwrite_output_dir \
--gradient_accumulation_steps 8 \
--evaluation_metric qa \
--static_prompt_tuning_mode \
--use_decoder_pt \
--num_encoder_prompt_tokens $NUM_EN_PT \
--num_decoder_prompt_tokens $NUM_DE_PT

echo "Job finished!"
""".strip()

g = open("run_pred_all_{}.sh".format(model_path.strip("/")), "w", encoding='utf-8')
g2 = open("eval_on_valid_all_{}.sh".format(model_path.strip("/")), "w", encoding='utf-8')
g3 = open("eval_on_test_all_{}.sh".format(model_path.strip("/")), "w", encoding='utf-8')

for checkpoint_step in all_checkpoint_steps:
    path_suffix = "checkpoint-{}/".format(checkpoint_step)
    file_name_sh = model_path.strip("/")+"-{}.sh".format(checkpoint_step)
    to_write = script.format(file_name_sh.replace(".sh", ""), model_path+path_suffix, model_path+path_suffix)
    f1 = open(file_name_sh, "w", encoding='utf-8')
    f1.write(to_write)
    f1.close()

    g.write("sbatch "+file_name_sh)
    g.write("\n")

    g2.write('echo {}\n'.format(checkpoint_step))
    g2.write('python eval_qa.py --predict_file <location for saving prediction files>/{}predictions_valid.txt --eval_type qc'
             ' --eval_split valid --single\n'
             .format(model_path + path_suffix))

    g3.write('echo {}\n'.format(checkpoint_step))
    g3.write('python eval_qa.py --predict_file <location for saving prediction files>/{}predictions_test.txt --eval_type qc'
             ' --eval_split test --single\n'
             .format(model_path+path_suffix))


file_name_sh = model_path.strip("/")+"-{}.sh".format("")
to_write = script.format(file_name_sh.replace(".sh", ""), model_path, model_path)
f1 = open(file_name_sh, "w", encoding='utf-8')
f1.write(to_write)
f1.close()
g.write("sbatch "+file_name_sh)
g.write("\n")
g.close()

g2.write('echo -1\n')
g2.write('python eval_qa.py --predict_file <location for saving prediction files>/{}predictions_valid.txt --eval_type qc'
         ' --eval_split valid --single\n'.format(model_path))
g2.close()

g3.write('echo -1\n')
g3.write('python eval_qa.py --predict_file .<location for saving prediction files>/{}predictions_test.txt --eval_type qc'
         ' --eval_split test --single\n'.format(model_path))
g3.close()