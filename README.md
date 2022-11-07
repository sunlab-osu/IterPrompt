# Iteratively Prompt Pre-trained Language Models for Chain of Thought

Original implementation of the paper "[Iteratively Prompt Pre-trained Language Models for Chain of Thought](https://arxiv.org/abs/2203.08383)" in EMNLP-22 by [Boshi Wang](https://boshi-wang.github.io/), [Xiang Deng](https://xiang-deng.github.io/) and [Huan Sun](http://web.cse.ohio-state.edu/~sun.397/).

## Environment Setup
First have python >= 3.8 installed, e.g., 
```bash
conda create -n <YOUR_ENV_NAME> python=3.8
```
Install dependencies via:
```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
cd transformers
pip install -e .
cd ..
pip install -e .
pip install -r requirements-dev.txt
```

## Repo Tour
    .
    ├── dataset_*/                     # preprocessed datasets
        ├── QA/                        # query (q) -> answer (a), for PLM-QA
        ├── IP/                        # [q; c_1; ...; c_{j-1}] -> c_j, for iterative prompting
        ├── IP_single/                 # q -> [c_1; ...; c_{n_q}], for non-iterative prompting
        ├── RD_Oracle/                 # [q; c_1; ...; c_{n_q}] -> a, for oracle reader
        ├── KE/                        # c_j(masked) -> c_j, for PLM knowledge enhancement
    ├── job_*/                         # commands for training
    ├── eval_*/                        # commands for evaluating
    ├── simpletransformers/seq2seq/    # main implementation of iCAP; with necessary modifications in transformers/
    ├── ...
    ├── utils.py                       # helper functions
    ├── soft_embedding.py              # soft embedding for virtual prompt tokens
    └── main.py                        # main script for training/evaluating

Our main code frame borrows from [this repo](https://github.com/wangcunxiang/Can-PLM-Serve-as-KB-for-CBQA) and the soft embedding module is adapted from [this implementation](https://github.com/kipgparker/soft-prompt-tuning).

## Usage
Our scripts are run on a cluster with SLURM scheduler. Remember to replace the ```<...>``` parts according to your preferences. You can also change the ```train_batch_size, gradient_accumulation_steps``` args according to your GPU memory. Use ```bash``` instead of ```sbatch``` to run on regular servers. The following commands are for 2wiki experiments; the other datasets are similar.
```bash
cd job_2wiki
```
### Knowledge Enhancement
```bash
sbatch KE-train.sh
```
Or alternatively, download the trained model checkpoints for [2wiki](https://drive.google.com/file/d/1hHLoXzQfhhIIkWGz234Rhv8Jpm90udWg/view?usp=sharing), [lot](https://drive.google.com/file/d/1DgaPkHpOu9EPFJH_ADZZ5olQLpLqxb7A/view?usp=sharing), [r4c](https://drive.google.com/file/d/1agIuQX_R9RiDhQYshRCgGN0L6eelF76j/view?usp=sharing).

### Training
```sbatch ${METHOD}-train.sh```

where METHOD is:
- iCAP: proposed iterative context-aware prompter
- iCAP_stopper: iCAP with stopper module
- PromptT: Prompt-Tuning
- PromptT_iter: Prompt-Tuning (iter)
- PLMFT: PLM fine-tuning
- PLMFT_iter: PLM fine-tuning (iter)
- PLMQA: fine-tuning PLM on (Q,A) directly
- RD_Oracle: Oracle_Reader

We used [this](https://github.com/jxhe/unify-parameter-efficient-tuning) implementation for Prefix-Tuning.

### Evaluation
```cd eval_2wiki```
#### Intrinsic Evaluation
run 
```python gen_eval_script_${METHOD}.py``` to generate the scripts for running predictions and evaluation. Then run
```bash
bash run_pred_all_${SAVE_PATH}.sh
```
to get predictions, and
```bash
bash eval_on_{valid/test}_all_${SAVE_PATH}.sh
```
to evaluate the predictions.

#### Extrinsic Evaluation
First prepare a dataset using the predicted contexts; this could be done using the script ```prep_reader.py``` by, e.g.,
```bash
python prep_reader.py --path dataset_2wiki_0.1/iCAP_RD/ --train <prediction file on train> --valid <prediction file on valid> --test <prediction file on test>
```
Then fine-tune the trained oracle reader on this dataset, and the results could be evaluated by setting ```--eval_type qa``` in ```eval_qa.py```.

## Citation
```
@inproceedings{wang2022iterative,
    title={Iteratively Prompt Pre-trained Language Models for Chain of Thought},
    author={Wang, Boshi and Deng, Xiang and Sun, Huan},
    booktitle={EMNLP},
    year={2022}
}
```