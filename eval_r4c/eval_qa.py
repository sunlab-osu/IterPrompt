"""
Used for evaluating the prediction file. Here the file should start with 3 non-content lines, then each following
3 lines are input, \t output, \t target
"""
import json
import argparse
import numpy as np
import sys
from datasets import load_metric

sys.path.append('..')
from utils import strip_string

parser = argparse.ArgumentParser()
parser.add_argument("--predict_file", default=None, type=str, required=True, help="predict file")
parser.add_argument("--eval_type", default=None, type=str, required=True, help="qa or qc")
parser.add_argument("--eval_split", default='test', type=str, required=False, help="")

args = parser.parse_args()

f = open(args.predict_file, 'r', encoding='utf-8')
data = f.readlines()[3:]
f.close()
assert len(data) % 3 == 0
num_samples = len(data) // 3

def tokenize(a):
    """
    lower, split, strip each token
    """
    b = a.lower().split()
    for ii in range(len(b)):
        b[ii] = b[ii].strip().strip('?.,\"\'').strip()
    return b

if args.eval_type == "qa":

    EM_list, f1_list = [], []
    Relaxed_EM_list = []

    for i in range(num_samples):
        question = data[3*i].strip().split("</s></s>")[0].strip()
        output = data[3*i+1].strip().replace("<special_sep>", "")
        target = data[3*i+2].strip()

        # EM
        if target.lower() == output.lower():
            EM = 1
        else:
            EM = 0
        EM_list.append(EM)

        # F1
        output_w = set(tokenize(output))
        target_w = set(tokenize(target))
        num_share_w = len(output_w & target_w)
        if num_share_w == 0:
            f1 = 0
        else:
            precision = num_share_w / len(output_w)
            recall = num_share_w / len(target_w)
            f1 = 2 * precision * recall / (precision + recall)
        f1_list.append(f1)

    print("In QA eval mode. Total {} samples. EM:{}, F1:{}.".format(
        num_samples, np.mean(EM_list), np.mean(f1_list)))


elif args.eval_type == 'qc':
    # here only see if the correct answer is inside the output. Two cases: filter if the answer exists in input
    count, count_contain = 0, 0
    for i in range(num_samples):
        question = data[3 * i].strip()
        output = data[3 * i + 1].strip()
        target = data[3 * i + 2].strip()
        if target.lower() in ['yes', 'no']:
            continue
        if target.lower() in question.lower():
            continue
        count += 1
        if target.lower() in output.lower():
            count_contain += 1
    print("Ans.R^hat:", count, count_contain, count_contain / count)

    count, count_contain = 0, 0
    for i in range(num_samples):
        question = data[3 * i].strip()
        output = data[3 * i + 1].strip()
        target = data[3 * i + 2].strip()
        if target.lower() in ['yes', 'no']:
            continue
        count += 1
        if target.lower() in output.lower():
            count_contain += 1

    print("Ans.R:", count, count_contain, count_contain / count)
