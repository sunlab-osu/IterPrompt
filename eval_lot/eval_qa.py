"""
Used for evaluating the prediction file. Here the file should start with 3 non-content lines, then each following
3 lines are input, \t output, \t target
"""
import json
import argparse
import numpy as np
import sys
sys.path.append('..')
from utils import strip_string, mask_sublist

parser = argparse.ArgumentParser()
parser.add_argument("--predict_file", default=None, type=str, required=True, help="predict file")
parser.add_argument("--eval_type", default=None, type=str, required=True, help="qa or qc")
parser.add_argument("--eval_split", default='test', type=str, required=False, help="train/valid/test")
parser.add_argument("--single", action="store_true", help="whether non-iter setting (for qc)")

args = parser.parse_args()

f = open(args.predict_file, 'r', encoding='utf-8')
data = f.readlines()[3:]
f.close()
assert len(data) % 3 == 0
num_samples = len(data) // 3


if args.eval_type == "qa":

    EM_list = []
    for i in range(num_samples):
        output = data[3*i+1].strip().replace("<special_sep>", "")
        target = data[3*i+2].strip()
        assert target in ['no', 'yes']
        if output.startswith(target):
            EM = 1
        else:
            EM = 0
        EM_list.append(EM)

    print("In QA eval mode. Total {} samples. EM:{}.".format(num_samples, np.mean(EM_list)))

elif args.eval_type == 'qc':

    single = arg.single

    wrong = 0
    for i in range(num_samples):
        question = data[3 * i].strip()
        output, target = data[3 * i + 1].strip(), data[3 * i + 2].strip()

        if single:
            rua = strip_string(output, "<special_sep>").split("; ")
        else:
            rua = output.split("<special_sep>")[:2]

        if len(rua) != 2:
            wrong += 1
            continue
        rule, prop = rua

        if 'not' in rule:
            wrong += 1
            continue
        if 'is the opposite of' in question:
            if target == 'yes' and ('not' not in prop):
                wrong += 1
            elif target == 'no' and ('not' in prop):
                wrong += 1
        else:
            if target == 'yes' and 'not' in prop:
                wrong += 1
            elif target == 'no' and ('not' not in prop):
                wrong += 1
    print("EM:", (num_samples - wrong) / num_samples, num_samples)

    count_match, count_ratio = 0, 0

    path = "../dataset_lot/RD_Oracle/" + args.eval_split + ".source"
    f = open(path, "r", encoding='utf-8')
    true_contexts = f.readlines()
    f.close()
    assert len(true_contexts) == num_samples
    for i in range(num_samples):
        question = data[3*i].strip()
        output = data[3 * i + 1].strip()
        if single:
            output = strip_string(output, "<special_sep>").split("; ")
        else:
            output = output.split("<special_sep>")[:2]

        if len(output) != 2:
            count_ratio += 0.0
            count_match += 0.0
            continue

        target = true_contexts[i].strip().split("</s></s>")[-2:]
        for j in range(len(output)):
            output[j] = output[j].strip()
        for j in range(len(target)):
            target[j] = target[j].strip()
        if target[0].lower() in output[0].lower() and output[1].lower() == target[1].lower():
            count_match += 1
        if target[0].lower() in output[0].lower():
            count_ratio += 1
        if output[1].lower() == target[1].lower():
            count_ratio += 1

    print("context match:", count_match/num_samples, num_samples)
    print("context ratio:", count_ratio/2/num_samples, num_samples)
