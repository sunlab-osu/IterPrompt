"""
Used for evaluating the prediction file. Here the file should start with 3 non-content lines, then each following
3 lines are input, \t output, \t target
"""
import json
import argparse
import numpy as np
import sys
sys.path.append('..')
from utils import strip_string

parser = argparse.ArgumentParser()
parser.add_argument("--predict_file", default=None, type=str, required=True, help="predict file")
parser.add_argument("--eval_type", default=None, type=str, required=True, help="qa or qc")
parser.add_argument("--true_contexts", default=None, type=str, required=False, help="path to RD_Oracle source file")

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

    print("In QA eval mode. Total {} samples. EM:{}, F1:{}".format(num_samples, np.mean(EM_list), np.mean(f1_list)))

elif args.eval_type == 'qc':
    assert not (args.true_contexts is None)
    g = open(args.true_contexts, 'r', encoding='utf-8')
    true_contexts = g.readlines()
    g.close()
    assert len(true_contexts) == num_samples

    auto_stop = False   # set True if using stopper to stop knowledge recall
    if auto_stop:
        accuracy_stop = []

    count, count_contain = 0, 0
    for i in range(num_samples):
        question = data[3 * i].strip()
        output = data[3 * i + 1].strip()
        target = data[3 * i + 2].strip()

        if auto_stop:
            # clip
            output = strip_string(output, "<special_sep>").split("<special_sep>")
            len_true_context = len(true_contexts[i].strip().split("</s></s>")[1:])
            clipped = []
            for step in output:
                clipped.append(step)
                if "<stopper_stop>" in step:
                    break
            if len(clipped) == len_true_context:
                accuracy_stop.append(1)
            else:
                accuracy_stop.append(0)
            output = "<special_sep>".join(clipped)

        if target.lower() in ['yes', 'no']:
            continue
        if target.lower() in question.lower():
            continue
        count += 1
        if target.lower() in output.lower():
            count_contain += 1
    print("Ans.R^hat:", count, count_contain, count_contain / count)

    if auto_stop:
        print("accuracy stop:", np.mean(accuracy_stop), len(accuracy_stop))

    count, count_contain = 0, 0
    for i in range(num_samples):
        question = data[3 * i].strip()
        output = data[3 * i + 1].strip()
        target = data[3 * i + 2].strip()

        if auto_stop:
            # clip
            output = strip_string(output, "<special_sep>").split("<special_sep>")
            len_true_context = len(true_contexts[i].strip().split("</s></s>")[1:])
            clipped = []
            for step in output:
                clipped.append(step)
                if "<stopper_stop>" in step:
                    break
            output = "<special_sep>".join(clipped)

        if target.lower() in ['yes', 'no']:
            continue
        count += 1
        if target.lower() in output.lower():
            count_contain += 1
    print("Ans.R:", count, count_contain, count_contain / count)

    ent_recall_full, ent_recall_ratios = [], []
    for i in range(num_samples):
        question = data[3 * i].strip()
        output = strip_string(data[3 * i + 1].strip(), "<special_sep>")

        if auto_stop:
            # clip
            output = output.split("<special_sep>")
            len_true_context = len(true_contexts[i].strip().split("</s></s>")[1:])
            clipped = []
            for step in output:
                clipped.append(step)
                if "<stopper_stop>" in step:
                    break
            output = "<special_sep>".join(clipped)

        true_context = true_contexts[i].strip().split("</s></s>")
        assert true_context[0].strip() == question.strip()
        true_context = true_context[1:]

        entities_true = set()

        for sent in true_context:
            ent = sent[:sent.find(" is ")]
            entities_true.add(ent)

        exist_count = 0
        for ent in entities_true:
            if ent.lower() in output.lower():
                exist_count += 1
        ratio = exist_count/len(entities_true)

        if ratio > 0.99:
            count = 1
        else:
            count = 0

        ent_recall_full.append(count)
        ent_recall_ratios.append(ratio)

    print("In QC eval mode. Num samples: {}, ent.R*:{}, ent.R: {}".format(
        num_samples, np.mean(ent_recall_full), np.mean(ent_recall_ratios)))

else:
    assert False
