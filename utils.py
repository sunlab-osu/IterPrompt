"""
utility helper functions.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy


def strip_string(text, stri):
    """
    strip the string STRI from both ends of TEXT
    """
    # strip from head
    while text.startswith(stri):
        text = text[len(stri):]
    # strip from tail
    while text.endswith(stri):
        text = text[:-len(stri)]
    return text


def detect_max_length(file_path, tokenizer, display=False):
    """
    file_path: the directory (with "/" at the end) which should contain some *.source, *.target files
    returns the max input (output) length of all *.source (.target) files inside file_path.
    """
    max_source, max_target = -1, -1
    all_file_names = os.listdir(file_path)
    for file_name in all_file_names:
        if file_name.endswith(".source"):
            f = open(file_path + file_name, 'r', encoding='utf-8')
            data = f.readlines()
            f.close()
            count = 0
            for line in data:
                count += 1
                if count % 10000 == 0 & display:
                    print(count, len(data))
                line = line.strip("\n")
                inputs = tokenizer(line)
                max_source = max(max_source, len(inputs["input_ids"]))
        elif file_name.endswith(".target"):
            f = open(file_path + file_name, 'r', encoding='utf-8')
            data = f.readlines()
            f.close()
            count = 0
            for line in data:
                count += 1
                if count % 10000 == 0 & display:
                    print(count, len(data))
                line = line.strip("\n")
                inputs = tokenizer(line)
                max_target = max(max_target, len(inputs["input_ids"]))
    return max_source, max_target


def IPoutput_to_RDinput(pred_file_loc):
    """
    file: prediction file for IterPrompt (to_predict, \t output, \t target)
    use_stop: whether do clip using the stopper marker
    returns the reader source input list and target output list
    """
    f = open(pred_file_loc, "r", encoding="utf-8")
    data = f.readlines()[3:]    # remove the first three indicator lines
    f.close()
    assert len(data) % 3 == 0
    num_samples = len(data)//3

    source_l, target_l = [], []
    for i in range(num_samples):
        to_predict = data[3*i].strip("\t").strip("\n")
        output = data[3*i+1].strip("\t").strip("\n")
        target = data[3*i+2].strip("\t").strip("\n")
        output = output.replace("<stopper_stop>", "")
        output = output.replace("<special_sep>", "</s></s>")
        output = strip_string(output, "</s></s>")

        source_l.append(to_predict + "</s></s>" + output)
        target_l.append(target)
    assert len(source_l) == len(target_l)
    return source_l, target_l


class FeedForwardNet(nn.Module):
    """
    A simple DNN with 2 hidden layers; for stopper module
    """
    def __init__(self, input_size, output_size, hid_layer=None):
        super(FeedForwardNet, self).__init__()
        self.hid_layer = hid_layer
        if self.hid_layer is None:
            self.fc1 = nn.Linear(input_size, output_size)
        else:
            assert type(self.hid_layer) == list
            assert len(self.hid_layer) == 2
            self.fc1 = nn.Linear(input_size, hid_layer[0])
            self.fc2 = nn.Linear(hid_layer[0], hid_layer[1])
            self.fc3 = nn.Linear(hid_layer[1], output_size)

    def forward(self, x):
        if self.hid_layer is None:
            return self.fc1(x)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def read_json_line(path):
    data = []
    with open(path, "r", encoding='utf-8') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

def tokenize(a):
    """
    lower, split, strip each token
    """
    b = a.lower().split()
    for ii in range(len(b)):
        b[ii] = b[ii].strip().strip('?.,\"\'').strip()
    return b

if __name__ == "__main__":
    pass
