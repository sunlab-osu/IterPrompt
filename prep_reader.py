import argparse

import os
from utils import IPoutput_to_RDinput

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True, help="path to create")
parser.add_argument("--train", default=None, type=str, required=False, help="train prediction file")
parser.add_argument("--valid", default=None, type=str, required=False, help="valid prediction file")
parser.add_argument("--test", default=None, type=str, required=False, help="test prediction file")

args = parser.parse_args()

path = args.path
os.makedirs(path, exist_ok=True)

# train
if not args.train is None:
    source_l, target_l = IPoutput_to_RDinput(args.train)
    f1 = open(path+"train.source", "w", encoding='utf-8')
    f2 = open(path+"train.target", "w", encoding='utf-8')
    for i in range(len(source_l)):
        f1.write(source_l[i])
        f1.write("\n")
        f2.write(target_l[i])
        f2.write("\n")
    f1.close()
    f2.close()

# valid
if not args.valid is None:
    source_l, target_l = IPoutput_to_RDinput(args.valid)
    f1 = open(path+"valid.source", "w", encoding='utf-8')
    f2 = open(path+"valid.target", "w", encoding='utf-8')
    for i in range(len(source_l)):
        f1.write(source_l[i])
        f1.write("\n")
        f2.write(target_l[i])
        f2.write("\n")
    f1.close()
    f2.close()

# test
if not args.test is None:
    source_l, target_l = IPoutput_to_RDinput(args.test)
    f1 = open(path+"test.source", "w", encoding='utf-8')
    f2 = open(path+"test.target", "w", encoding='utf-8')
    for i in range(len(source_l)):
        f1.write(source_l[i])
        f1.write("\n")
        f2.write(target_l[i])
        f2.write("\n")
    f1.close()
    f2.close()
