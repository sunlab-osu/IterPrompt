import argparse
import os
os.environ['TRANSFORMERS_CACHE'] = 'D:/huggingface_cache/transformers'
os.environ['HF_DATASETS_CACHE'] = 'D:/huggingface_cache/datasets'

from transformers import BartTokenizer
from utils import detect_max_length

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True, help="path containing .source/.target files")

args = parser.parse_args()

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
max_source, max_target = detect_max_length(args.path, tokenizer, display=True)
print("source & target max length:", max_source, max_target)
