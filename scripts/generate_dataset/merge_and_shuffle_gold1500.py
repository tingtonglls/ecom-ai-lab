#!/usr/bin/env python3
"""Merge two JSONL intent files and shuffle into a single gold_1500.jsonl.

Usage:
  python scripts/generate_dataset/merge_and_shuffle_gold1500.py \
    -i dataset/intent_recognition/chat_intent_500.jsonl dataset/intent_recognition/ecommerce_intent_1000.jsonl \
    -o dataset/intent_recognition/gold_1500.jsonl -s 42
"""
import argparse
import json
import random
from pathlib import Path
import sys

sys.argv =[
    "merge_and_shuffle_gold1500.py",
    "-i",
    "dataset/intent_recognition/chat_intent_500.jsonl",
    "dataset/intent_recognition/ecommerce_intent_1000.jsonl",
    "-o",
    "dataset/intent_recognition/gold_1500.jsonl",
    "-s",
    "42"
    ]

def load_jsonl(path: Path):
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, items):
    with path.open('w', encoding='utf-8') as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputs', nargs='+', required=True, help='Input JSONL files')
    parser.add_argument('-o', '--output', required=True, help='Output JSONL file')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed for shuffling')
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_items = []
    for p in args.inputs:
        items = load_jsonl(Path(p))
        all_items.extend(items)

    random.seed(args.seed)
    random.shuffle(all_items)

    write_jsonl(out_path, all_items)
    print(f'Wrote {len(all_items)} records to {out_path}')


if __name__ == '__main__':
    main()
