import argparse
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL_TRACE = ROOT / "dataset" / "tool_calling_trace"
TRAIN = TOOL_TRACE / "train" / "intent_train.jsonl"
TEST = TOOL_TRACE / "test" / "intent_test.jsonl"


def shuffle_file(path: Path, seed: int | None = None, backup: bool = True):
    if not path.exists():
        print(f"Not found: {path}")
        return
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy(path, bak)
        print(f"Backup created: {bak}")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    records = [json.loads(line) for line in lines]
    if seed is not None:
        random.seed(seed)
    random.shuffle(records)

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Shuffled {path} -> {len(records)} records")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Shuffle intent train/test jsonl files (inplace with backup)")
    p.add_argument("--seed", type=int, help="Optional random seed for reproducibility")
    p.add_argument("--no-backup", action="store_true", help="Do not create .bak backup files")
    args = p.parse_args()

    shuffle_file(TRAIN, seed=args.seed, backup=(not args.no_backup))
    shuffle_file(TEST, seed=args.seed, backup=(not args.no_backup))
