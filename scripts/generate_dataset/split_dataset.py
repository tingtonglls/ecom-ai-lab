import argparse
import json
import math
import random
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_PATH = ROOT_DIR / "configs" / "dataset_config.json"
SINGLE_INPUT_DIR = ROOT_DIR / "outputs" / "single_intent"
MULTI_INPUT_DIR = ROOT_DIR / "outputs" / "multi_intent"
TOOL_TRACE_DIR = ROOT_DIR / "dataset" / "tool_calling_trace"
SELECTED_DIR = TOOL_TRACE_DIR / "selected"
TRAIN_DIR = TOOL_TRACE_DIR / "train"
TEST_DIR = TOOL_TRACE_DIR / "test"

# collect invalid/raw fragments that could not be parsed
INVALIDS = {}


def load_jsonl(path):
    text = Path(path).read_text(encoding="utf-8")

    # extract JSON objects by matching braces while respecting string literals
    objs = []
    cur = []
    depth = 0
    in_str = False
    esc = False
    for ch in text:
        cur.append(ch)
        if ch == '"' and not esc:
            in_str = not in_str
        if ch == '\\' and not esc:
            esc = True
            continue
        else:
            esc = False

        if not in_str:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    candidate = ''.join(cur).strip()
                    objs.append(candidate)
                    cur = []

    records = []
    for i, s in enumerate(objs, start=1):
        if not s:
            continue
        try:
            records.append(json.loads(s))
            continue
        except Exception:
            s2 = s.lstrip('`')
            s2 = s2.replace('，', ',').replace('：', ':').replace('“', '"').replace('”', '"').replace('、', ',')
            s2 = s2.replace('\u3000', ' ')
            try:
                records.append(json.loads(s2))
                continue
            except Exception:
                # Attempt to salvage by extracting braces
                start = s2.find('{')
                end = s2.rfind('}')
                if start != -1 and end != -1 and end > start:
                    candidate = s2[start:end+1]
                    # Heuristic: if there's stray text after a "思考" array, try to append it into the array
                    try:
                        records.append(json.loads(candidate))
                        continue
                    except Exception:
                        # try to locate "思考" array and absorb trailing stray text into it
                        think_key = '"思考"'
                        kpos = candidate.find(think_key)
                        if kpos != -1:
                            arr_start = candidate.find('[', kpos)
                            if arr_start != -1:
                                # find matching closing bracket for the 思考 array
                                depth2 = 0
                                in_str2 = False
                                esc2 = False
                                close_idx = -1
                                for idx in range(arr_start, len(candidate)):
                                    ch2 = candidate[idx]
                                    if ch2 == '"' and not esc2:
                                        in_str2 = not in_str2
                                    if ch2 == '\\' and not esc2:
                                        esc2 = True
                                        continue
                                    else:
                                        esc2 = False
                                    if not in_str2:
                                        if ch2 == '[':
                                            depth2 += 1
                                        elif ch2 == ']':
                                            depth2 -= 1
                                            if depth2 == 0:
                                                close_idx = idx
                                                break
                                if close_idx != -1:
                                    tail = candidate[close_idx+1:-1].strip()
                                    if tail:
                                        tail_clean = tail.lstrip(',').strip()
                                        tail_clean = tail_clean.rstrip(',').strip()
                                        new_candidate = candidate[:close_idx] + ', "' + tail_clean.replace('"', '\\"') + '"' + candidate[close_idx:]
                                        try:
                                            records.append(json.loads(new_candidate))
                                            continue
                                        except Exception as e:
                                            print(f"Failed to parse after merging stray tail in {path} object {i}: {e}")
                            print(f"Failed to parse candidate JSON in {path} object {i}")
                        print(f"Unable to parse JSON in {path} object {i}: {s[:200]}")
                        INVALIDS.setdefault(Path(path).stem, []).append(s)
                        # skip malformed object
                        continue
    return records


def load_grouped_jsonl(dir_path):
    grouped = {}
    for path in sorted(Path(dir_path).glob("*.jsonl")):
        grouped[path.stem] = load_jsonl(path)
    return grouped


def save_jsonl(records, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def scale_counts(base_counts, target_total):
    total = sum(base_counts.values())
    if total == 0:
        return {k: 0 for k in base_counts}

    scaled = {k: v * target_total / total for k, v in base_counts.items()}
    rounded = {k: math.floor(v) for k, v in scaled.items()}
    remainder = target_total - sum(rounded.values())
    fractions = sorted(
        ((scaled[k] - rounded[k], k) for k in scaled),
        key=lambda item: item[0],
        reverse=True,
    )
    for _, key in fractions[:remainder]:
        rounded[key] += 1
    return rounded


def equal_counts(keys, total):
    if not keys:
        return {}
    base = total // len(keys)
    counts = {key: base for key in keys}
    remainder = total - base * len(keys)
    for key in keys[:remainder]:
        counts[key] += 1
    return counts


def split_counts(counts, train_ratio):
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}

    scaled = {k: v * train_ratio for k, v in counts.items()}
    rounded = {k: math.floor(v) for k, v in scaled.items()}
    remainder = round(total * train_ratio) - sum(rounded.values())
    fractions = sorted(
        ((scaled[k] - rounded[k], k) for k in scaled),
        key=lambda item: item[0],
        reverse=True,
    )
    for _, key in fractions[:remainder]:
        rounded[key] += 1
    return rounded


def sample_exact(records, count):
    if count > len(records):
        raise ValueError(f"目标数量 {count} 大于可用记录数 {len(records)}")
    if count == len(records):
        return list(records)
    return random.sample(records, count)


def sample_single_intent(raw_single, target_total):
    tools = sorted(raw_single.keys())
    target_counts = equal_counts(tools, target_total)
    selected = {}
    for tool, count in target_counts.items():
        selected[tool] = sample_exact(raw_single[tool], count)
    return selected, target_counts


def sample_multi_intent(raw_multi, flow_distribution, target_total):
    flow_counts = scale_counts(flow_distribution, target_total)
    selected = {}
    for flow, count in flow_counts.items():
        if flow not in raw_multi:
            raise ValueError(f"多意图数据中缺少 flow: {flow}")
        selected[flow] = sample_exact(raw_multi[flow], count)
    return selected, flow_counts


def split_selected(selected_by_group, train_counts):
    train = {}
    test = {}
    for group_name, records in selected_by_group.items():
        group_total = len(records)
        desired_train = train_counts.get(group_name, 0)
        if desired_train > group_total:
            raise ValueError(
                f"group={group_name} 的训练集目标数 {desired_train} 大于总数 {group_total}"
            )
        shuffled = list(records)
        random.shuffle(shuffled)
        train[group_name] = shuffled[:desired_train]
        test[group_name] = shuffled[desired_train:]
    return train, test


def flatten(grouped_records):
    return [record for records in grouped_records.values() for record in records]


def print_count_table(title, counts):
    print(f"\n{title}")
    for key, value in sorted(counts.items()):
        print(f"  {key}: {value}")
    print(f"  total: {sum(counts.values())}\n")


def main():
    parser = argparse.ArgumentParser(description="按比例抽取单意图/多意图数据并切分训练/测试集")
    parser.add_argument("--single-total", type=int, default=2000, help="单意图抽样总数")
    parser.add_argument("--multi-total", type=int, default=3000, help="多意图抽样总数")
    parser.add_argument("--test-total", type=int, default=4000, help="最终测试集总数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    random.seed(args.seed)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    raw_single = load_grouped_jsonl(SINGLE_INPUT_DIR)
    raw_multi = load_grouped_jsonl(MULTI_INPUT_DIR)

    if not raw_single:
        raise RuntimeError(f"未找到单意图数据：{SINGLE_INPUT_DIR}")
    if not raw_multi:
        raise RuntimeError(f"未找到多意图数据：{MULTI_INPUT_DIR}")

    single_selected, single_counts = sample_single_intent(raw_single, args.single_total)
    multi_selected, multi_counts = sample_multi_intent(
        raw_multi,
        config["multi_intent"]["flow_count_distribution"],
        args.multi_total,
    )

    print_count_table("单意图抽样数量", single_counts)
    print_count_table("多意图抽样数量", multi_counts)

    train_ratio = 1 - args.test_total / (args.single_total + args.multi_total)
    single_train_counts = split_counts(single_counts, train_ratio=train_ratio)
    multi_train_counts = split_counts(multi_counts, train_ratio=train_ratio)

    print_count_table("单意图训练集数量", single_train_counts)
    print_count_table("多意图训练集数量", multi_train_counts)

    single_train, single_test = split_selected(single_selected, single_train_counts)
    multi_train, multi_test = split_selected(multi_selected, multi_train_counts)

    save_jsonl(flatten(single_selected), SELECTED_DIR / f"single_intent_{args.single_total}.jsonl")
    save_jsonl(flatten(multi_selected), SELECTED_DIR / f"multi_intent_{args.multi_total}.jsonl")

    # NOTE: swap outputs so file names match expected train/test contents
    save_jsonl(flatten(single_train), TEST_DIR / "single_intent_test.jsonl")
    save_jsonl(flatten(multi_train), TEST_DIR / "multi_intent_test.jsonl")
    save_jsonl(flatten(single_test), TRAIN_DIR / "single_intent_train.jsonl")
    save_jsonl(flatten(multi_test), TRAIN_DIR / "multi_intent_train.jsonl")

    save_jsonl(flatten(single_train) + flatten(multi_train), TEST_DIR / "intent_test.jsonl")
    save_jsonl(flatten(single_test) + flatten(multi_test), TRAIN_DIR / "intent_train.jsonl")

    print(f"\n已生成：")
    print(f"  {SELECTED_DIR / f'single_intent_{args.single_total}.jsonl'}")
    print(f"  {SELECTED_DIR / f'multi_intent_{args.multi_total}.jsonl'}")
    print(f"  {TRAIN_DIR / 'single_intent_train.jsonl'}")
    print(f"  {TRAIN_DIR / 'multi_intent_train.jsonl'}")
    print(f"  {TRAIN_DIR / 'intent_train.jsonl'}")
    print(f"  {TEST_DIR / 'single_intent_test.jsonl'}")
    print(f"  {TEST_DIR / 'multi_intent_test.jsonl'}")
    print(f"  {TEST_DIR / 'intent_test.jsonl'}")

    # export invalid fragments if any
    invalid_dir = TOOL_TRACE_DIR / "invalid"
    if INVALIDS:
        invalid_dir.mkdir(parents=True, exist_ok=True)
        for grp, items in INVALIDS.items():
            out_path = invalid_dir / f"{grp}_invalid.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                for it in items:
                    # replace real newlines with literal \n so each entry stays on one line
                    f.write(it.replace('\n', '\\n') + "\n\n")
        print(f"导出 {len(INVALIDS)} 个分组的无效条目到 {invalid_dir}")


if __name__ == "__main__":
    main()