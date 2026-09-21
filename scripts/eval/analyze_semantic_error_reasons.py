import argparse
import json
import csv
from pathlib import Path
from collections import Counter


def classify_reason(reason: str):
    if not reason:
        return {"omission": False, "extra": False, "semantic": False}

    r = reason.lower()
    # High-precision triggers
    omission_keys = [
        "缺少", "缺失", "未覆盖", "未包含", "未完全覆盖", "未完整覆盖",
        "未覆盖核心含义"
    ]
    extra_keys = [
        "多出", "多余", "额外", "误抽", "误将", "替换为", "替换", "替代", "被替换"
    ]
    # Only explicit negative semantic-failure phrases are treated as semantic failures
    semantic_keys = [
        "未识别到", "未识别", "没有识别", "语义不一致", "语义不等价", "语义不相符",
        "语义偏差", "未完全等价", "没有完全等价"
    ]

    # ambiguous group (上下位/实例/泛化/相近) — do NOT treat as semantic failure by default
    ambiguous_keys = [
        "上下位", "下位", "上位", "具体实例", "泛化概念", "语义相近", "语义相关", "近义", "同义", "包含关系", "隐含"
    ]

    omission = any(k in r for k in omission_keys)
    extra = any(k in r for k in extra_keys)
    semantic = any(k in r for k in semantic_keys)

    # Conservative handling for ambiguous mentions: promote to omission/extra if co-occurring
    if not semantic:
        if any(k in r for k in ambiguous_keys):
            # if ambiguous mention co-occurs with omission trigger -> omission
            if not omission and any(k in r for k in omission_keys):
                omission = True
            # if ambiguous mention co-occurs with extra trigger -> extra
            if not extra and any(k in r for k in extra_keys):
                extra = True
            # otherwise keep semantic=False and leave for manual review

    return {"omission": omission, "extra": extra, "semantic": semantic}


def analyze(input_path: Path):
    strict_zero_items = []
    counts = Counter()

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            strict_match = item.get("严格匹配")
            if strict_match == 0:
                reason = item.get("原因", "")
                labels = classify_reason(reason)
                strict_zero_items.append((item.get("id"), item.get("问题"), reason, labels))

                if labels["omission"]:
                    counts["关键词遗漏"] += 1
                if labels["extra"]:
                    counts["关键词误抽/多余/替换"] += 1
                if labels["semantic"]:
                    counts["语义同义未识别"] += 1

    return strict_zero_items, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="语义评测 jsonl 文件路径")
    args = parser.parse_args()

    input_path = Path(args.input)
    items, counts = analyze(input_path)

    print(f"严格匹配=0 的样本数：{len(items)}")
    print("按原因标签统计（可重复计数）：")
    print(f"- 关键词遗漏: {counts['关键词遗漏']}")
    print(f"- 关键词误抽/多余/替换: {counts['关键词误抽/多余/替换']}")
    print(f"- 语义同义未识别: {counts['语义同义未识别']}")

    print("\n样例：")  #输出5个样例看看效果
    for item_id, question, reason, labels in items[:5]:
        print(f"ID {item_id}: omission={labels['omission']} extra={labels['extra']} semantic={labels['semantic']}")
        print(f"  问题: {question}")
        print(f"  原因: {reason}")

    # write CSV of strict==0 items for manual review
    out_dir = input_path.parent
    out_csv = out_dir / (input_path.stem + "_strict0_labeled.csv")
    with out_csv.open("w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["id", "问题", "原因", "omission", "extra", "semantic"])
        for item_id, question, reason, labels in items:
            writer.writerow([item_id, question, reason, int(labels["omission"]), int(labels["extra"]), int(labels["semantic"])])

    print(f"\n已导出 strict==0 的标注 CSV：{out_csv}")


if __name__ == "__main__":
    main()
