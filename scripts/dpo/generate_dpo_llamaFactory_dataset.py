"""从 9k SFT 数据及其错误挖掘结果构造 LLaMA-Factory DPO 数据集。

构造原则：
1. 每个非空用户问题只保留一条 DPO 数据，避免同一 query 重复出现。
2. SFT 模型预测错误的 query 优先保留真实错误作为 rejected。
3. 其余 query 选择最困难的候选工具版本，并生成一种受控的规则负样本。
4. chosen 与 rejected 均为严格 JSON，且只能使用该样本的候选工具。
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SFT_INPUT_PATH = (
    PROJECT_ROOT / "dataset/tool_calling_trace/train/sft_train.jsonl"
)
PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs/dpo/qwen3_8b_sft_train_9k_predictions.jsonl"
)
TOOLS_PATH = PROJECT_ROOT / "configs/tools.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "dataset/tool_calling_trace/train/dpo_train.jsonl"
)
STATS_PATH = (
    PROJECT_ROOT / "dataset/tool_calling_trace/train/dpo_train_stats.json"
)
SEED = 42

TOOL_LINE_PATTERN = re.compile(r"^\d+\.\s*工具名：([A-Za-z0-9_]+)\s*$")

ERROR_ORDER = (
    "工具顺序错误",
    "遗漏工具",
    "多余工具",
    "错误工具替换",
)


def canonical_tool_json(tools):
    """输出统一、紧凑的工具轨迹 JSON 字符串。"""
    return json.dumps(
        {"工具": tools},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def read_jsonl(path, label):
    """读取 JSONL，并在错误信息中保留准确行号。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到{label}：{path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{label}第 {line_number} 行为空")
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label}第 {line_number} 行不是合法 JSON"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(f"{label}第 {line_number} 行必须是 JSON 对象")
            records.append(item)

    if not records:
        raise ValueError(f"{label}为空：{path}")
    return records


def load_allowed_tools(path):
    """读取工具库，工具定义的键即为允许使用的工具名。"""
    if not path.is_file():
        raise FileNotFoundError(f"找不到工具配置：{path}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"工具配置不是合法 JSON：{path}") from error
    if not isinstance(config, dict) or not config:
        raise ValueError("工具配置必须是非空 JSON 对象")
    if any(not isinstance(name, str) or not name for name in config):
        raise ValueError("工具配置包含非法工具名")
    return set(config)


def parse_tool_json(text, allowed_tools, candidate_tools, context):
    """解析并严格校验 chosen/rejected 使用的工具轨迹 JSON。"""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{context}不是非空字符串")
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as error:
        raise ValueError(f"{context}不是合法 JSON") from error

    if not isinstance(parsed, dict) or set(parsed) != {"工具"}:
        raise ValueError(f"{context}必须是只含‘工具’字段的 JSON 对象")
    tools = parsed["工具"]
    if not isinstance(tools, list) or not tools:
        raise ValueError(f"{context}中的‘工具’必须是非空数组")
    if any(not isinstance(tool, str) or not tool for tool in tools):
        raise ValueError(f"{context}含有非法工具名")
    if len(tools) != len(set(tools)):
        raise ValueError(f"{context}含有重复工具")

    unknown = [tool for tool in tools if tool not in allowed_tools]
    if unknown:
        raise ValueError(f"{context}含未知工具：{unknown}")
    outside = [tool for tool in tools if tool not in candidate_tools]
    if outside:
        raise ValueError(f"{context}含候选范围外工具：{outside}")
    return tools


def extract_input_parts(input_text, allowed_tools, line_number):
    """从 Alpaca input 中提取候选工具（保留顺序）和用户问题。"""
    if not isinstance(input_text, str) or not input_text.strip():
        raise ValueError(f"SFT 第 {line_number} 行的 input 为空或不是字符串")
    if "【候选工具】" not in input_text or "【用户问题】" not in input_text:
        raise ValueError(
            f"SFT 第 {line_number} 行缺少候选工具或用户问题区块"
        )

    candidate_block, query_block = input_text.split(
        "【用户问题】", maxsplit=1
    )
    candidate_block = candidate_block.split("【候选工具】", maxsplit=1)[1]
    candidate_tools = []
    for line in candidate_block.splitlines():
        match = TOOL_LINE_PATTERN.match(line.strip())
        if match:
            candidate_tools.append(match.group(1))

    if not candidate_tools:
        raise ValueError(f"SFT 第 {line_number} 行未提取到候选工具")
    if len(candidate_tools) != len(set(candidate_tools)):
        raise ValueError(f"SFT 第 {line_number} 行的候选工具有重复")
    unknown = [tool for tool in candidate_tools if tool not in allowed_tools]
    if unknown:
        raise ValueError(
            f"SFT 第 {line_number} 行候选中有未知工具：{unknown}"
        )

    query = query_block.strip()
    if not query:
        raise ValueError(f"SFT 第 {line_number} 行的用户问题为空")
    return candidate_tools, query


def load_sft_records(path, allowed_tools):
    """读取并规范化 9k Alpaca SFT 数据。"""
    raw_records = read_jsonl(path, "SFT训练集")
    records = []
    for line_number, item in enumerate(raw_records, start=1):
        if set(item) != {"instruction", "input", "output"}:
            raise ValueError(
                f"SFT 第 {line_number} 行必须只含 instruction、input、output"
            )
        instruction = item["instruction"]
        input_text = item["input"]
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"SFT 第 {line_number} 行的 instruction 非法")

        candidate_tools, query = extract_input_parts(
            input_text, allowed_tools, line_number
        )
        gold_tools = parse_tool_json(
            item["output"],
            allowed_tools,
            candidate_tools,
            f"SFT 第 {line_number} 行 output",
        )
        records.append({
            "id": line_number,
            "instruction": instruction,
            "input": input_text,
            "query": query,
            "candidate_tools": candidate_tools,
            "gold_tools": gold_tools,
        })
    return records


def classify_tools(gold_tools, predicted_tools):
    """独立复算错误标签，标签允许多选。"""
    if predicted_tools == gold_tools:
        return []

    error_types = []
    missing = [tool for tool in gold_tools if tool not in predicted_tools]
    extra = [tool for tool in predicted_tools if tool not in gold_tools]
    if missing:
        error_types.append("遗漏工具")
    if extra:
        error_types.append("多余工具")
    if missing and extra:
        error_types.append("错误工具替换")
    if (
        len(predicted_tools) == len(gold_tools)
        and set(predicted_tools) == set(gold_tools)
        and predicted_tools != gold_tools
    ):
        error_types.append("工具顺序错误")
    if not error_types:
        error_types.append("其他轨迹错误")
    return error_types


def validate_predictions(path, sft_records, allowed_tools):
    """确认推理结果与 9k SFT 数据逐行对应，并复算正确性。"""
    predictions = read_jsonl(path, "SFT推理结果")
    if len(predictions) != len(sft_records):
        raise ValueError(
            "SFT推理结果与SFT训练集数量不一致："
            f"{len(predictions)} != {len(sft_records)}"
        )

    validated = []
    stable_fields = {
        "instruction": "instruction",
        "input": "input",
        "候选工具": "candidate_tools",
        "GT工具": "gold_tools",
    }
    for line_number, (prediction, sft) in enumerate(
        zip(predictions, sft_records, strict=True), start=1
    ):
        if prediction.get("id") != line_number:
            raise ValueError(f"推理结果第 {line_number} 行的 id 不连续")
        for prediction_field, sft_field in stable_fields.items():
            if prediction.get(prediction_field) != sft[sft_field]:
                raise ValueError(
                    f"推理结果第 {line_number} 行的 {prediction_field} "
                    "与SFT训练集不一致"
                )

        predicted_tools = parse_tool_json(
            prediction.get("模型预测输出"),
            allowed_tools,
            sft["candidate_tools"],
            f"推理结果第 {line_number} 行的模型预测输出",
        )
        is_correct = predicted_tools == sft["gold_tools"]
        expected_result = "正确" if is_correct else "错误"
        if prediction.get("推理结果") != expected_result:
            raise ValueError(
                f"推理结果第 {line_number} 行的正确/错误标记复算不一致"
            )
        if prediction.get("预测工具") != predicted_tools:
            raise ValueError(
                f"推理结果第 {line_number} 行的预测工具复算不一致"
            )
        if prediction.get("解析错误") is not None:
            raise ValueError(
                f"推理结果第 {line_number} 行存在解析错误，"
                "不能直接作为严格 DPO rejected"
            )

        error_types = classify_tools(sft["gold_tools"], predicted_tools)
        if prediction.get("错误类型") != error_types:
            raise ValueError(
                f"推理结果第 {line_number} 行的错误类型复算不一致："
                f"保存值={prediction.get('错误类型')}，复算值={error_types}"
            )
        validated.append({
            **sft,
            "predicted_tools": predicted_tools,
            "is_correct": is_correct,
            "error_types": error_types,
        })
    return validated


def candidate_inversion_count(candidate_tools, gold_tools):
    """计算候选展示中的 GT 相对顺序与真实调用顺序的逆序数。"""
    gold_index = {tool: index for index, tool in enumerate(gold_tools)}
    displayed_gold = [
        gold_index[tool] for tool in candidate_tools if tool in gold_index
    ]
    return sum(
        displayed_gold[left] > displayed_gold[right]
        for left in range(len(displayed_gold))
        for right in range(left + 1, len(displayed_gold))
    )


def hardness_score(record):
    """为同一 query 的不同候选版本计算困难度。"""
    distractors = [
        tool
        for tool in record["candidate_tools"]
        if tool not in record["gold_tools"]
    ]
    return (
        bool(distractors),
        len(distractors),
        len(record["candidate_tools"]),
        candidate_inversion_count(
            record["candidate_tools"], record["gold_tools"]
        ),
    )


def choose_hardest(records, rng):
    """选最高困难度版本；完全并列时用固定随机种子选择。"""
    best_score = max(hardness_score(record) for record in records)
    tied = [record for record in records if hardness_score(record) == best_score]
    tied.sort(key=lambda record: record["id"])
    return rng.choice(tied)


def choose_error_type(feasible_types, error_counter, rng):
    """优先选择当前使用次数最少的可行错误类型。"""
    minimum = min(error_counter[error_type] for error_type in feasible_types)
    tied = [
        error_type
        for error_type in ERROR_ORDER
        if error_type in feasible_types and error_counter[error_type] == minimum
    ]
    return rng.choice(tied)


def build_synthetic_rejected(record, error_counter, rng):
    """构造只包含一种受控错误的困难负样本。"""
    gold_tools = record["gold_tools"]
    distractors = [
        tool
        for tool in record["candidate_tools"]
        if tool not in gold_tools
    ]

    if len(gold_tools) == 1:
        feasible = ["错误工具替换", "多余工具"]
    else:
        feasible = ["工具顺序错误", "遗漏工具"]
        if distractors:
            feasible.extend(["多余工具", "错误工具替换"])

    if any(error_type in ("多余工具", "错误工具替换") for error_type in feasible):
        if not distractors:
            feasible = [
                error_type
                for error_type in feasible
                if error_type not in ("多余工具", "错误工具替换")
            ]
    if not feasible:
        raise ValueError(f"第 {record['id']} 条没有可构造的负样本类型")

    error_type = choose_error_type(feasible, error_counter, rng)
    rejected_tools = list(gold_tools)
    if error_type == "工具顺序错误":
        swap_index = rng.randrange(len(rejected_tools) - 1)
        rejected_tools[swap_index], rejected_tools[swap_index + 1] = (
            rejected_tools[swap_index + 1],
            rejected_tools[swap_index],
        )
    elif error_type == "遗漏工具":
        del rejected_tools[rng.randrange(len(rejected_tools))]
    elif error_type == "多余工具":
        insert_index = rng.randrange(len(rejected_tools) + 1)
        rejected_tools.insert(insert_index, rng.choice(distractors))
    elif error_type == "错误工具替换":
        rejected_tools[rng.randrange(len(rejected_tools))] = rng.choice(distractors)
    else:
        raise AssertionError(f"未实现的错误类型：{error_type}")

    if not rejected_tools or rejected_tools == gold_tools:
        raise AssertionError(f"第 {record['id']} 条生成了无效负样本")
    if len(rejected_tools) != len(set(rejected_tools)):
        raise AssertionError(f"第 {record['id']} 条负样本含重复工具")
    return rejected_tools, [error_type]


def build_dpo_records(validated_records, allowed_tools, seed):
    """按 query 去重，并构造最终 DPO 记录及内部统计信息。"""
    groups = defaultdict(list)
    for record in validated_records:
        groups[record["query"]].append(record)

    for query, records in groups.items():
        gold_variants = {tuple(record["gold_tools"]) for record in records}
        if len(gold_variants) != 1:
            raise ValueError(
                f"同一用户问题对应多个不同 GT，不能自动构造 DPO：{query}"
            )

    rng = random.Random(seed)
    # 同一 query 可能在不同候选排列下都预测错误。先为每个错误 query
    # 只选一个最困难版本，再用这些真正会进入 DPO 的错误初始化平衡计数。
    selected_natural_by_query = {}
    for query in sorted(groups):
        natural_errors = [
            record for record in groups[query] if not record["is_correct"]
        ]
        if natural_errors:
            selected_natural_by_query[query] = choose_hardest(
                natural_errors, rng
            )

    error_counter = Counter()
    for record in selected_natural_by_query.values():
        error_counter.update(record["error_types"])

    output_records = []
    metadata_records = []
    selected_natural_ids = []
    natural_error_query_count = 0

    for query in sorted(groups):
        variants = groups[query]
        if query in selected_natural_by_query:
            natural_error_query_count += 1
            selected = selected_natural_by_query[query]
            rejected_tools = selected["predicted_tools"]
            error_types = selected["error_types"]
            source = "SFT模型真实错误"
            selected_natural_ids.append(selected["id"])
        else:
            selected = choose_hardest(variants, rng)
            rejected_tools, error_types = build_synthetic_rejected(
                selected, error_counter, rng
            )
            error_counter.update(error_types)
            source = "规则构造"

        chosen = canonical_tool_json(selected["gold_tools"])
        rejected = canonical_tool_json(rejected_tools)
        if chosen == rejected:
            raise AssertionError(f"第 {selected['id']} 条 chosen 与 rejected 相同")

        parse_tool_json(
            chosen,
            allowed_tools,
            selected["candidate_tools"],
            f"第 {selected['id']} 条 chosen",
        )
        parse_tool_json(
            rejected,
            allowed_tools,
            selected["candidate_tools"],
            f"第 {selected['id']} 条 rejected",
        )

        output_records.append({
            "instruction": selected["instruction"],
            "input": selected["input"],
            "chosen": chosen,
            "rejected": rejected,
        })
        metadata_records.append({
            "query": query,
            "source_id": selected["id"],
            "source": source,
            "intent_type": (
                "单意图" if len(selected["gold_tools"]) == 1 else "多意图"
            ),
            "error_types": error_types,
            "gold_tools": selected["gold_tools"],
            "rejected_tools": rejected_tools,
            "candidate_tools": selected["candidate_tools"],
        })

    paired = list(zip(output_records, metadata_records, strict=True))
    rng.shuffle(paired)
    output_records = [pair[0] for pair in paired]
    metadata_records = [pair[1] for pair in paired]

    natural_error_ids = {
        record["id"] for record in validated_records if not record["is_correct"]
    }
    skipped_duplicate_natural_ids = sorted(
        natural_error_ids - set(selected_natural_ids)
    )

    return (
        output_records,
        metadata_records,
        len(groups),
        natural_error_query_count,
        sorted(selected_natural_ids),
        skipped_duplicate_natural_ids,
    )


def distribution(values):
    """生成键按数值升序排列的分布字典。"""
    counter = Counter(values)
    return {str(key): counter[key] for key in sorted(counter)}


def build_stats(
    output_records,
    metadata_records,
    unique_query_count,
    natural_error_query_count,
    natural_error_ids,
    raw_natural_error_count,
    skipped_duplicate_natural_ids,
    allowed_tools,
    args,
):
    """生成可复核的 DPO 数据集统计信息。"""
    query_counter = Counter(item["query"] for item in metadata_records)
    pair_counter = Counter(
        (
            row["instruction"],
            row["input"],
            row["chosen"],
            row["rejected"],
        )
        for row in output_records
    )
    source_counter = Counter(item["source"] for item in metadata_records)
    intent_counter = Counter(item["intent_type"] for item in metadata_records)
    error_type_counter = Counter()
    chosen_tool_counter = Counter()
    rejected_tool_counter = Counter()
    for item in metadata_records:
        error_type_counter.update(item["error_types"])
        chosen_tool_counter.update(item["gold_tools"])
        rejected_tool_counter.update(item["rejected_tools"])

    missing_chosen_tools = sorted(allowed_tools - set(chosen_tool_counter))
    if missing_chosen_tools:
        raise ValueError(
            f"最终 chosen 未覆盖全部工具：{missing_chosen_tools}"
        )

    return {
        "总数": len(output_records),
        "唯一query数": unique_query_count,
        "意图类型统计": {
            "单意图": intent_counter["单意图"],
            "多意图": intent_counter["多意图"],
        },
        "负样本来源统计": {
            "SFT模型真实错误": source_counter["SFT模型真实错误"],
            "规则构造": source_counter["规则构造"],
        },
        "SFT原始错误记录数": raw_natural_error_count,
        "真实错误query数": natural_error_query_count,
        "纳入的真实错误原始id": natural_error_ids,
        "因query重复而跳过的真实错误原始id": skipped_duplicate_natural_ids,
        "错误类型统计": {
            error_type: error_type_counter[error_type]
            for error_type in ERROR_ORDER
        },
        "错误类型说明": (
            "SFT模型真实错误允许多标签，因此错误类型数量之和可能大于总数。"
        ),
        "GT工具数量分布": distribution(
            len(item["gold_tools"]) for item in metadata_records
        ),
        "候选工具数量分布": distribution(
            len(item["candidate_tools"]) for item in metadata_records
        ),
        "chosen工具覆盖": {
            "已覆盖工具数": len(chosen_tool_counter),
            "工具总数": len(allowed_tools),
            "未覆盖工具": missing_chosen_tools,
            "各工具出现次数": {
                tool: chosen_tool_counter[tool] for tool in sorted(allowed_tools)
            },
        },
        "rejected工具覆盖": {
            "已覆盖工具数": len(rejected_tool_counter),
            "工具总数": len(allowed_tools),
            "未覆盖工具": sorted(allowed_tools - set(rejected_tool_counter)),
            "各工具出现次数": {
                tool: rejected_tool_counter[tool] for tool in sorted(allowed_tools)
            },
        },
        "重复检查": {
            "重复query数": sum(count > 1 for count in query_counter.values()),
            "重复DPO对数": sum(count > 1 for count in pair_counter.values()),
        },
        "metadata": {
            "seed": args.seed,
            "sft_input": str(args.sft_input),
            "predictions": str(args.predictions),
            "tools": str(args.tools),
            "output": str(args.output),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def validate_final_records(output_records, metadata_records, allowed_tools):
    """写盘前重新验证四字段格式、query 唯一性和 DPO 对唯一性。"""
    if len(output_records) != len(metadata_records):
        raise AssertionError("DPO 数据与内部统计记录数量不一致")
    expected_fields = {"instruction", "input", "chosen", "rejected"}
    seen_queries = set()
    seen_pairs = set()
    for index, (row, metadata) in enumerate(
        zip(output_records, metadata_records, strict=True), start=1
    ):
        if set(row) != expected_fields:
            raise AssertionError(f"最终第 {index} 条字段不正确")
        if not all(isinstance(row[field], str) and row[field].strip() for field in row):
            raise AssertionError(f"最终第 {index} 条含空字段或非字符串字段")
        if metadata["query"] in seen_queries:
            raise AssertionError(f"最终数据存在重复 query：{metadata['query']}")
        seen_queries.add(metadata["query"])

        pair = tuple(row[field] for field in (
            "instruction", "input", "chosen", "rejected"
        ))
        if pair in seen_pairs:
            raise AssertionError(f"最终第 {index} 条形成重复 DPO 对")
        seen_pairs.add(pair)

        chosen_tools = parse_tool_json(
            row["chosen"],
            allowed_tools,
            metadata["candidate_tools"],
            f"最终第 {index} 条 chosen",
        )
        rejected_tools = parse_tool_json(
            row["rejected"],
            allowed_tools,
            metadata["candidate_tools"],
            f"最终第 {index} 条 rejected",
        )
        if chosen_tools != metadata["gold_tools"]:
            raise AssertionError(f"最终第 {index} 条 chosen 与 GT 不一致")
        if rejected_tools == chosen_tools:
            raise AssertionError(f"最终第 {index} 条 chosen 与 rejected 相同")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "基于 9k SFT 数据和 SFT 错误挖掘结果，构造 query 唯一的 "
            "LLaMA-Factory DPO 数据集"
        )
    )
    parser.add_argument("--sft-input", type=Path, default=SFT_INPUT_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--tools", type=Path, default=TOOLS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--stats", type=Path, default=STATS_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的 output 和 stats；默认拒绝覆盖",
    )
    return parser.parse_args()


def validate_paths(args):
    """避免输出互相覆盖或误覆盖任何输入文件。"""
    input_paths = {
        args.sft_input.resolve(),
        args.predictions.resolve(),
        args.tools.resolve(),
    }
    output_paths = {args.output.resolve(), args.stats.resolve()}
    if len(output_paths) != 2:
        raise ValueError("output 和 stats 不能是同一路径")
    if input_paths & output_paths:
        raise ValueError("输出路径不能覆盖 SFT、推理结果或工具配置输入")
    if not args.overwrite:
        for path in (args.output, args.stats):
            if path.exists():
                raise FileExistsError(
                    f"输出文件已存在：{path}。请更换路径，或明确使用 --overwrite。"
                )


def write_outputs(output_records, stats, args):
    """仅在全部校验通过后写入最终数据集与统计文件。"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with args.output.open(mode, encoding="utf-8") as file:
        for row in output_records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with args.stats.open(mode, encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    args = parse_args()
    validate_paths(args)

    allowed_tools = load_allowed_tools(args.tools)
    sft_records = load_sft_records(args.sft_input, allowed_tools)
    validated_records = validate_predictions(
        args.predictions, sft_records, allowed_tools
    )
    natural_error_count = sum(
        not record["is_correct"] for record in validated_records
    )

    (
        output_records,
        metadata_records,
        unique_query_count,
        natural_error_query_count,
        natural_error_ids,
        skipped_duplicate_natural_ids,
    ) = build_dpo_records(validated_records, allowed_tools, args.seed)
    validate_final_records(output_records, metadata_records, allowed_tools)

    if len(output_records) != unique_query_count:
        raise AssertionError("最终 DPO 数量与唯一 query 数不一致")
    stats = build_stats(
        output_records,
        metadata_records,
        unique_query_count,
        natural_error_query_count,
        natural_error_ids,
        natural_error_count,
        skipped_duplicate_natural_ids,
        allowed_tools,
        args,
    )
    write_outputs(output_records, stats, args)

    print(f"SFT 原始数据：{len(sft_records)} 条")
    print(f"有效唯一 query：{unique_query_count} 条")
    print(
        f"真实模型错误：{natural_error_count} 条，涉及 "
        f"{natural_error_query_count} 个唯一 query；"
        f"规则负样本：{stats['负样本来源统计']['规则构造']} 条"
    )
    print(
        f"最终 DPO 数据：{len(output_records)} 条（"
        f"单意图 {stats['意图类型统计']['单意图']}，"
        f"多意图 {stats['意图类型统计']['多意图']}）"
    )
    print(f"DPO 数据已保存：{args.output}")
    print(f"统计结果已保存：{args.stats}")


if __name__ == "__main__":
    main()
