"""构建 LLaMA-Factory Alpaca 格式的工具调用轨迹 SFT 数据集。

默认生成：
- 3000 条单意图数据：1 个 GT 工具 + 1～4 个同类干扰工具。
- 4000 条多意图排序数据：只包含 GT 工具，候选顺序与 GT 不同。
- 2000 条多意图困难数据：GT 工具 + 1～3 个同类干扰工具，候选工具最多 5 个。

输出 JSONL 每行只含 instruction、input、output 三个字段。
原始数据中的“思考”字段不会进入 SFT 输入或输出。
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SINGLE_INPUT_PATH = (
    PROJECT_ROOT / "dataset/tool_calling_trace/train/single_intent_train.jsonl"
)
MULTI_INPUT_PATH = (
    PROJECT_ROOT / "dataset/tool_calling_trace/train/multi_intent_train.jsonl"
)
TOOLS_PATH = PROJECT_ROOT / "configs/tools.json"
COMBINATIONS_PATH = PROJECT_ROOT / "configs/multi_intent_combinations.json"
OUTPUT_DIR = PROJECT_ROOT / "dataset/tool_calling_trace/train"

SINGLE_OUTPUT_NAME = "single_intent_sft_train.jsonl"
MULTI_OUTPUT_NAME = "multi_intent_sft_train.jsonl"
MERGED_OUTPUT_NAME = "sft_train.jsonl"
STATS_OUTPUT_NAME = "sft_train_stats.json"

DEFAULT_SEED = 42
DEFAULT_SINGLE_COUNT = 3000
DEFAULT_MULTI_SHUFFLE_COUNT = 4000
DEFAULT_MULTI_HARD_COUNT = 2000
MAX_CANDIDATE_TOOLS = 5

# 已知源数据中有 1 处工具名拼写错误。仅在构建时内存纠正，不修改源文件。
TOOL_NAME_ALIASES = {
    "g'et_item_info": "get_item_info",
}

INSTRUCTION = (
    "你是电商导购 Agent 的工具调用规划器。请根据用户问题，从候选工具中选择"
    "完成请求所必需的工具，并按实际调用顺序输出。候选工具的展示顺序不代表"
    "调用顺序，其中可能包含不需要调用的干扰工具。不得使用候选列表之外的工具，"
    "不得重复工具。只输出一个合法的 JSON 对象，格式为 "
    "{\"工具\":[\"工具名1\",\"工具名2\"]}，不输出分析过程、调用参数、Markdown 或其他内容。"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="构建 LLaMA-Factory Alpaca 格式的工具调用轨迹 SFT 数据集"
    )
    parser.add_argument("--single-input", type=Path, default=SINGLE_INPUT_PATH)
    parser.add_argument("--multi-input", type=Path, default=MULTI_INPUT_PATH)
    parser.add_argument("--tools", type=Path, default=TOOLS_PATH)
    parser.add_argument("--combinations", type=Path, default=COMBINATIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--single-count",
        type=int,
        default=DEFAULT_SINGLE_COUNT,
        help="单意图 SFT 样本数",
    )
    parser.add_argument(
        "--multi-shuffle-count",
        type=int,
        default=DEFAULT_MULTI_SHUFFLE_COUNT,
        help="只打乱 GT 候选顺序的多意图样本数",
    )
    parser.add_argument(
        "--multi-hard-count",
        type=int,
        default=DEFAULT_MULTI_HARD_COUNT,
        help="增加同类干扰工具的多意图样本数",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的四个目标输出文件",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_tools_config(tools_config):
    if not isinstance(tools_config, dict) or not tools_config:
        raise ValueError("tools.json 必须是非空 JSON 对象")

    for tool_name, config in tools_config.items():
        if not isinstance(tool_name, str) or not isinstance(config, dict):
            raise ValueError("tools.json 的工具配置格式错误")
        for field in ("description", "enhanced_description"):
            value = config.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"工具 {tool_name} 缺少有效的 {field}")


def normalize_tool_name(tool_name, correction_counter):
    normalized_name = TOOL_NAME_ALIASES.get(tool_name, tool_name)
    if normalized_name != tool_name:
        correction_counter[f"{tool_name} -> {normalized_name}"] += 1
    return normalized_name


def load_source_jsonl(
    path,
    expected_kind,
    allowed_tools,
    correction_counter,
    repeated_gt_counter,
    skipped_source_counter,
):
    """读取问题和有序 GT 工具，不使用原始“思考”字段。"""
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path} 第 {line_number} 行为空")

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from error

            question = item.get("问题")
            raw_gold = item.get("工具")
            if not isinstance(question, str) or not question.strip():
                skipped_source_counter[f"{expected_kind}_empty_question"] += 1
                continue

            if isinstance(raw_gold, str):
                gold_tools = [raw_gold]
            elif isinstance(raw_gold, list):
                gold_tools = raw_gold.copy()
            else:
                raise ValueError(f"{path} 第 {line_number} 行缺少有效的‘工具’")

            if not gold_tools or any(not isinstance(tool, str) for tool in gold_tools):
                raise ValueError(f"{path} 第 {line_number} 行的 GT 工具格式错误")

            gold_tools = [
                normalize_tool_name(tool, correction_counter)
                for tool in gold_tools
            ]
            unknown_tools = [tool for tool in gold_tools if tool not in allowed_tools]
            if unknown_tools:
                raise ValueError(
                    f"{path} 第 {line_number} 行含有未知工具：{unknown_tools}"
                )

            # 输出格式规定调用轨迹不重复工具。保留首次出现，移除后续重复项。
            unique_gold_tools = []
            seen_gold_tools = set()
            for tool in gold_tools:
                if tool in seen_gold_tools:
                    repeated_gt_counter[tool] += 1
                    continue
                seen_gold_tools.add(tool)
                unique_gold_tools.append(tool)
            gold_tools = unique_gold_tools

            if expected_kind == "single" and len(gold_tools) != 1:
                raise ValueError(f"{path} 第 {line_number} 行不是单意图数据")
            if expected_kind == "multi" and not 2 <= len(gold_tools) <= MAX_CANDIDATE_TOOLS:
                raise ValueError(f"{path} 第 {line_number} 行不是 2～5 工具的多意图数据")

            records.append({
                "source_line": line_number,
                "question": question.strip(),
                "gold_tools": gold_tools,
            })

    if not records:
        raise ValueError(f"源数据文件为空：{path}")
    return records


def deduplicate_records(records):
    """按“问题 + 有序 GT”去重，保留首次出现的数据。"""
    unique_records = []
    seen = set()
    duplicate_count = 0
    for record in records:
        key = (record["question"], tuple(record["gold_tools"]))
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_records.append(record)
    return unique_records, duplicate_count


def build_flow_pools(combinations, allowed_tools):
    """将每个业务 flow 中出现的工具合并为同类候选池。"""
    if not isinstance(combinations, dict) or not combinations:
        raise ValueError("multi_intent_combinations.json 格式错误")

    flow_pools = {}
    for flow_name, chains in combinations.items():
        if not isinstance(chains, list) or not chains:
            raise ValueError(f"flow {flow_name} 没有有效的工具组合")
        pool = set()
        for chain in chains:
            if not isinstance(chain, list) or not chain:
                raise ValueError(f"flow {flow_name} 中存在无效工具组合")
            unknown_tools = [tool for tool in chain if tool not in allowed_tools]
            if unknown_tools:
                raise ValueError(f"flow {flow_name} 含未知工具：{unknown_tools}")
            pool.update(chain)
        flow_pools[flow_name] = pool
    return flow_pools


def build_single_distractor_pools(flow_pools, allowed_tools):
    """单意图优先使用非 complex flow 中的同类工具。"""
    distractor_pools = {}
    for gold_tool in allowed_tools:
        related_flows = [
            flow_name
            for flow_name, pool in flow_pools.items()
            if flow_name != "complex_agent_flow" and gold_tool in pool
        ]
        if not related_flows:
            related_flows = [
                flow_name
                for flow_name, pool in flow_pools.items()
                if gold_tool in pool
            ]

        related_tools = set()
        for flow_name in related_flows:
            related_tools.update(flow_pools[flow_name])
        related_tools.discard(gold_tool)
        if not related_tools:
            raise ValueError(f"工具 {gold_tool} 没有可用的同类干扰工具")
        distractor_pools[gold_tool] = sorted(related_tools)
    return distractor_pools


def classify_multi_flow(gold_tools, flow_pools):
    """找出能完整覆盖 GT 工具的唯一业务 flow。"""
    gold_set = set(gold_tools)
    matching_flows = [
        flow_name
        for flow_name, pool in flow_pools.items()
        if gold_set.issubset(pool)
    ]
    regular_flows = [
        flow_name
        for flow_name in matching_flows
        if flow_name != "complex_agent_flow"
    ]
    candidates = regular_flows or matching_flows
    if len(candidates) != 1:
        raise ValueError(
            f"无法为 GT 工具 {gold_tools} 唯一确定同类 flow：{candidates}"
        )
    return candidates[0]


def attach_multi_flows(records, flow_pools):
    return [
        {
            **record,
            "flow": classify_multi_flow(record["gold_tools"], flow_pools),
        }
        for record in records
    ]


def render_tool_catalog(candidate_tools, tools_config):
    blocks = []
    for index, tool_name in enumerate(candidate_tools, start=1):
        config = tools_config[tool_name]
        blocks.append(
            f"{index}. 工具名：{tool_name}\n"
            f"   基本描述：{config['description']}\n"
            f"   详细说明：{config['enhanced_description']}"
        )
    return "\n\n".join(blocks)


def make_built_sample(record, candidate_tools, sample_kind, tools_config):
    gold_tools = record["gold_tools"].copy()
    alpaca_record = {
        "instruction": INSTRUCTION,
        "input": (
            "【候选工具】\n"
            f"{render_tool_catalog(candidate_tools, tools_config)}\n\n"
            "【用户问题】\n"
            f"{record['question']}"
        ),
        "output": json.dumps(
            {"工具": gold_tools},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }
    return {
        "alpaca": alpaca_record,
        "question": record["question"],
        "gold_tools": gold_tools,
        "candidate_tools": candidate_tools.copy(),
        "sample_kind": sample_kind,
        "flow": record.get("flow"),
    }


def shuffled_candidates(tools, rng):
    candidates = tools.copy()
    rng.shuffle(candidates)
    return candidates


def allocate_balanced_targets(total_count, tool_names):
    """将目标数尽量平均地分配给每个 GT 工具。"""
    base_count, remainder = divmod(total_count, len(tool_names))
    return {
        tool_name: base_count + (1 if index < remainder else 0)
        for index, tool_name in enumerate(tool_names)
    }


def generate_unique_single_variant(
    record,
    distractor_pool,
    used_candidate_orders,
    rng,
    tools_config,
):
    gold_tool = record["gold_tools"][0]
    max_distractors = min(4, len(distractor_pool))
    for _ in range(200):
        distractor_count = rng.randint(1, max_distractors)
        distractors = rng.sample(distractor_pool, distractor_count)
        candidates = shuffled_candidates([gold_tool, *distractors], rng)
        candidate_key = tuple(candidates)
        if candidate_key in used_candidate_orders:
            continue
        used_candidate_orders.add(candidate_key)
        return make_built_sample(record, candidates, "single", tools_config)
    raise RuntimeError(f"无法为问题生成新的单意图候选组合：{record['question']}")


def generate_single_samples(
    records,
    target_count,
    distractor_pools,
    tools_config,
    rng,
):
    records_by_tool = defaultdict(list)
    for record in records:
        records_by_tool[record["gold_tools"][0]].append(record)

    configured_tools = list(tools_config)
    missing_tools = [tool for tool in configured_tools if tool not in records_by_tool]
    if missing_tools:
        raise ValueError(f"单意图源数据缺少 GT 工具：{missing_tools}")

    per_tool_targets = allocate_balanced_targets(target_count, configured_tools)
    samples = []
    for gold_tool in configured_tools:
        source_records = records_by_tool[gold_tool].copy()
        target_for_tool = per_tool_targets[gold_tool]
        selected_records = []

        # 每个原始问题先使用一次，不足部分再循环产生新候选变体。
        while len(selected_records) < target_for_tool:
            current_round = source_records.copy()
            rng.shuffle(current_round)
            remaining = target_for_tool - len(selected_records)
            selected_records.extend(current_round[:remaining])

        used_by_question = defaultdict(set)
        for record in selected_records:
            samples.append(generate_unique_single_variant(
                record=record,
                distractor_pool=distractor_pools[gold_tool],
                used_candidate_orders=used_by_question[record["question"]],
                rng=rng,
                tools_config=tools_config,
            ))

    rng.shuffle(samples)
    return samples, per_tool_targets


def random_nonidentity_permutation(tools, used_orders, rng):
    original_order = tuple(tools)
    if len(used_orders) >= math.factorial(len(tools)) - 1:
        return None

    for _ in range(300):
        candidate_order = tools.copy()
        rng.shuffle(candidate_order)
        candidate_key = tuple(candidate_order)
        if candidate_key == original_order or candidate_key in used_orders:
            continue
        used_orders.add(candidate_key)
        return candidate_order

    for candidate_key in permutations(tools):
        if candidate_key == original_order or candidate_key in used_orders:
            continue
        used_orders.add(candidate_key)
        return list(candidate_key)
    return None


def generate_multi_shuffle_samples(records, target_count, tools_config, rng):
    if target_count < len(records):
        raise ValueError(
            f"多意图排序目标数 {target_count} 小于去重后源数据数 {len(records)}，"
            "无法保证每个原始问题至少使用一次"
        )

    samples = []
    used_by_question = defaultdict(set)
    first_round = records.copy()
    rng.shuffle(first_round)

    # 第一轮覆盖所有原始多意图问题。
    for record in first_round:
        candidates = random_nonidentity_permutation(
            record["gold_tools"],
            used_by_question[record["question"]],
            rng,
        )
        if candidates is None:
            raise RuntimeError(f"无法打乱多意图 GT：{record['gold_tools']}")
        samples.append(make_built_sample(
            record, candidates, "multi_shuffle", tools_config
        ))

    # 后续轮次只生成尚未用过的新排列。
    while len(samples) < target_count:
        round_records = records.copy()
        rng.shuffle(round_records)
        generated_in_round = 0
        for record in round_records:
            if len(samples) >= target_count:
                break
            candidates = random_nonidentity_permutation(
                record["gold_tools"],
                used_by_question[record["question"]],
                rng,
            )
            if candidates is None:
                continue
            samples.append(make_built_sample(
                record, candidates, "multi_shuffle", tools_config
            ))
            generated_in_round += 1
        if generated_in_round == 0:
            raise RuntimeError("多意图 GT 的可用新排列已耗尽，无法达到目标数")
    return samples


def generate_multi_hard_samples(
    records,
    target_count,
    flow_pools,
    tools_config,
    rng,
):
    eligible_records = []
    for record in records:
        distractor_pool = flow_pools[record["flow"]] - set(record["gold_tools"])
        available_slots = MAX_CANDIDATE_TOOLS - len(record["gold_tools"])
        if available_slots >= 1 and distractor_pool:
            eligible_records.append(record)

    if target_count > len(eligible_records):
        raise ValueError(
            f"可增加干扰工具的多意图源数据只有 {len(eligible_records)} 条，"
            f"不足以不重复抽取 {target_count} 条"
        )

    selected_records = rng.sample(eligible_records, target_count)
    samples = []
    for record in selected_records:
        gold_tools = record["gold_tools"]
        distractor_pool = sorted(flow_pools[record["flow"]] - set(gold_tools))
        max_distractors = min(
            3,
            MAX_CANDIDATE_TOOLS - len(gold_tools),
            len(distractor_pool),
        )
        distractor_count = rng.randint(1, max_distractors)
        distractors = rng.sample(distractor_pool, distractor_count)
        candidates = shuffled_candidates([*gold_tools, *distractors], rng)
        samples.append(make_built_sample(
            record, candidates, "multi_hard", tools_config
        ))
    return samples, len(eligible_records)


def validate_built_samples(
    single_samples,
    multi_shuffle_samples,
    multi_hard_samples,
    expected_counts,
    allowed_tools,
):
    groups = {
        "single": single_samples,
        "multi_shuffle": multi_shuffle_samples,
        "multi_hard": multi_hard_samples,
    }
    for group_name, samples in groups.items():
        if len(samples) != expected_counts[group_name]:
            raise ValueError(
                f"{group_name} 样本数错误：{len(samples)} != {expected_counts[group_name]}"
            )

        serialized_records = []
        for index, sample in enumerate(samples, start=1):
            gold_tools = sample["gold_tools"]
            candidates = sample["candidate_tools"]

            if not 2 <= len(candidates) <= MAX_CANDIDATE_TOOLS:
                raise ValueError(f"{group_name} 第 {index} 条的候选数不在 2～5 范围")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"{group_name} 第 {index} 条含重复候选工具")
            if any(tool not in allowed_tools for tool in candidates):
                raise ValueError(f"{group_name} 第 {index} 条含未知候选工具")
            if not set(gold_tools).issubset(candidates):
                raise ValueError(f"{group_name} 第 {index} 条的候选工具缺少 GT")

            parsed_output = json.loads(sample["alpaca"]["output"])
            if parsed_output != {"工具": gold_tools}:
                raise ValueError(f"{group_name} 第 {index} 条的 output 与 GT 不一致")

            if group_name == "single":
                if len(gold_tools) != 1 or len(candidates) <= len(gold_tools):
                    raise ValueError(f"单意图第 {index} 条没有同类干扰工具")
            elif group_name == "multi_shuffle":
                if set(candidates) != set(gold_tools):
                    raise ValueError(f"多意图排序第 {index} 条含有非 GT 工具")
                if candidates == gold_tools:
                    raise ValueError(f"多意图排序第 {index} 条未改变 GT 顺序")
            elif group_name == "multi_hard":
                if len(candidates) <= len(gold_tools):
                    raise ValueError(f"多意图困难第 {index} 条没有新增干扰工具")

            serialized_records.append(json.dumps(
                sample["alpaca"], ensure_ascii=False, sort_keys=True
            ))

        if len(serialized_records) != len(set(serialized_records)):
            raise ValueError(f"{group_name} 中存在完全重复的 SFT 样本")


def counter_to_dict(counter):
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def build_stats(
    args,
    source_stats,
    correction_counter,
    repeated_gt_counter,
    skipped_source_counter,
    single_samples,
    multi_shuffle_samples,
    multi_hard_samples,
    per_tool_targets,
    hard_eligible_count,
):
    multi_samples = [*multi_shuffle_samples, *multi_hard_samples]
    merged_samples = [*single_samples, *multi_samples]
    return {
        "seed": args.seed,
        "source_files": {
            "single": str(args.single_input),
            "multi": str(args.multi_input),
            "tools": str(args.tools),
            "combinations": str(args.combinations),
        },
        "source_counts": source_stats,
        "in_memory_tool_name_corrections": dict(correction_counter),
        "in_memory_repeated_gt_removals": dict(repeated_gt_counter),
        "skipped_source_records": dict(skipped_source_counter),
        "target_counts": {
            "single": args.single_count,
            "multi_shuffle": args.multi_shuffle_count,
            "multi_hard": args.multi_hard_count,
            "multi_total": args.multi_shuffle_count + args.multi_hard_count,
            "total": args.single_count + args.multi_shuffle_count + args.multi_hard_count,
        },
        "generated_counts": {
            "single": len(single_samples),
            "multi_shuffle": len(multi_shuffle_samples),
            "multi_hard": len(multi_hard_samples),
            "multi_total": len(multi_samples),
            "total": len(merged_samples),
        },
        "single_gt_tool_distribution": per_tool_targets,
        "single_candidate_count_distribution": counter_to_dict(Counter(
            len(sample["candidate_tools"]) for sample in single_samples
        )),
        "multi_shuffle_gt_length_distribution": counter_to_dict(Counter(
            len(sample["gold_tools"]) for sample in multi_shuffle_samples
        )),
        "multi_hard_gt_length_distribution": counter_to_dict(Counter(
            len(sample["gold_tools"]) for sample in multi_hard_samples
        )),
        "multi_hard_candidate_count_distribution": counter_to_dict(Counter(
            len(sample["candidate_tools"]) for sample in multi_hard_samples
        )),
        "multi_hard_eligible_source_count": hard_eligible_count,
        "validation": {
            "all_candidate_lists_include_gt": True,
            "all_candidate_lists_have_2_to_5_tools": True,
            "all_multi_shuffle_orders_differ_from_gt": True,
            "all_multi_hard_samples_include_distractors": True,
            "duplicate_sft_records_within_each_group": 0,
        },
    }


def ensure_output_paths_available(output_paths, overwrite):
    existing_paths = [path for path in output_paths if path.exists()]
    if existing_paths and not overwrite:
        path_list = "\n".join(f"- {path}" for path in existing_paths)
        raise FileExistsError(
            "以下输出文件已存在，为避免覆盖未执行写入：\n"
            f"{path_list}\n"
            "确认需要重建后，可显式增加 --overwrite。"
        )


def write_jsonl(path, samples):
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample["alpaca"], ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if min(
        args.single_count,
        args.multi_shuffle_count,
        args.multi_hard_count,
    ) < 1:
        raise ValueError("single-count、multi-shuffle-count 和 multi-hard-count 必须大于 0")

    output_paths = {
        "single": args.output_dir / SINGLE_OUTPUT_NAME,
        "multi": args.output_dir / MULTI_OUTPUT_NAME,
        "merged": args.output_dir / MERGED_OUTPUT_NAME,
        "stats": args.output_dir / STATS_OUTPUT_NAME,
    }
    ensure_output_paths_available(output_paths.values(), args.overwrite)

    tools_config = load_json(args.tools)
    validate_tools_config(tools_config)
    allowed_tools = set(tools_config)
    combinations = load_json(args.combinations)
    flow_pools = build_flow_pools(combinations, allowed_tools)
    single_distractor_pools = build_single_distractor_pools(
        flow_pools, allowed_tools
    )

    correction_counter = Counter()
    repeated_gt_counter = Counter()
    skipped_source_counter = Counter()
    raw_single_records = load_source_jsonl(
        args.single_input,
        "single",
        allowed_tools,
        correction_counter,
        repeated_gt_counter,
        skipped_source_counter,
    )
    raw_multi_records = load_source_jsonl(
        args.multi_input,
        "multi",
        allowed_tools,
        correction_counter,
        repeated_gt_counter,
        skipped_source_counter,
    )
    single_records, single_duplicate_count = deduplicate_records(raw_single_records)
    multi_records, multi_duplicate_count = deduplicate_records(raw_multi_records)
    multi_records = attach_multi_flows(multi_records, flow_pools)

    source_stats = {
        "single_loaded": len(raw_single_records),
        "single_unique": len(single_records),
        "single_duplicates_removed": single_duplicate_count,
        "multi_loaded": len(raw_multi_records),
        "multi_unique": len(multi_records),
        "multi_duplicates_removed": multi_duplicate_count,
        "multi_gt_length_distribution": counter_to_dict(Counter(
            len(record["gold_tools"]) for record in multi_records
        )),
        "multi_flow_distribution": counter_to_dict(Counter(
            record["flow"] for record in multi_records
        )),
    }

    rng = random.Random(args.seed)
    single_samples, per_tool_targets = generate_single_samples(
        records=single_records,
        target_count=args.single_count,
        distractor_pools=single_distractor_pools,
        tools_config=tools_config,
        rng=rng,
    )
    multi_shuffle_samples = generate_multi_shuffle_samples(
        records=multi_records,
        target_count=args.multi_shuffle_count,
        tools_config=tools_config,
        rng=rng,
    )
    multi_hard_samples, hard_eligible_count = generate_multi_hard_samples(
        records=multi_records,
        target_count=args.multi_hard_count,
        flow_pools=flow_pools,
        tools_config=tools_config,
        rng=rng,
    )

    validate_built_samples(
        single_samples=single_samples,
        multi_shuffle_samples=multi_shuffle_samples,
        multi_hard_samples=multi_hard_samples,
        expected_counts={
            "single": args.single_count,
            "multi_shuffle": args.multi_shuffle_count,
            "multi_hard": args.multi_hard_count,
        },
        allowed_tools=allowed_tools,
    )

    multi_samples = [*multi_shuffle_samples, *multi_hard_samples]
    rng.shuffle(multi_samples)
    merged_samples = [*single_samples, *multi_samples]
    rng.shuffle(merged_samples)

    stats = build_stats(
        args=args,
        source_stats=source_stats,
        correction_counter=correction_counter,
        repeated_gt_counter=repeated_gt_counter,
        skipped_source_counter=skipped_source_counter,
        single_samples=single_samples,
        multi_shuffle_samples=multi_shuffle_samples,
        multi_hard_samples=multi_hard_samples,
        per_tool_targets=per_tool_targets,
        hard_eligible_count=hard_eligible_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_paths["single"], single_samples)
    write_jsonl(output_paths["multi"], multi_samples)
    write_jsonl(output_paths["merged"], merged_samples)
    with output_paths["stats"].open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print("数据集构建完成：")
    print(f"- 单意图：{len(single_samples)} 条 -> {output_paths['single']}")
    print(f"- 多意图：{len(multi_samples)} 条 -> {output_paths['multi']}")
    print(f"- 合并训练集：{len(merged_samples)} 条 -> {output_paths['merged']}")
    print(f"- 统计报告：{output_paths['stats']}")


if __name__ == "__main__":
    main()
