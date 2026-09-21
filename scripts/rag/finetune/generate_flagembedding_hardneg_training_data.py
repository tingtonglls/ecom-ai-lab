import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()
    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "dataset").exists():
            return path
    raise FileNotFoundError("无法定位项目根目录，请确认脚本位于仓库内。")


def load_tools(tools_path: Path) -> Dict[str, Dict[str, Any]]:
    with open(tools_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_tool_category_map() -> Tuple[Dict[str, str], List[str]]:
    # 这里用 5 类来覆盖 20 个 tool，和“5种MCP功能分类”对齐。
    tool_groups = {
        "order": [
            "get_order_list",
            "get_order_detail",
            "get_order_logistics",
            "can_merge_ship",
        ],
        "aftersale": [
            "create_refund_request",
            "get_return_policy",
            "get_refund_status",
            "create_aftersale_ticket",
        ],
        "shopping": [
            "get_item_info",
            "get_stock",
            "get_similar_items",
            "create_cart_item",
        ],
        "promotion": [
            "get_cart_summary",
            "get_current_promotions",
            "calc_discount",
            "get_available_coupons",
            "apply_coupon",
        ],
        "profile": [
            "get_user_profile",
            "get_browsing_history",
            "get_purchase_history",
        ],
    }
    category_of: Dict[str, str] = {}
    for category, tools in tool_groups.items():
        for tool in tools:
            category_of[tool] = category
    tool_names = list(category_of.keys())
    return category_of, tool_names


def extract_gold_tools(sample: Dict[str, Any]) -> List[str]:
    raw = sample.get("tools") or sample.get("工具")
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(tool) for tool in raw if str(tool).strip()]
    return [str(raw)]


def select_descriptions(tool_names: List[str], tool_descriptions: Dict[str, str], category_of: Dict[str, str], gold_tools: List[str], mode: str, rng: random.Random) -> List[str]:
    gold_set = set(gold_tools)
    picked: List[str] = []
    picked_names: List[str] = []

    if mode == "single":
        # 单意图：同类型 + 跨类型，1:1
        gold_tool = gold_tools[0]
        gold_category = category_of[gold_tool]
        same_pool = [name for name in tool_names if name not in gold_set and category_of[name] == gold_category]
        cross_pool = [name for name in tool_names if name not in gold_set and category_of[name] != gold_category]

        if same_pool:
            same_name = rng.choice(same_pool)
            picked_names.append(same_name)
        if cross_pool:
            cross_name = rng.choice(cross_pool)
            picked_names.append(cross_name)

    elif mode == "multi_partial":
        # 多意图：部分替代
        # 规则是：从多意图 positive 工具集中，保留一部分正确工具不变，
        # 用错误工具替换另一部分；同时尽量让替换后的负样本仍然与原工具链相关，
        # 以形成更难的 hard negative。
        if len(gold_tools) >= 2:
            keep_count = max(1, len(gold_tools) // 2)
            replace_count = len(gold_tools) - keep_count

            keep_names = rng.sample(gold_tools, k=keep_count)
            kept_set = set(keep_names)
            picked_names.extend(keep_names)

            replace_pool: List[str] = []
            for tool in gold_tools:
                if tool in kept_set:
                    continue
                gold_category = category_of[tool]
                candidates = [
                    name
                    for name in tool_names
                    if name not in gold_set and name not in picked_names and category_of[name] == gold_category
                ]
                if candidates:
                    replace_pool.extend(candidates)

            if not replace_pool:
                replace_pool = [
                    name
                    for name in tool_names
                    if name not in gold_set and name not in picked_names
                ]

            if replace_pool:
                replace_names = rng.sample(replace_pool, k=min(replace_count, len(replace_pool)))
                picked_names.extend(replace_names)

            remaining_pool = [name for name in tool_names if name not in gold_set and name not in picked_names]
            if remaining_pool:
                picked_names.append(rng.choice(remaining_pool))

    else:
        # 多意图：纯随机采样 3 个错误工具
        remaining_pool = [name for name in tool_names if name not in gold_set]
        if len(remaining_pool) >= 3:
            picked_names = rng.sample(remaining_pool, 3)
        else:
            picked_names = remaining_pool[:3]

    for name in picked_names:
        desc = tool_descriptions.get(name, "")
        if desc:
            picked.append(desc)

    # 补齐到 2/3 个，避免空缺
    if mode == "single" and len(picked) < 2:
        fallback_pool = [name for name in tool_names if name not in gold_set and name not in picked_names]
        if fallback_pool:
            fallback_name = rng.choice(fallback_pool)
            picked.append(tool_descriptions.get(fallback_name, ""))
    if mode != "single" and len(picked) < 3:
        fallback_pool = [name for name in tool_names if name not in gold_set and name not in picked_names]
        while len(picked) < 3 and fallback_pool:
            fallback_name = rng.choice(fallback_pool)
            fallback_pool.remove(fallback_name)
            picked_names.append(fallback_name)
            desc = tool_descriptions.get(fallback_name, "")
            if desc:
                picked.append(desc)

    return picked[:3]


def build_training_example(sample: Dict[str, Any], tool_descriptions: Dict[str, str], category_of: Dict[str, str], tool_names: List[str], rng: random.Random, index: int, mode: str) -> Optional[Dict[str, Any]]:
    query = sample.get("query") or sample.get("问题")
    gold_tools = extract_gold_tools(sample)
    if query is None or not gold_tools:
        return None

    pos_names = gold_tools
    pos = []
    for tool_name in pos_names:
        desc = tool_descriptions.get(tool_name)
        if desc:
            pos.append(desc)
    if not pos:
        return None

    neg = select_descriptions(tool_names, tool_descriptions, category_of, gold_tools, mode, rng)
    if not neg:
        return None

    return {
        "id": str(index),
        "query": query,
        "pos": pos,
        "neg": neg,
        "prompt": "Represent this sentence for searching relevant tool descriptions: ",
    }


def save_jsonl(output_path: Path, rows: List[Dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    root = find_project_root()
    tools_path = root / "configs" / "tools.json"
    input_path = root / "dataset" / "tool_calling_trace" / "train" / "intent_train.jsonl"
    output_path = root / "dataset" / "tool_calling_trace" / "train" / "intent_train_flagembedding_hardneg.jsonl"

    tools = load_tools(tools_path)
    tool_descriptions = {}
    for tool_name, tool_info in tools.items():
        desc = (tool_info.get("enhanced_description") or "").strip()
        if desc:
            tool_descriptions[tool_name] = desc

    category_of, tool_names = build_tool_category_map()
    samples = load_samples(input_path)

    rng = random.Random(42)
    training_rows: List[Dict[str, Any]] = []
    single_count = 0
    multi_count = 0
    partial_count = 0
    random_count = 0

    multi_counter = 0
    for index, sample in enumerate(samples, start=1):
        gold_tools = extract_gold_tools(sample)
        if len(gold_tools) == 1:
            mode = "single"
            single_count += 1
        else:
            mode = "multi_partial" if multi_counter < 1200 else "multi_random"
            multi_counter += 1
            multi_count += 1
            if mode == "multi_partial":
                partial_count += 1
            else:
                random_count += 1

        row = build_training_example(sample, tool_descriptions, category_of, tool_names, rng, index, mode)
        if row is not None:
            training_rows.append(row)

    save_jsonl(output_path, training_rows)
    print(f"已生成 {len(training_rows)} 条训练样本")
    print(f"单意图样本: {single_count}")
    print(f"多意图样本: {multi_count}，其中 partial: {partial_count}，random: {random_count}")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
