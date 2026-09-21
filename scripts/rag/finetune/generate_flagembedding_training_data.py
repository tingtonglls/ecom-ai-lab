import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional


def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()
    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "dataset").exists():
            return path
    raise FileNotFoundError(f"无法定位项目根目录，请确认脚本位于仓库内。起始路径：{current}")


def load_tools(tools_path: Path) -> Dict[str, str]:
    with open(tools_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tool_description_map: Dict[str, str] = {}
    for tool_name, tool_info in tools.items():
        desc = tool_info.get("enhanced_description", "").strip()
        if desc:
            tool_description_map[tool_name] = desc
    return tool_description_map


def load_samples(input_path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples


def build_training_example(sample: Dict[str, Any], tool_description_map: Dict[str, str], rng: random.Random, index: int) -> Optional[Dict[str, Any]]:
    query = sample.get("query") or sample.get("问题")
    raw_tools = sample.get("tools") or sample.get("工具")
    if query is None or raw_tools is None:
        return None

    if isinstance(raw_tools, str):
        gold_tools = [raw_tools]
    elif isinstance(raw_tools, list):
        gold_tools = [str(tool) for tool in raw_tools if str(tool).strip()]
    else:
        gold_tools = [str(raw_tools)]

    if not gold_tools:
        return None

    gold_tool_set = set(gold_tools)
    pos = []
    for tool_name in gold_tools:
        desc = tool_description_map.get(tool_name)
        if desc:
            pos.append(desc)

    if not pos:
        return None

    candidate_negative_names = [name for name in tool_description_map.keys() if name not in gold_tool_set]
    if len(candidate_negative_names) < 3:
        return None

    rng.shuffle(candidate_negative_names)
    neg_names = candidate_negative_names[:3]
    neg = []
    for tool_name in neg_names:
        desc = tool_description_map.get(tool_name)
        if desc:
            neg.append(desc)

    if len(neg) < 3:
        return None

    return {
        "id": str(index),
        "query": query,
        "pos": pos,
        "neg": neg,
        "prompt": "Represent this sentence for searching relevant tool descriptions: ",
        # "type": "normal",
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
    output_path = root / "dataset" / "tool_calling_trace" / "train" / "intent_train_flagembedding.jsonl"

    tool_description_map = load_tools(tools_path)
    samples = load_samples(input_path)

    rng = random.Random(42)
    training_rows = []
    for index, sample in enumerate(samples, start=1):
        row = build_training_example(sample, tool_description_map, rng, index)
        if row is not None:
            training_rows.append(row)

    save_jsonl(output_path, training_rows)
    print(f"已生成 {len(training_rows)} 条训练样本")
    print(f"输出文件：{output_path}")


if __name__ == "__main__":
    main()
