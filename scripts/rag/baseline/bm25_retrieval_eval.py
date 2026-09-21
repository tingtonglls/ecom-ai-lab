import csv
import json
import time
from pathlib import Path
from typing import Optional

import jieba
import joblib


def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()

    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "data").exists() and (path / "dataset").exists():
            return path

    raise FileNotFoundError(
        f"无法定位项目根目录，请确认脚本位于仓库内。起始路径：{current}"
    )


BM25_CONFIG = {
    "vector_dir": "data/vector_store/bm25"
}


TOP_K_LIST = [5, 8, 10, 12, 15]


def load_test_data(test_path: Path):
    data = []

    with test_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)

            query = item.get("问题")
            gold_tools = item.get("工具")

            if isinstance(gold_tools, str):
                gold_tools = [gold_tools]

            if not query or not gold_tools:
                continue

            data.append(
                {
                    "query": query,
                    "gold_tools": gold_tools
                }
            )

    return data


def load_metadata(metadata_path: Path):

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    index_to_tool = {}

    for item in metadata:
        index_to_tool[item["index"]] = item["tool_name"]

    return index_to_tool


def load_tokenizer(base_dir: Path):

    config_path = (
        base_dir
        / "configs"
        / "bm25_tokenizer_config.json"
    )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_name = config["tokenizer"]

    if tokenizer_name != "jieba":
        raise NotImplementedError(
            f"暂不支持 tokenizer：{tokenizer_name}"
        )

    user_dict = config.get("user_dict")

    if user_dict:

        user_dict_path = base_dir / user_dict

        if user_dict_path.exists():
            jieba.load_userdict(str(user_dict_path))

    return jieba


def evaluate_bm25(test_data, base_dir):

    vector_dir = (
        base_dir
        / BM25_CONFIG["vector_dir"]
    )

    bm25_path = vector_dir / "tool_bm25.pkl"

    metadata_path = (
        vector_dir
        / "tool_metadata.json"
    )

    print("\n========== 评测 BM25 ==========")

    print("加载 BM25 Index...")

    bm25 = joblib.load(bm25_path)

    print("加载 metadata...")

    index_to_tool = load_metadata(metadata_path)

    print("初始化 Tokenizer...")

    tokenizer = load_tokenizer(base_dir)

    hit_counts = {
        k: 0
        for k in TOP_K_LIST
    }

    total_latency = 0.0

    for item in test_data:

        query = item["query"]

        gold_tools = set(item["gold_tools"])

        start_time = time.time()

        query_tokens = list(
            tokenizer.cut(
                query,
                cut_all=False
            )
        )

        scores = bm25.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        latency = time.time() - start_time

        total_latency += latency

        retrieved_tools = [
            index_to_tool[idx]
            for idx in ranked_indices
        ]

        for k in TOP_K_LIST:

            top_k_tools = set(
                retrieved_tools[:k]
            )

            if gold_tools.issubset(top_k_tools):
                hit_counts[k] += 1

    total = len(test_data)

    result = {
        "model": "BM25",
        "total": total,
        "avg_latency_ms": round(
            total_latency / total * 1000,
            4
        )
    }

    for k in TOP_K_LIST:
        result[f"all_hit@{k}"] = round(
            hit_counts[k] / total,
            4
        )

    return result


def main():

    base_dir = find_project_root()

    test_path = (
        base_dir
        / "dataset"
        / "tool_calling_trace"
        / "test"
        / "intent_test.jsonl"
    )

    output_dir = (
        base_dir
        / "outputs"
        / "rag"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / "bm25_retrieval_summary.csv"
    )

    print(f"读取测试集：{test_path}")

    test_data = load_test_data(test_path)

    print(f"测试样本数量：{len(test_data)}")

    result = evaluate_bm25(
        test_data=test_data,
        base_dir=base_dir
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )

        writer.writeheader()
        writer.writerow(result)

    print("\n========== BM25 Retrieval 评测结果 ==========")

    print(result)

    print(f"\n结果已保存到：{output_path}")


if __name__ == "__main__":
    main()