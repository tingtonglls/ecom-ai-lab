import csv
import json
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer


def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()

    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "data").exists() and (path / "dataset").exists():
            return path

    raise FileNotFoundError(
        f"无法定位项目根目录，请确认脚本位于仓库内。起始路径：{current}"
    )


MODEL_CONFIG = {
    "bge-base": {
        "model_id": "BAAI/bge-base-zh-v1.5",
        "vector_dir": "data/vector_store/bge-base"
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-zh-v1.5",
        "vector_dir": "data/vector_store/bge-large"
    },
    "qwen3-embedding-0.6b": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "vector_dir": "data/vector_store/qwen3-embedding-0.6b"
    }
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

            if isinstance(gold_tools, str):  #将单个字符串转换成列表，统一单意图和多意图格式
                gold_tools = [gold_tools]

            if not query or not gold_tools:
                continue
            
            #统一数据格式
            data.append({
                "query": query,
                "gold_tools": gold_tools
            })

    return data


def load_metadata(metadata_path: Path):
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    index_to_tool = {}

    for item in metadata:
        index_to_tool[item["index"]] = item["tool_name"]

    return index_to_tool


def evaluate_one_model(model_key, test_data, base_dir):
    config = MODEL_CONFIG[model_key]

    vector_dir = base_dir / config["vector_dir"]
    index_path = vector_dir / "tool.index"
    metadata_path = vector_dir / "tool_metadata.json"

    print(f"\n========== 评测模型：{model_key} ==========")

    print("加载 Faiss index...")
    index = faiss.read_index(str(index_path))

    print("加载 metadata...")
    index_to_tool = load_metadata(metadata_path)

    print("加载 embedding 模型...")
    model_dir = snapshot_download(config["model_id"])
    encoder = SentenceTransformer(
        model_dir,
        trust_remote_code=True
    )

    hit_counts = {k: 0 for k in TOP_K_LIST}
    total_latency = 0.0

    for item in test_data:
        query = item["query"]
        gold_tools = set(item["gold_tools"])  #把工具列表转换成集合

        start_time = time.time()

        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True  #让返回结果是NumPy 数组
        ).astype("float32")

        scores, indices = index.search(
            query_embedding,
            max(TOP_K_LIST)  #只做取K最大值的检索，也就是只做top-15检索，然后分别截取前5，8，10，12，15个
        )

        #耗时包含query embedding编码时间和Faiss检索时间
        latency = time.time() - start_time  #latency是单条query的端到端检索延迟
        total_latency += latency

        retrieved_tools = [
            index_to_tool[int(idx)]
            for idx in indices[0]
            if int(idx) != -1
        ]

        for k in TOP_K_LIST:
            top_k_tools = set(retrieved_tools[:k])  #截取前k个工具

            # 单意图和多意图统一用 all-hit
            if gold_tools.issubset(top_k_tools):
                hit_counts[k] += 1

    total = len(test_data)

    result = {
        "model": model_key,
        "total": total,
        "avg_latency_ms": round(total_latency / total * 1000, 4) #毫秒，保留4位小数
    }

    for k in TOP_K_LIST:
        result[f"all_hit@{k}"] = round(hit_counts[k] / total, 4)

    return result


def main():
    base_dir = find_project_root()

    test_path = (
        base_dir
        / "dataset" / "tool_calling_trace" / "test" / "intent_test.jsonl"
    )

    output_dir = base_dir / "outputs" / "rag"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "dense_retrieval_summary.csv"

    print(f"读取测试集：{test_path}")
    test_data = load_test_data(test_path)

    print(f"测试样本数量：{len(test_data)}")

    results = []

    for model_key in MODEL_CONFIG.keys():
        result = evaluate_one_model(
            model_key=model_key,
            test_data=test_data,
            base_dir=base_dir
        )
        results.append(result)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n========== Dense Retrieval 评测结果 ==========")

    for r in results:
        print(r)

    print(f"\n结果已保存到：{output_path}")


if __name__ == "__main__":
    main()