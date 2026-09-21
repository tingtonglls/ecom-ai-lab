import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import jieba
import joblib
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
    },
    "my-bge-base-bs16_ep2_lr1e-5": {
        "model_path": "model/my-bge-base/bs16_ep2_lr1e-5",
        "vector_dir": "data/vector_store/my-bge-base/bs16_ep2_lr1e-5",
        "output_name": "my-bge-base_bs16_ep2_lr1e-5"
    },
    "my-bge-base-bs16_ep3_lr1e-5": {
        "model_path": "model/my-bge-base/bs16_ep3_lr1e-5",
        "vector_dir": "data/vector_store/my-bge-base/bs16_ep3_lr1e-5",
        "output_name": "my-bge-base_bs16_ep3_lr1e-5"
    },
    "my-bge-base-bs16_ep3_lr1e-5_hardneg": {
        "model_path": "model/my-bge-base/bs16_ep3_lr1e-5_hardneg",
        "vector_dir": "data/vector_store/my-bge-base/bs16_ep3_lr1e-5_hardneg",
        "output_name": "my-bge-base_bs16_ep3_lr1e-5_hardneg"
    },
    "my-bge-base-bs16_ep1_lr1e-5": {
        "model_path": "model/my-bge-base/bs16_ep1_lr1e-5",
        "vector_dir": "data/vector_store/my-bge-base/bs16_ep1_lr1e-5",
        "output_name": "my-bge-base_bs16_ep1_lr1e-5"
    }

}

BM25_CONFIG = {
    "vector_dir": "data/vector_store/bm25"
}

TOP_K_LIST = [5, 8, 10, 12, 15]


def get_model_output_name(model_key: str) -> str:
    return MODEL_CONFIG[model_key].get("output_name", model_key)


def load_test_data(test_path: Path):
    # 读取评测数据，统一成 {query, gold_tools} 的格式。
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
                    "gold_tools": gold_tools,
                }
            )

    return data


def load_metadata(metadata_path: Path):
    # 读取索引元数据，将 index 映射回工具名。
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    index_to_tool = {}

    for item in metadata:
        index_to_tool[item["index"]] = item["tool_name"]

    return index_to_tool


def resolve_model_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def load_tokenizer(base_dir: Path):
    # 加载 BM25 分词器，当前实现使用 jieba。
    config_path = base_dir / "configs" / "bm25_tokenizer_config.json"

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_name = config["tokenizer"]

    if tokenizer_name != "jieba":
        raise NotImplementedError(f"暂不支持 tokenizer：{tokenizer_name}")

    user_dict = config.get("user_dict")

    if user_dict:
        user_dict_path = base_dir / user_dict

        if user_dict_path.exists():
            jieba.load_userdict(str(user_dict_path))

    return jieba


def normalize_scores(scores: List[float]):
    # 将一组分数映射到[0, 1]区间，便于和 dense 的相似度分数做加权融合。
    if not scores:
        return []
    
    # 采用MinMax归一化
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [0.0 for _ in scores]

    return [
        (score - min_score) / (max_score - min_score)
        for score in scores
    ]


def build_hybrid_candidates(
    dense_hits: List[Tuple[str, float]],
    sparse_hits: List[Tuple[str, float]],
    method: str,
    dense_top_k: int,
    sparse_top_k: int,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
):
    # 将 dense 和 sparse 的检索结果按照指定策略融合成一个候选列表。
    if method == "union":
        dense_candidates = [tool for tool, _ in dense_hits[:dense_top_k]]
        sparse_candidates = [tool for tool, _ in sparse_hits[:sparse_top_k]]
        merged = []
        seen = set()

        for tool in dense_candidates + sparse_candidates:
            if tool not in seen:
                merged.append(tool)
                seen.add(tool)

        return merged

    if method == "weighted_fusion":
        # 法一：分别对 dense 和 sparse 的分数做 min-max 归一化，再按权重融合。
        # 法二：只对 sparse 的分数做 min-max 归一化，再按权重融合
        scores = {}

        dense_scores = [score for _, score in dense_hits[:dense_top_k]]
        sparse_scores = [score for _, score in sparse_hits[:sparse_top_k]]

        normalized_dense_scores = normalize_scores(dense_scores)
        normalized_sparse_scores = normalize_scores(sparse_scores)

        for (tool, _), score in zip(dense_hits[:dense_top_k], normalized_dense_scores):
            scores[tool] = scores.get(tool, 0.0) + score * dense_weight

        for (tool, _), score in zip(sparse_hits[:sparse_top_k], normalized_sparse_scores):
            scores[tool] = scores.get(tool, 0.0) + score * sparse_weight

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [tool for tool, _ in ranked]

    if method == "rrf":
        # RRF 会根据每个结果在各自榜单中的排名位置给出加权分数，而不是直接用原始分数。
        scores = {}

        for rank, (tool, _) in enumerate(dense_hits[:dense_top_k], start=1):
            scores[tool] = scores.get(tool, 0.0) + 1.0 / (rank + 60)

        for rank, (tool, _) in enumerate(sparse_hits[:sparse_top_k], start=1):
            scores[tool] = scores.get(tool, 0.0) + 1.0 / (rank + 60)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [tool for tool, _ in ranked]

    raise ValueError(f"不支持的 hybrid 方法：{method}")


def evaluate_hybrid_method(
    method,
    model_key,
    test_data,
    base_dir,
    dense_weight: float = 0.5, # 默认值，可以在终端以输入参数的形式更改
    sparse_weight: float = 0.5,
):
    # 对某一种混合策略和某一个 embedding 模型做完整评测。
    config = MODEL_CONFIG[model_key]

    vector_dir = resolve_model_path(base_dir, config["vector_dir"])
    index_path = vector_dir / "tool.index"
    metadata_path = vector_dir / "tool_metadata.json"

    bm25_dir = resolve_model_path(base_dir, BM25_CONFIG["vector_dir"])
    bm25_path = bm25_dir / "tool_bm25.pkl"
    bm25_metadata_path = bm25_dir / "tool_metadata.json"

    print(f"\n========== 评测混合检索：{method} | {model_key} ==========")
    print(f"Dense vector dir: {vector_dir}")
    print(f"Dense index path: {index_path}")
    print(f"Dense metadata path: {metadata_path}")
    print(f"BM25 dir: {bm25_dir}")

    print("加载 Faiss index...")
    index = faiss.read_index(str(index_path))

    print("加载 dense metadata...")
    dense_index_to_tool = load_metadata(metadata_path)

    print("加载 BM25 Index...")
    bm25 = joblib.load(bm25_path)

    print("加载 sparse metadata...")
    sparse_index_to_tool = load_metadata(bm25_metadata_path)

    print("加载 embedding 模型...")
    model_path = config.get("model_path")
    if model_path:
        model_dir = base_dir / model_path
        if not model_dir.exists():
            raise FileNotFoundError(f"找不到本地模型目录：{model_dir}")
        print(f"从本地加载模型：{model_dir}")
    else:
        model_dir = snapshot_download(config["model_id"])
        print(f"从 ModelScope 下载/读取模型：{model_dir}")
    encoder = SentenceTransformer(str(model_dir), trust_remote_code=True)

    tokenizer = load_tokenizer(base_dir)

    hit_counts = {k: 0 for k in TOP_K_LIST}
    total_latency = 0.0

    for item in test_data:
        query = item["query"]
        gold_tools = set(item["gold_tools"])

        start_time = time.time()

        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # 1) 先做 dense 检索，拿到语义相似度最高的候选工具。
        dense_scores, dense_indices = index.search(
            query_embedding,
            max(TOP_K_LIST),
        )

        dense_hits = [
            (dense_index_to_tool[int(idx)], float(score))
            for score, idx in zip(dense_scores[0], dense_indices[0])
            if int(idx) != -1
        ]

        # 2) 再做 sparse 检索，拿到关键词匹配更强的候选工具。
        query_tokens = list(tokenizer.cut(query, cut_all=False))
        sparse_scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i],
            reverse=True,
        )

        sparse_hits = [
            (sparse_index_to_tool[idx], float(sparse_scores[idx]))
            for idx in ranked_indices[:max(TOP_K_LIST)]
        ]

        # 3) 将两边结果融合，得到一个统一的候选列表。
        hybrid_candidates = build_hybrid_candidates(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            method=method,
            dense_top_k=max(TOP_K_LIST),
            sparse_top_k=max(TOP_K_LIST),
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

        latency = time.time() - start_time
        total_latency += latency

        for k in TOP_K_LIST:
            top_k_tools = set(hybrid_candidates[:k])

            if gold_tools.issubset(top_k_tools):
                hit_counts[k] += 1

    total = len(test_data)

    result = {
        "model": f"{model_key}_{method}",
        "total": total,
        "avg_latency_ms": round(total_latency / total * 1000, 4),
    }

    for k in TOP_K_LIST:
        result[f"all_hit@{k}"] = round(hit_counts[k] / total, 4)

    return result


def main():
    # 主函数：支持按指定方法单独评测，默认评测三种混合方法。
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["union", "weighted_fusion", "rrf"],
        choices=["union", "weighted_fusion", "rrf"],
        help="指定要评测的混合检索方法，默认评测全部三种",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="weighted_fusion 的 dense 权重",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="指定要评测的单个 embedding 模型；不传时评测所有模型",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.5,
        help="weighted_fusion 的 sparse 权重",
    )

    args = parser.parse_args()

    base_dir = find_project_root()

    test_path = base_dir / "dataset" / "tool_calling_trace" / "test" / "intent_test.jsonl"
    output_dir = base_dir / "outputs" / "rag"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取测试集：{test_path}")
    test_data = load_test_data(test_path)

    print(f"测试样本数量：{len(test_data)}")

    methods = args.methods
    results = []

    model_keys = [args.model_key] if args.model_key else list(MODEL_CONFIG.keys())

    for method in methods:
        for model_key in model_keys:
            result = evaluate_hybrid_method(
                method=method,
                model_key=model_key,
                test_data=test_data,
                base_dir=base_dir,
                dense_weight=args.dense_weight,
                sparse_weight=args.sparse_weight,
            )
            results.append(result)

    output_file_name = "hybrid_retrieval_summary.csv"
    if len(methods) == 1:
        output_file_name = f"hybrid_retrieval_summary_{methods[0]}.csv"

    if args.model_key:
        model_label = get_model_output_name(args.model_key)
        if model_label == args.model_key:
            output_path = output_dir / output_file_name
        else:
            output_path = output_dir / f"{model_label}_{output_file_name}"
    else:
        output_path = output_dir / output_file_name

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n========== Hybrid Retrieval 评测结果 ==========")

    for r in results:
        print(r)

    print(f"\n结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
