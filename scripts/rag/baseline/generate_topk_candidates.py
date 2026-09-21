import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Optional

import faiss
import joblib
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


def load_hybrid_module():
    module_path = Path(__file__).resolve().parent / "hybrid_retrieval_eval.py"

    spec = importlib.util.spec_from_file_location(
        "hybrid_retrieval_eval",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        default="weighted_fusion",
        choices=["union", "weighted_fusion", "rrf"],
        help="指定混合检索方法",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="指定候选列表的长度",
    )
    parser.add_argument(
        "--dense-model-key",
        default="qwen3-embedding-0.6b",
        help="指定用于 dense 检索的 embedding 模型",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="weighted_fusion 的 dense 权重",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=0.5,
        help="weighted_fusion 的 sparse 权重",
    )

    args = parser.parse_args()

    hybrid_module = load_hybrid_module()

    base_dir = find_project_root()

    test_path = (
        base_dir
        / "dataset"
        / "tool_calling_trace"
        / "test"
        / "intent_test.jsonl"
    )

    model_label = hybrid_module.get_model_output_name(args.dense_model_key)
    output_dir = base_dir / "outputs" / "rag" / "rerank_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_label == args.dense_model_key:
        output_path = output_dir / f"{args.dense_model_key}_{args.method}_top{args.top_k}.csv"
    else:
        output_path = output_dir / f"{model_label}_{args.method}_top{args.top_k}.csv"

    print(f"读取测试集：{test_path}")
    test_data = hybrid_module.load_test_data(test_path)

    print(f"测试样本数量：{len(test_data)}")

    dense_model_key = args.dense_model_key
    dense_config = hybrid_module.MODEL_CONFIG[dense_model_key]

    dense_vector_dir = base_dir / dense_config["vector_dir"]
    dense_index_path = dense_vector_dir / "tool.index"
    dense_metadata_path = dense_vector_dir / "tool_metadata.json"

    bm25_dir = base_dir / hybrid_module.BM25_CONFIG["vector_dir"]
    bm25_path = bm25_dir / "tool_bm25.pkl"
    bm25_metadata_path = bm25_dir / "tool_metadata.json"

    print("加载 Faiss index...")
    dense_index = faiss.read_index(str(dense_index_path))

    print("加载 dense metadata...")
    dense_index_to_tool = hybrid_module.load_metadata(dense_metadata_path)

    print("加载 BM25 Index...")
    bm25 = joblib.load(bm25_path)

    print("加载 sparse metadata...")
    sparse_index_to_tool = hybrid_module.load_metadata(bm25_metadata_path)

    print("加载 embedding 模型...")
    model_path = dense_config.get("model_path")
    if model_path:
        model_dir = base_dir / model_path
        if not model_dir.exists():
            raise FileNotFoundError(f"找不到本地模型目录：{model_dir}")
        print(f"从本地加载模型：{model_dir}")
    else:
        model_dir = snapshot_download(dense_config["model_id"])
        print(f"从 ModelScope 下载/读取模型：{model_dir}")
    encoder = SentenceTransformer(str(model_dir), trust_remote_code=True)

    tokenizer = hybrid_module.load_tokenizer(base_dir)

    rows = []
    top_k = args.top_k

    for item in test_data:
        query = item["query"]

        query_embedding = encoder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        dense_scores, dense_indices = dense_index.search(
            query_embedding,
            top_k,
        )

        dense_hits = [
            (dense_index_to_tool[int(idx)], float(score))
            for score, idx in zip(dense_scores[0], dense_indices[0])
            if int(idx) != -1
        ]

        query_tokens = list(tokenizer.cut(query, cut_all=False))
        sparse_scores = bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(sparse_scores)),
            key=lambda i: sparse_scores[i],
            reverse=True,
        )[:top_k]

        sparse_hits = [
            (sparse_index_to_tool[idx], float(sparse_scores[idx]))
            for idx in ranked_indices
        ]

        hybrid_candidates = hybrid_module.build_hybrid_candidates(
            dense_hits=dense_hits,
            sparse_hits=sparse_hits,
            method=args.method,
            dense_top_k=top_k,
            sparse_top_k=top_k,
            dense_weight=args.dense_weight,
            sparse_weight=args.sparse_weight,
        )

        candidate_tools = hybrid_candidates[:top_k]
        row = {"query": query}

        for i, tool in enumerate(candidate_tools, start=1):
            row[f"candidate_{i}"] = tool

        rows.append(row)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["query"] + [f"candidate_{i}" for i in range(1, top_k + 1)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
