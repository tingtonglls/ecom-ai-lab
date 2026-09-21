import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sentence_transformers import CrossEncoder


def find_project_root(start_path: Path | None = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()

    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "data").exists() and (path / "dataset").exists():
            return path

    raise FileNotFoundError(
        f"无法定位项目根目录，请确认脚本位于仓库内。起始路径：{current}"
    )


# =========================
# 配置
# =========================

MODEL_CONFIG = {
    "bge-reranker": "BAAI/bge-reranker-base",
    "qwen3-reranker": "Qwen/Qwen3-Reranker-0.6B"
}

TOP_K_LIST = [3, 4, 5]


# =========================
# 读取工具描述
# =========================

def load_tools(tools_path: Path):
    """
    返回：
    {
        tool_name: enhanced_description
    }
    """

    with open(tools_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tool_description_map = {}

    for tool_name, tool_info in tools.items():
        tool_description_map[tool_name] = tool_info["enhanced_description"]

    return tool_description_map


# =========================
# 读取Hybrid候选结果
# =========================

def load_candidate_results(candidate_path: Path):
    """
    返回DataFrame

    兼容两种候选列命名格式：
    - candidate1 ... candidate10
    - candidate_1 ... candidate_10
    """

    df = pd.read_csv(candidate_path)

    query_col = None
    for candidate in ["query", "问题"]:
        if candidate in df.columns:
            query_col = candidate
            break

    if query_col is None:
        raise ValueError("缺少字段：query/问题")

    if query_col != "query":
        df = df.rename(columns={query_col: "query"})

    candidate_columns = []

    for i in range(1, 11):
        for alias in [f"candidate_{i}", f"candidate{i}"]:
            if alias in df.columns:
                df[f"candidate_{i}"] = df[alias]
                candidate_columns.append(f"candidate_{i}")
                break
        else:
            raise ValueError(f"缺少字段：candidate_{i}")

    return df


# =========================
# 读取Ground Truth
# =========================

def load_gold_labels(test_path: Path):
    """
    返回：

    {
        query:
            set(tool_names)
    }

    兼容 query/tools 和 问题/工具 两种字段名。
    """

    gold_map = {}

    with open(test_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)

            query = sample.get("query") or sample.get("问题")
            tools = sample.get("tools") or sample.get("工具")

            if query is None or tools is None:
                continue

            if isinstance(tools, str):
                tools = [tools]

            gold_map[query] = set(tools)

    return gold_map


# =========================
# 加载Reranker
# =========================

def resolve_model_path(model_name_or_path: str) -> str:
    """将内置模型别名解析成 Hugging Face 名称，本地路径则原样返回。"""
    return MODEL_CONFIG.get(model_name_or_path, model_name_or_path)


def load_reranker(model_name_or_path: str):
    """
    加载Cross Encoder模型
    """

    model = CrossEncoder(
        resolve_model_path(model_name_or_path),
        trust_remote_code=True
    )

    return model


# =========================
# 评估指标
# =========================

def is_all_hit_at_k(gold_tools: set, reranked_tools: list, k: int) -> bool:
    """
    计算 AllHit@k。

    对于多意图问题，若 gold 工具数大于 k，则该样本对当前 k 来说不可达成，
    不参与该 k 的命中率统计，只保留为“不可达成”。
    """

    if k <= 0:
        return False

    if len(gold_tools) > k:
        return False

    topk_tools = set(reranked_tools[:k])
    return gold_tools.issubset(topk_tools)


# =========================
# 单条Query重排
# =========================

def rerank_one_query(
        query: str,
        candidate_tools: list,
        tool_description_map: dict,
        reranker
):
    """
    参数
    ----
    query

    candidate_tools:
        Top10工具名列表

    返回
    ----
    reranked_tools:
        重排后的工具列表
    """

    sentence_pairs = []

    valid_tools = []

    for tool_name in candidate_tools:

        if pd.isna(tool_name):
            continue

        tool_name = str(tool_name).strip()

        if tool_name not in tool_description_map:
            continue

        description = tool_description_map[tool_name]

        sentence_pairs.append(
            (
                query,
                description
            )
        )

        valid_tools.append(tool_name)

    if len(valid_tools) == 0:
        return []

    scores = reranker.predict(
        sentence_pairs,
        batch_size=32,
        show_progress_bar=False
    )

    rerank_results = list(
        zip(
            valid_tools,
            scores
        )
    )

    rerank_results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    reranked_tools = [
        tool
        for tool, score in rerank_results
    ]

    return reranked_tools

# =========================
# 单模型评测
# =========================

def evaluate_one_model(
        model_label: str,
        model_name_or_path: str,
        candidate_df: pd.DataFrame,
        gold_map: dict,
        tool_description_map: dict,
        output_dir: Path
):
    """
    使用一个Reranker模型进行评测
    """

    print("=" * 60)
    print(f"Evaluating: {model_label}")
    print(f"Model path: {resolve_model_path(model_name_or_path)}")

    reranker = load_reranker(model_name_or_path)

    hit_count = {k: 0 for k in TOP_K_LIST}
    valid_count = {k: 0 for k in TOP_K_LIST}

    total_count = 0

    result_rows = []

    candidate_columns = [
        f"candidate_{i}"
        for i in range(1, 11)
    ]

    for _, row in candidate_df.iterrows():

        query = row["query"]

        if query not in gold_map:
            continue

        gold_tools = gold_map[query]

        candidate_tools = [
            row[col]
            for col in candidate_columns
        ]

        reranked_tools = rerank_one_query(
            query=query,
            candidate_tools=candidate_tools,
            tool_description_map=tool_description_map,
            reranker=reranker
        )

        total_count += 1

        result = {
            "query": query,
            "gold_tools": json.dumps(
                sorted(gold_tools),
                ensure_ascii=False
            ),
            "reranked_tools": json.dumps(
                reranked_tools,
                ensure_ascii=False
            )
        }

        for k in TOP_K_LIST:

            eligible = len(gold_tools) <= k

            all_hit = is_all_hit_at_k(
                gold_tools=gold_tools,
                reranked_tools=reranked_tools,
                k=k,
            )

            if eligible:
                valid_count[k] += 1

                if all_hit:
                    hit_count[k] += 1

            result[f"EligibleForAllHit@{k}"] = int(eligible)
            result[f"AllHit@{k}"] = int(all_hit) if eligible else 0

        result_rows.append(result)

    result_df = pd.DataFrame(result_rows)

    result_path = output_dir / f"{model_label}_detail.csv"

    result_df.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {
        "model": model_label,
        "model_path": resolve_model_path(model_name_or_path),
        "total_queries": total_count
    }

    print()

    for k in TOP_K_LIST:

        if valid_count[k] == 0:
            all_hit_rate = 0.0
        else:
            all_hit_rate = hit_count[k] / valid_count[k]

        summary[f"AllHit@{k}"] = all_hit_rate
        summary[f"ValidQueriesForAllHit@{k}"] = valid_count[k]

        print(
            f"All-Hit@{k}: {all_hit_rate:.4f}"
        )
        print(
            f"Valid queries for AllHit@{k}: {valid_count[k]}"
        )

    return summary


def parse_model_specs(model_specs: list[str] | None) -> list[tuple[str, str]]:
    """解析 ``标签=模型路径``；内置别名也可单独传入。"""
    if not model_specs:
        return [(name, path) for name, path in MODEL_CONFIG.items()]

    parsed = []
    for spec in model_specs:
        if "=" in spec:
            label, model_path = spec.split("=", 1)
        else:
            label, model_path = spec, resolve_model_path(spec)

        label = label.strip()
        model_path = model_path.strip()
        if not label or not model_path:
            raise ValueError(f"无效的模型参数：{spec}")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", label):
            raise ValueError(f"模型标签只能包含字母、数字、点、下划线和连字符：{label}")
        parsed.append((label, model_path))

    labels = [label for label, _ in parsed]
    if len(labels) != len(set(labels)):
        raise ValueError("模型标签不能重复")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评测 baseline 或本地微调 reranker")
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help=(
            "可重复传入。格式为 标签=模型名或本地路径；也可直接传内置别名 "
            "bge-reranker/qwen3-reranker"
        ),
    )
    parser.add_argument("--candidate-path", type=Path, help="固定的 Top10 候选 CSV")
    parser.add_argument("--test-path", type=Path, help="测试集 JSONL")
    parser.add_argument("--tools-path", type=Path, help="工具描述 JSON")
    parser.add_argument("--output-dir", type=Path, help="评测输出根目录")
    return parser.parse_args()


def update_summary_csv(summary_path: Path, summaries: list[dict]) -> None:
    """将本次评测结果按模型标签更新到统一的汇总文件。"""
    new_df = pd.DataFrame(summaries)
    if summary_path.exists():
        existing_df = pd.read_csv(summary_path, encoding="utf-8-sig")
        summary_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        summary_df = new_df

    summary_df = summary_df.drop_duplicates(subset=["model"], keep="last")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

# =========================
# 主函数
# =========================

def main():
    args = parse_args()
    base_dir = find_project_root()
    model_specs = parse_model_specs(args.models)

    tools_path = args.tools_path or (
        base_dir /
        "configs" /
        "tools.json"
    )

    candidate_path = args.candidate_path or (
        base_dir /
        "outputs" /
        "rag" /
        "rerank_candidates" /
        "weighted_fusion_top10.csv"
    )

    test_path = args.test_path or (
        base_dir /
        "dataset" /
        "tool_calling_trace" /
        "test" /
        "intent_test.jsonl"
    )

    output_root = args.output_dir or (
        base_dir /
        "outputs" /
        "rag" /
        "reranker"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading tool descriptions...")
    tool_description_map = load_tools(
        tools_path
    )

    print("Loading retrieval candidates...")
    candidate_df = load_candidate_results(
        candidate_path
    )

    print("Loading ground truth...")
    gold_map = load_gold_labels(
        test_path
    )

    summary_results = []

    for model_label, model_name_or_path in model_specs:

        summary = evaluate_one_model(
            model_label=model_label,
            model_name_or_path=model_name_or_path,
            candidate_df=candidate_df,
            gold_map=gold_map,
            tool_description_map=tool_description_map,
            output_dir=output_root
        )

        summary_results.append(summary)

    summary_path = output_root / "reranker_summary.csv"
    update_summary_csv(summary_path, summary_results)

    print()
    print("=" * 60)
    print("Reranker evaluation completed.")
    print(f"Detail results saved to: {output_root}")
    print(f"Summary updated at: {summary_path}")


if __name__ == "__main__":
    main()
