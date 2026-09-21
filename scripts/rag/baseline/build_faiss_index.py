### 离线-构建索引

import argparse
from pathlib import Path
from typing import Optional

import faiss
import numpy as np


# ======================
# 向量库配置
# ======================
def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()

    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "data").exists() and (path / "dataset").exists():
            return path

    raise FileNotFoundError(
        f"无法定位项目根目录，请确认脚本位于仓库内。起始路径：{current}"
    )


VECTOR_STORE_CONFIG = {
    "bge-base": "data/vector_store/bge-base",
    "bge-large": "data/vector_store/bge-large",
    "qwen3-embedding-0.6b": "data/vector_store/qwen3-embedding-0.6b",
    "my-bge-base-bs16_ep2_lr1e-5": "data/vector_store/my-bge-base/bs16_ep2_lr1e-5",
    "my-bge-base-bs16_ep3_lr1e-5": "data/vector_store/my-bge-base/bs16_ep3_lr1e-5",
    "my-bge-base-bs16_ep3_lr1e-5_hardneg": "data/vector_store/my-bge-base/bs16_ep3_lr1e-5_hardneg",
    "my-bge-base-bs16_ep1_lr1e-5": "data/vector_store/my-bge-base/bs16_ep1_lr1e-5"

}   


def build_faiss_index(embedding_path: Path, index_path: Path):
    print(f"读取向量文件：{embedding_path}")

    embeddings = np.load(embedding_path)

    if embeddings.ndim != 2:
        raise ValueError(f"向量矩阵必须是二维数组，但当前 shape={embeddings.shape}")

    embeddings = embeddings.astype("float32")

    num_vectors, dimension = embeddings.shape

    print(f"向量数量：{num_vectors}")
    print(f"向量维度：{dimension}")

    # 使用内积检索。前提：build_embeddings.py 中已经 normalize_embeddings=True
    index = faiss.IndexFlatIP(dimension)

    print("添加向量到 Faiss Index...")

    index.add(embeddings)

    print(f"Faiss Index 中向量数量：{index.ntotal}")

    index_path.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))

    print(f"Faiss Index 已保存：{index_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_key",
        required=True,
        choices=list(VECTOR_STORE_CONFIG.keys()),
        help="选择要构建 Faiss 的 embedding 模型"
    )

    args = parser.parse_args()

    base_dir = find_project_root()

    vector_dir = base_dir / VECTOR_STORE_CONFIG[args.model_key]

    embedding_path = vector_dir / "tool_embeddings.npy"
    index_path = vector_dir / "tool.index"

    if not embedding_path.exists():
        raise FileNotFoundError(f"找不到向量文件：{embedding_path}")

    build_faiss_index(
        embedding_path=embedding_path,
        index_path=index_path
    )


if __name__ == "__main__":
    main()