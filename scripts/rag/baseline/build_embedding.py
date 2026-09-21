###  离线-将工具应用描述向量化

import json
from pathlib import Path
from typing import Optional

import numpy as np
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer


# ======================
# 模型配置
# ======================

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
        "output_dir": "data/vector_store/bge-base"
    },

    "bge-large": {
        "model_id": "BAAI/bge-large-zh-v1.5",
        "output_dir": "data/vector_store/bge-large"
    },

    "qwen3-embedding-0.6b": {
        "model_id": "Qwen/Qwen3-Embedding-0.6B",
        "output_dir": "data/vector_store/qwen3-embedding-0.6b"
    },

    "my-bge-base-bs16_ep2_lr1e-5": {
        "model_path": "model/my-bge-base/bs16_ep2_lr1e-5",
        "output_dir": "data/vector_store/my-bge-base/bs16_ep2_lr1e-5"
    },

    "my-bge-base-bs16_ep3_lr1e-5": {
        "model_path": "model/my-bge-base/bs16_ep3_lr1e-5",
        "output_dir": "data/vector_store/my-bge-base/bs16_ep3_lr1e-5"
    },

    "my-bge-base-bs16_ep3_lr1e-5_hardneg": {
        "model_path": "model/my-bge-base/bs16_ep3_lr1e-5_hardneg",
        "output_dir": "data/vector_store/my-bge-base/bs16_ep3_lr1e-5_hardneg"
    },

    "my-bge-base-bs16_ep1_lr1e-5": {
            "model_path": "model/my-bge-base/bs16_ep1_lr1e-5",
            "output_dir": "data/vector_store/my-bge-base/bs16_ep1_lr1e-5"
        }


}


def load_tools(tools_path):
    with open(tools_path, "r", encoding="utf-8") as f:
        tools = json.load(f)

    tool_names = []
    texts = []

    for tool_name, tool_info in tools.items():
        enhanced_description = tool_info.get("enhanced_description", "")

        if not enhanced_description:
            raise ValueError(f"{tool_name} 缺少 enhanced_description 字段")

        tool_names.append(tool_name)
        texts.append(enhanced_description)

    return tool_names, texts


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_key",
        required=True,
        choices=list(MODEL_CONFIG.keys())
    )

    args = parser.parse_args()

    base_dir = find_project_root()

    tools_path = base_dir / "configs" / "tools.json"

    config = MODEL_CONFIG[args.model_key]

    model_id = config.get("model_id")
    model_path = config.get("model_path")

    output_dir = base_dir / config["output_dir"]

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"读取工具配置：{tools_path}")

    tool_names, texts = load_tools(tools_path)

    print(f"工具数量：{len(tool_names)}")

    if model_path:
        model_dir = base_dir / model_path
        if not model_dir.exists():
            raise FileNotFoundError(f"找不到本地微调模型目录：{model_dir}")
        print(f"从本地加载模型：{model_dir}")
    else:
        print(f"从 ModelScope 下载/读取模型：{model_id}")
        model_dir = snapshot_download(model_id)
        print(f"模型目录：{model_dir}")

    print("加载 embedding 模型...")

    model = SentenceTransformer(
        str(model_dir),
        trust_remote_code=True
    )

    print("开始编码 enhanced_description...")

    embeddings = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,  #向量归一化 方便faiss计算内积时直接得到余弦相似度
        convert_to_numpy=True,
        show_progress_bar=True
    )

    embeddings = embeddings.astype("float32")

    embedding_path = output_dir / "tool_embeddings.npy"

    metadata_path = output_dir / "tool_metadata.json"

    np.save(
        embedding_path,
        embeddings
    )

    metadata = []

    for idx, tool_name in enumerate(tool_names):
        metadata.append(
            {
                "index": idx,
                "tool_name": tool_name,
                # "text": texts[idx]
            }
        )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Embedding 构建完成")
    print(f"向量矩阵形状：{embeddings.shape}")
    print(f"向量保存到：{embedding_path}")
    print(f"元数据保存到：{metadata_path}")


if __name__ == "__main__":
    main()