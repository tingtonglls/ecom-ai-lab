### 
# ① 读取 bm25_tokenizer_config.json
# ② 初始化 tokenizer，根据bm25配置文件，这里选用jieba
# ③ 加载 user_query_dict.txt 读取 tools.json 提取enhanced_description
# ④ 分词、建立 BM25 Index
# ⑤ 保存 pkl、元数据
###


import json
from pathlib import Path
from typing import Optional

import jieba
import joblib
from rank_bm25 import BM25Okapi


def find_project_root(start_path: Optional[Path] = None) -> Path:
    current = (start_path or Path(__file__).resolve()).resolve()

    for path in [current, *current.parents]:
        if (path / "configs").exists() and (path / "data").exists() and (path / "dataset").exists():
            return path

    raise FileNotFoundError(
        f"无法定位项目根目录，请确认脚本位于仓库内。起始路径：{current}"
    )


def load_config(config_path: Path) -> dict:
    """读取 BM25 Tokenizer 配置"""

    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_tokenizer(config: dict, base_dir: Path):
    """初始化中文分词器"""

    tokenizer_name = config["tokenizer"]

    if tokenizer_name == "jieba":

        user_dict = config.get("user_dict")

        if user_dict:

            user_dict_path = base_dir / user_dict

            if user_dict_path.exists():
                jieba.load_userdict(str(user_dict_path))
                print(f"加载用户词典：{user_dict_path}")
            else:
                print(f"警告：未找到用户词典：{user_dict_path}")

        return jieba

    raise NotImplementedError(
        f"暂不支持 tokenizer：{tokenizer_name}"
    )


def load_tools(tools_path: Path) -> dict:
    """读取 tools.json"""

    with tools_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_corpus(
    tools: dict,
    tokenizer,
    description_field: str,
):
    """构建 BM25 Corpus"""

    corpus = []
    metadata = []

    for index, (tool_name, tool_info) in enumerate(tools.items()):

        if description_field not in tool_info:
            raise ValueError(
                f"{tool_name} 缺少字段：{description_field}"
            )

        text = tool_info[description_field].strip()

        tokens = list(
            tokenizer.cut(
                text,
                cut_all=False,
            )
        )

        corpus.append(tokens)

        metadata.append(
            {
                "index": index,
                "tool_name": tool_name,
            }
        )

    return corpus, metadata


def build_bm25_index(corpus):
    """构建 BM25 Index"""

    return BM25Okapi(corpus)


def main():

    base_dir = Path(__file__).resolve().parents[2]

    config_path = (
        base_dir
        / "configs"
        / "bm25_tokenizer_config.json"
    )

    tools_path = (
        base_dir
        / "configs"
        / "tools.json"
    )

    output_dir = (
        base_dir
        / "data"
        / "vector_store"
        / "bm25"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bm25_path = output_dir / "tool_bm25.pkl"

    metadata_path = (
        output_dir
        / "tool_metadata.json"
    )

    print("=" * 60)
    print("加载 BM25 配置...")

    config = load_config(config_path)

    print(f"Tokenizer：{config['tokenizer']}")
    print(f"Description Field：{config['description_field']}")

    print("=" * 60)

    tokenizer = build_tokenizer(
        config=config,
        base_dir=base_dir,
    )

    print("读取 tools.json...")

    tools = load_tools(tools_path)

    print(f"工具数量：{len(tools)}")

    print("构建 BM25 Corpus...")

    corpus, metadata = build_corpus(
        tools=tools,
        tokenizer=tokenizer,
        description_field=config["description_field"],
    )

    print("构建 BM25 Index...")

    bm25 = build_bm25_index(corpus)

    print("保存 BM25 Index...")

    joblib.dump(
        bm25,
        bm25_path,
    )

    print("保存 Metadata...")

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 60)
    print("BM25 Index 构建完成")
    print(f"Index：{bm25_path}")
    print(f"Metadata：{metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()