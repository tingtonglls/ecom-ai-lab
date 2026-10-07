"""Qwen3-8B LoRA DPO 模型基于 RAG Top-K 候选的轨迹推理。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import tool_trace_sft_rag_best_baseline as shared


# 复用已经验证过的 SFT + RAG 数据匹配、解析与推理函数。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_PATH = PROJECT_ROOT / "model/qwen3-8b-tool-trace-lora-dpo"
BATCH_SIZE = 2
TOP_K = 5
MAX_NEW_TOKENS = 128
MAX_ITEMS = None

INPUT_PATH = PROJECT_ROOT / "dataset/tool_calling_trace/test/intent_test.jsonl"
RAG_PATH = PROJECT_ROOT / "outputs/rag/reranker/best_pipeline_detail.csv"
TOOLS_PATH = PROJECT_ROOT / "configs/tools.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/qwen3_tool_trace_rag_baseline_prompt.txt"
OUTPUT_PATH = (
    PROJECT_ROOT / "outputs/infer/qwen3_8b_tool_trace_dpo_rag_top5.jsonl"
)


def load_model(model_name, adapter_path):
    """加载基础 Qwen3-8B，并挂载 DPO 训练完成的 LoRA adapter。"""
    from modelscope import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = Path(model_name).expanduser()
    model_path = (
        str(local_path) if local_path.exists() else snapshot_download(model_name)
    )
    adapter_path = Path(adapter_path).expanduser().resolve()

    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter 缺少 adapter_config.json：{adapter_path}"
        )
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(
            f"LoRA adapter 缺少 adapter_model.safetensors：{adapter_path}"
        )

    print(f"加载 Tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载基础模型：{model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    print(f"挂载 LoRA DPO adapter：{adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print(f"DPO 模型加载完成，device={model.device}")
    return tokenizer, model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-8B LoRA DPO + RAG Top-K 工具调用轨迹推理"
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help="ModelScope 模型 ID 或本地基础模型目录",
    )
    parser.add_argument(
        "--adapter", type=Path, default=ADAPTER_PATH, help="LoRA DPO adapter 目录"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="每批独立推理的问题数"
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K, help="每题使用 RAG 排名最前面的 K 个工具"
    )
    parser.add_argument(
        "--max-items", type=int, default=MAX_ITEMS, help="只运行前 N 条，便于小规模测试"
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--rag", type=Path, default=RAG_PATH)
    parser.add_argument("--tools", type=Path, default=TOOLS_PATH)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.top_k < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size、top-k 和 max-new-tokens 必须大于 0")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("max-items 必须大于 0")
    if args.output.exists():
        raise FileExistsError(
            f"输出文件已存在：{args.output}。请更换输出路径，避免覆盖旧实验结果。"
        )

    tools_config = json.loads(args.tools.read_text(encoding="utf-8"))
    allowed_tools = set(tools_config)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    records = shared.load_test_data(args.input, allowed_tools)
    records = shared.attach_rag_candidates(
        records, args.rag, allowed_tools, args.top_k
    )
    if args.max_items is not None:
        records = records[:args.max_items]

    first_record = records[0]
    prompt_template.format(
        tool_catalog=shared.build_tool_catalog(
            first_record["RAG候选工具"], tools_config
        ),
        question=first_record["问题"],
    )
    print(
        f"测试题：{len(records)} 条；"
        f"工具库总数：{len(allowed_tools)} 个（仅用于名称校验）；"
        f"每题候选工具：RAG Top-{args.top_k}；"
        f"模型：Qwen3-8B + LoRA DPO；batch_size={args.batch_size}"
    )

    tokenizer, model = load_model(args.model, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("x", encoding="utf-8") as output_file:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            raw_outputs = shared.infer_batch(
                batch,
                prompt_template,
                tools_config,
                tokenizer,
                model,
                args.max_new_tokens,
            )

            for item, raw_output in zip(batch, raw_outputs, strict=True):
                predicted_tools, parse_error = shared.parse_model_output(
                    raw_output, item["RAG候选工具"], allowed_tools
                )
                inference_result = (
                    "正确"
                    if parse_error is None
                    and predicted_tools == item["GT工具"]
                    else "错误"
                )
                result = {
                    "id": item["id"],
                    "问题": item["问题"],
                    "GT工具": item["GT工具"],
                    "RAG候选工具": item["RAG候选工具"],
                    "预测工具": predicted_tools,
                    "推理结果": inference_result,
                    "解析错误": parse_error,
                    "模型原始输出": raw_output,
                    "metadata": {
                        "model": args.model,
                        "adapter": str(args.adapter),
                        "finetuning": "lora_dpo",
                        "rag": True,
                        "top_k": args.top_k,
                        "batch_size": args.batch_size,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
            output_file.flush()
            print(
                f"已完成 {min(start + len(batch), len(records))}/"
                f"{len(records)} 条"
            )

    print(f"推理结果已保存：{args.output}")


if __name__ == "__main__":
    main()
