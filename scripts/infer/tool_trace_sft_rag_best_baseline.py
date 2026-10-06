"""实验三第 2 项：基于 RAG 候选工具的 Qwen3-8B LoRA SFT 模型推理。"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


# 默认设置与 RAG 基础模型 baseline 保持一致；也可以通过命令行参数覆盖。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_PATH = PROJECT_ROOT / "model/qwen3-8b-tool-trace-lora-sft"
BATCH_SIZE = 2
TOP_K = 5
MAX_NEW_TOKENS = 128
MAX_ITEMS = None

INPUT_PATH = PROJECT_ROOT / "dataset/tool_calling_trace/test/intent_test.jsonl"
RAG_PATH = PROJECT_ROOT / "outputs/rag/reranker/best_pipeline_detail.csv"
TOOLS_PATH = PROJECT_ROOT / "configs/tools.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/qwen3_tool_trace_rag_baseline_prompt.txt"
OUTPUT_PATH = PROJECT_ROOT / "outputs/infer/qwen3_8b_tool_trace_sft_rag_top5.jsonl"


def load_test_data(input_path, allowed_tools):
    """读取有序 GT；GT 只用于评测和保存结果，不会进入模型 Prompt。"""
    records = []
    seen_questions = set()

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"测试集第 {line_number} 行为空")

            item = json.loads(line)
            question = item.get("问题")
            gold = item.get("工具")
            gold_tools = [gold] if isinstance(gold, str) else gold

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"测试集第 {line_number} 行缺少有效的‘问题’")
            if question in seen_questions:
                raise ValueError(f"测试集第 {line_number} 行的问题重复，无法唯一匹配 RAG 结果")
            if not isinstance(gold_tools, list) or not gold_tools:
                raise ValueError(f"测试集第 {line_number} 行缺少有效的‘工具’")
            if any(not isinstance(tool, str) or tool not in allowed_tools for tool in gold_tools):
                raise ValueError(f"测试集第 {line_number} 行含有未知 GT 工具")

            seen_questions.add(question)
            records.append({
                "id": line_number,
                "问题": question,
                "GT工具": gold_tools,
            })

    if not records:
        raise ValueError("测试集为空")
    return records


def attach_rag_candidates(records, rag_path, allowed_tools, top_k):
    """按问题匹配 RAG 结果；只取 reranked_tools，不使用 CSV 中的 GT 或评测字段。"""
    rag_by_question = {}

    with rag_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"query", "reranked_tools"}.issubset(reader.fieldnames):
            raise ValueError("RAG CSV 缺少 query 或 reranked_tools 列")

        for line_number, row in enumerate(reader, start=2):
            question = row["query"]
            if not question:
                raise ValueError(f"RAG CSV 第 {line_number} 行的 query 为空")
            if question in rag_by_question:
                raise ValueError(f"RAG CSV 第 {line_number} 行的 query 重复")

            try:
                ranked_tools = json.loads(row["reranked_tools"])
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"RAG CSV 第 {line_number} 行的 reranked_tools 不是合法 JSON") from error

            if not isinstance(ranked_tools, list) or len(ranked_tools) < top_k:
                raise ValueError(f"RAG CSV 第 {line_number} 行不足 {top_k} 个候选工具")

            candidates = ranked_tools[:top_k]
            if any(not isinstance(tool, str) or tool not in allowed_tools for tool in candidates):
                raise ValueError(f"RAG CSV 第 {line_number} 行的前 {top_k} 个候选中有未知工具")
            if len(candidates) != len(set(candidates)):
                raise ValueError(f"RAG CSV 第 {line_number} 行的前 {top_k} 个候选有重复工具")

            rag_by_question[question] = candidates

    test_questions = {record["问题"] for record in records}
    rag_questions = set(rag_by_question)
    if test_questions != rag_questions:
        missing = len(test_questions - rag_questions)
        extra = len(rag_questions - test_questions)
        raise ValueError(f"测试集与 RAG CSV 的问题不一致：缺少 {missing} 条，多出 {extra} 条")

    return [
        {**record, "RAG候选工具": rag_by_question[record["问题"]]}
        for record in records
    ]


def build_tool_catalog(candidate_tools, tools_config):
    """仅拼接候选工具名、description 和 enhanced_description。"""
    blocks = []
    for tool_name in candidate_tools:
        config = tools_config[tool_name]
        blocks.append(
            f"工具名：{tool_name}\n"
            f"基本描述：{config['description']}\n"
            f"详细说明：{config['enhanced_description']}"
        )
    return "\n\n".join(blocks)


def parse_model_output(raw_output, candidate_tools, allowed_tools):
    """严格解析输出；JSON、工具名或候选范围不合要求时记录错误。"""
    try:
        parsed = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return None, "输出不是合法的 JSON"

    if not isinstance(parsed, dict) or set(parsed) != {"工具"}:
        return None, "输出必须是只含‘工具’字段的 JSON 对象"

    predicted_tools = parsed["工具"]
    if not isinstance(predicted_tools, list) or not predicted_tools:
        return None, "‘工具’必须是非空数组"
    if any(not isinstance(tool, str) or not tool for tool in predicted_tools):
        return None, "工具名必须是非空字符串"

    unknown_tools = [tool for tool in predicted_tools if tool not in allowed_tools]
    if unknown_tools:
        return predicted_tools, f"出现未知工具：{unknown_tools}"

    noncandidate_tools = [tool for tool in predicted_tools if tool not in candidate_tools]
    if noncandidate_tools:
        return predicted_tools, f"出现非候选工具：{noncandidate_tools}"
    if len(predicted_tools) != len(set(predicted_tools)):
        return predicted_tools, "工具列表中有重复工具"

    return predicted_tools, None


def load_model(model_name, adapter_path):
    """加载基础 Qwen3-8B，并在其上挂载 LoRA SFT adapter。"""
    from modelscope import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = Path(model_name).expanduser()
    model_path = str(local_path) if local_path.exists() else snapshot_download(model_name)
    adapter_path = Path(adapter_path).expanduser().resolve()

    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter 缺少 adapter_config.json：{adapter_path}")
    if not (adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"LoRA adapter 缺少 adapter_model.safetensors：{adapter_path}")

    print(f"加载 Tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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
    print(f"挂载 LoRA SFT adapter：{adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    print(f"SFT 模型加载完成，device={model.device}")
    return tokenizer, model


def infer_batch(batch, prompt_template, tools_config, tokenizer, model, max_new_tokens):
    """一批问题分别构造 Prompt；候选工具来自各自的 RAG 检索结果。"""
    import torch

    chat_texts = []
    for item in batch:
        tool_catalog = build_tool_catalog(item["RAG候选工具"], tools_config)
        prompt = prompt_template.format(
            tool_catalog=tool_catalog,
            question=item["问题"],
        )
        messages = [{"role": "user", "content": prompt}]
        chat_texts.append(tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ))

    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_length:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-8B LoRA SFT + RAG Top-K 工具调用轨迹推理")
    parser.add_argument("--model", default=MODEL_NAME, help="ModelScope 模型 ID 或本地基础模型目录")
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH, help="LoRA SFT adapter 目录")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="每批独立推理的问题数")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="每题使用 RAG 排名最前面的 K 个工具")
    parser.add_argument("--max-items", type=int, default=MAX_ITEMS, help="只运行前 N 条，便于小规模测试")
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
        raise FileExistsError(f"输出文件已存在：{args.output}。请更换输出路径，避免覆盖旧实验结果。")

    tools_config = json.loads(args.tools.read_text(encoding="utf-8"))
    allowed_tools = set(tools_config)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    records = load_test_data(args.input, allowed_tools)
    records = attach_rag_candidates(records, args.rag, allowed_tools, args.top_k)
    if args.max_items is not None:
        records = records[:args.max_items]

    # 下载模型前，先检查当题的候选工具能否正确填入模板。
    first_record = records[0]
    prompt_template.format(
        tool_catalog=build_tool_catalog(first_record["RAG候选工具"], tools_config),
        question=first_record["问题"],
    )
    print(
        f"测试题：{len(records)} 条；工具库总数：{len(allowed_tools)} 个（仅用于名称校验）；"
        f"每题候选工具：RAG Top-{args.top_k}；模型：Qwen3-8B + LoRA SFT；"
        f"batch_size={args.batch_size}"
    )

    tokenizer, model = load_model(args.model, args.adapter)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 每批立即写入 JSONL；中途停止时，已完成的结果仍在文件中。
    with args.output.open("x", encoding="utf-8") as output_file:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            raw_outputs = infer_batch(
                batch, prompt_template, tools_config, tokenizer, model, args.max_new_tokens
            )

            for item, raw_output in zip(batch, raw_outputs, strict=True):
                predicted_tools, parse_error = parse_model_output(
                    raw_output, item["RAG候选工具"], allowed_tools
                )
                inference_result = (
                    "正确"
                    if parse_error is None and predicted_tools == item["GT工具"]
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
                        "finetuning": "lora_sft",
                        "rag": True,
                        "top_k": args.top_k,
                        "batch_size": args.batch_size,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_file.flush()
            print(f"已完成 {min(start + len(batch), len(records))}/{len(records)} 条")

    print(f"推理结果已保存：{args.output}")


if __name__ == "__main__":
    main()
