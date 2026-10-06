"""实验三第 0 项：Qwen3-8B 无 RAG 工具调用轨迹推理。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# 可以直接修改这些默认值，也可以在运行时通过命令行参数覆盖。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen3-8B"
BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
MAX_ITEMS = None  # 首次检查可改成 10；正式运行改回 None。

INPUT_PATH = PROJECT_ROOT / "dataset/tool_calling_trace/test/intent_test.jsonl"
TOOLS_PATH = PROJECT_ROOT / "configs/tools.json"
PROMPT_PATH = PROJECT_ROOT / "prompts/qwen3_tool_trace_baseline_prompt.txt"
OUTPUT_PATH = PROJECT_ROOT / "outputs/infer/qwen3_8b_tool_trace_baseline.jsonl"


def load_test_data(input_path, allowed_tools):
    """读取问题和 GT；GT 只用于保存结果，不会进入模型 Prompt。"""
    records = []
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
            if not isinstance(gold_tools, list) or not gold_tools:
                raise ValueError(f"测试集第 {line_number} 行缺少有效的‘工具’")
            if any(tool not in allowed_tools for tool in gold_tools):
                raise ValueError(f"测试集第 {line_number} 行含有未知 GT 工具")

            records.append({
                "id": line_number,
                "问题": question,
                "GT工具": gold_tools,
            })
    return records


def build_tool_catalog(tools_config):
    """只拼接工具名、description 和 enhanced_description。"""
    blocks = []
    for tool_name, config in tools_config.items():
        description = config["description"]
        enhanced_description = config["enhanced_description"]
        blocks.append(
            f"工具名：{tool_name}\n"
            f"基本描述：{description}\n"
            f"详细说明：{enhanced_description}"
        )
    return "\n\n".join(blocks)


def parse_model_output(raw_output, allowed_tools):
    """严格解析模型输出；格式不合要求时保留原文并记录错误。"""
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
    if len(predicted_tools) != len(set(predicted_tools)):
        return predicted_tools, "工具列表中有重复工具"

    return predicted_tools, None


def load_model(model_name):
    """本地路径直接加载；ModelScope 模型 ID 则先下载或读取缓存。"""
    from modelscope import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    local_path = Path(model_name).expanduser()
    model_path = str(local_path) if local_path.exists() else snapshot_download(model_name)

    print(f"加载 Tokenizer：{model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # 自回归模型批量生成需要左侧补齐。
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"模型加载完成，device={model.device}")
    return tokenizer, model


def infer_batch(batch, prompt_template, tool_catalog, tokenizer, model, max_new_tokens):
    """一批有 N 份独立 Prompt，不是把 N 个问题拼成一个 Prompt。"""
    import torch

    chat_texts = []
    for item in batch:
        prompt = prompt_template.format(
            tool_catalog=tool_catalog,
            question=item["问题"],
        )
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        chat_texts.append(chat_text)

    inputs = tokenizer(chat_texts, return_tensors="pt", padding=True)
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 左侧 padding 后，每条输入的张量长度相同；切掉共同的 Prompt 长度。
    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[:, prompt_length:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-8B 工具调用轨迹 baseline 推理")
    parser.add_argument("--model", default=MODEL_NAME, help="ModelScope 模型 ID 或本地模型目录")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="每批独立推理的问题数")
    parser.add_argument("--max-items", type=int, default=MAX_ITEMS, help="只运行前 N 条，便于小规模测试")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--tools", type=Path, default=TOOLS_PATH)
    parser.add_argument("--prompt", type=Path, default=PROMPT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size 和 max-new-tokens 必须大于 0")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("max-items 必须大于 0")
    if args.output.exists():
        raise FileExistsError(f"输出文件已存在：{args.output}。请更换输出路径，避免覆盖旧实验结果。")

    tools_config = json.loads(args.tools.read_text(encoding="utf-8"))
    allowed_tools = set(tools_config)
    tool_catalog = build_tool_catalog(tools_config)
    prompt_template = args.prompt.read_text(encoding="utf-8")
    records = load_test_data(args.input, allowed_tools)
    if args.max_items is not None:
        records = records[:args.max_items]

    # 在下载或加载 8B 模型之前先检查模板能否正常填充。
    prompt_template.format(tool_catalog=tool_catalog, question=records[0]["问题"])
    print(f"测试题：{len(records)} 条；可用工具：{len(allowed_tools)} 个；batch_size={args.batch_size}")

    tokenizer, model = load_model(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # 每批立即写入 JSONL；中途停止时，已完成的结果仍在文件中。
    with args.output.open("x", encoding="utf-8") as output_file:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            raw_outputs = infer_batch(
                batch, prompt_template, tool_catalog, tokenizer, model, args.max_new_tokens
            )

            for item, raw_output in zip(batch, raw_outputs, strict=True):
                predicted_tools, parse_error = parse_model_output(raw_output, allowed_tools)
                inference_result = (
                    "正确"
                    if parse_error is None and predicted_tools == item["GT工具"]
                    else "错误"
                )
                result = {
                    "id": item["id"],
                    "问题": item["问题"],
                    "GT工具": item["GT工具"],
                    "预测工具": predicted_tools,
                    "推理结果": inference_result,
                    "解析错误": parse_error,
                    "模型原始输出": raw_output,
                    "metadata": {
                        "model": args.model,
                        "rag": False,
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
