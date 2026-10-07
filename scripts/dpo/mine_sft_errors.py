"""使用 Qwen3-8B LoRA SFT 模型挖掘 9k 训练集错误。"""

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


# 默认设置与前面的 SFT 模型评测保持一致；也可以通过命令行参数覆盖。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_PATH = PROJECT_ROOT / "model/qwen3-8b-tool-trace-lora-sft"
BATCH_SIZE = 2
MAX_NEW_TOKENS = 128
MAX_ITEMS = None  # 首次检查可通过 --max-items 10 只运行前 10 条。

INPUT_PATH = PROJECT_ROOT / "dataset/tool_calling_trace/train/sft_train.jsonl"
TOOLS_PATH = PROJECT_ROOT / "configs/tools.json"
OUTPUT_PATH = PROJECT_ROOT / "outputs/dpo/qwen3_8b_sft_train_9k_predictions.jsonl"
STATS_PATH = PROJECT_ROOT / "outputs/dpo/qwen3_8b_sft_train_9k_stats.json"

TOOL_LINE_PATTERN = re.compile(r"^\d+\.\s*工具名：([A-Za-z0-9_]+)\s*$")


def parse_tool_json(text, allowed_tools, candidate_tools=None):
    """解析严格的工具 JSON；返回工具列表、解析错误和对应错误类型。"""
    try:
        parsed = json.loads(text.strip())
    except (AttributeError, json.JSONDecodeError):
        return None, "输出不是合法的 JSON", ["JSON格式错误"]

    if not isinstance(parsed, dict) or set(parsed) != {"工具"}:
        return None, "输出必须是只含‘工具’字段的 JSON 对象", ["输出结构错误"]

    predicted_tools = parsed["工具"]
    if not isinstance(predicted_tools, list) or not predicted_tools:
        return None, "‘工具’必须是非空数组", ["工具列表格式错误"]
    if any(not isinstance(tool, str) or not tool for tool in predicted_tools):
        return None, "工具名必须是非空字符串", ["工具名格式错误"]

    validation_errors = []
    error_types = []

    unknown_tools = [tool for tool in predicted_tools if tool not in allowed_tools]
    if unknown_tools:
        validation_errors.append(f"出现未知工具：{unknown_tools}")
        error_types.append("未知工具")

    if candidate_tools is not None:
        noncandidate_tools = [
            tool for tool in predicted_tools
            if tool not in candidate_tools
        ]
        if noncandidate_tools:
            validation_errors.append(f"出现非候选工具：{noncandidate_tools}")
            error_types.append("候选范围外工具")

    if len(predicted_tools) != len(set(predicted_tools)):
        validation_errors.append("工具列表中有重复工具")
        error_types.append("重复工具")

    parse_error = "；".join(validation_errors) if validation_errors else None
    return predicted_tools, parse_error, error_types


def extract_candidate_tools(input_text, allowed_tools, line_number):
    """从 SFT input 的候选工具区块中提取有展示顺序的候选工具。"""
    if "【候选工具】" not in input_text or "【用户问题】" not in input_text:
        raise ValueError(f"训练集第 {line_number} 行缺少候选工具或用户问题区块")

    candidate_block = input_text.split("【候选工具】", maxsplit=1)[1]
    candidate_block = candidate_block.split("【用户问题】", maxsplit=1)[0]
    candidate_tools = []
    for line in candidate_block.splitlines():
        match = TOOL_LINE_PATTERN.match(line.strip())
        if match:
            candidate_tools.append(match.group(1))

    if not candidate_tools:
        raise ValueError(f"训练集第 {line_number} 行没有提取到候选工具")
    if len(candidate_tools) != len(set(candidate_tools)):
        raise ValueError(f"训练集第 {line_number} 行的候选工具有重复")

    unknown_tools = [tool for tool in candidate_tools if tool not in allowed_tools]
    if unknown_tools:
        raise ValueError(f"训练集第 {line_number} 行的候选中有未知工具：{unknown_tools}")
    return candidate_tools


def load_sft_data(input_path, allowed_tools):
    """读取并校验 Alpaca SFT 数据；GT 只用于推理后的评测。"""
    records = []
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"训练集第 {line_number} 行为空")

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"训练集第 {line_number} 行不是合法 JSON") from error

            if set(item) != {"instruction", "input", "output"}:
                raise ValueError(
                    f"训练集第 {line_number} 行必须只含 instruction、input、output"
                )

            instruction = item["instruction"]
            input_text = item["input"]
            gold_output = item["output"]
            if not all(
                isinstance(value, str) and value.strip()
                for value in (instruction, input_text, gold_output)
            ):
                raise ValueError(f"训练集第 {line_number} 行含有空字段或非字符串字段")

            candidate_tools = extract_candidate_tools(
                input_text, allowed_tools, line_number
            )
            gold_tools, gold_error, _ = parse_tool_json(
                gold_output, allowed_tools, candidate_tools
            )
            if gold_error is not None:
                raise ValueError(
                    f"训练集第 {line_number} 行的 GT 输出不合要求：{gold_error}"
                )

            records.append({
                "id": line_number,
                "instruction": instruction,
                "input": input_text,
                "GT输出": gold_output,
                "GT工具": gold_tools,
                "候选工具": candidate_tools,
            })

    if not records:
        raise ValueError("SFT训练集为空")
    return records


def classify_prediction(gold_tools, predicted_tools, parse_error_types):
    """为一条错误预测添加可多选的错误类型。"""
    error_types = list(parse_error_types)
    if predicted_tools is None:
        return error_types

    missing_tools = [tool for tool in gold_tools if tool not in predicted_tools]
    extra_tools = [tool for tool in predicted_tools if tool not in gold_tools]

    if missing_tools:
        error_types.append("遗漏工具")
    if extra_tools:
        error_types.append("多余工具")
    if missing_tools and extra_tools:
        error_types.append("错误工具替换")

    if (
        len(predicted_tools) == len(gold_tools)
        and len(predicted_tools) == len(set(predicted_tools))
        and set(predicted_tools) == set(gold_tools)
        and predicted_tools != gold_tools
    ):
        error_types.append("工具顺序错误")

    if not error_types and predicted_tools != gold_tools:
        error_types.append("其他轨迹错误")

    # 保持可读顺序，同时移除同一条记录中的重复标签。
    return list(dict.fromkeys(error_types))


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


def infer_batch(batch, tokenizer, model, max_new_tokens):
    """按训练时的 Alpaca 结构，将 instruction 与 input 合为用户消息。"""
    import torch

    chat_texts = []
    for item in batch:
        user_content = f"{item['instruction']}\n{item['input']}"
        messages = [{"role": "user", "content": user_content}]
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


def evaluate_prediction(item, raw_output, allowed_tools):
    """解析并评估一条输出，供新推理和断点恢复共同使用。"""
    predicted_tools, parse_error, parse_error_types = parse_tool_json(
        raw_output, allowed_tools, item["候选工具"]
    )
    is_correct = (
        parse_error is None
        and predicted_tools == item["GT工具"]
    )
    inference_result = "正确" if is_correct else "错误"
    intent_type = "单意图" if len(item["GT工具"]) == 1 else "多意图"
    error_types = [] if is_correct else classify_prediction(
        item["GT工具"], predicted_tools, parse_error_types
    )
    return {
        "预测工具": predicted_tools,
        "推理结果": inference_result,
        "意图类型": intent_type,
        "错误类型": error_types,
        "解析错误": parse_error,
    }


def update_counters(evaluation, result_counters, error_type_counter):
    """把一条评估结果加入总体计数器。"""
    inference_result = evaluation["推理结果"]
    intent_type = evaluation["意图类型"]
    result_counters[inference_result] += 1
    result_counters[f"{intent_type}{inference_result}"] += 1
    if evaluation["解析错误"] is not None:
        result_counters["解析错误"] += 1
    error_type_counter.update(evaluation["错误类型"])


def restore_existing_results(output_path, records, allowed_tools):
    """严格校验已有 JSONL，并恢复断点位置和统计计数。"""
    if output_path.stat().st_size:
        with output_path.open("rb") as binary_file:
            binary_file.seek(-1, 2)
            if binary_file.read(1) != b"\n":
                raise ValueError(
                    "已有结果文件的最后一行没有完整换行，可能在写入时中断；"
                    "请先人工检查末行，不会自动修改原文件。"
                )

    result_counters = Counter()
    error_type_counter = Counter()
    completed_count = 0
    stable_fields = (
        "instruction",
        "input",
        "候选工具",
        "GT输出",
        "GT工具",
    )
    evaluation_fields = (
        "预测工具",
        "推理结果",
        "意图类型",
        "错误类型",
        "解析错误",
    )

    with output_path.open("r", encoding="utf-8") as output_file:
        for line_number, line in enumerate(output_file, start=1):
            if not line.strip():
                raise ValueError(f"已有结果文件第 {line_number} 行为空")
            if line_number > len(records):
                raise ValueError("已有结果数量超过本次输入数据数量，不能安全续推")

            try:
                saved_result = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"已有结果文件第 {line_number} 行不是合法 JSON"
                ) from error

            expected_record = records[line_number - 1]
            if saved_result.get("id") != line_number:
                raise ValueError(
                    f"已有结果文件第 {line_number} 行的 id 不连续"
                )
            for field in stable_fields:
                if saved_result.get(field) != expected_record[field]:
                    raise ValueError(
                        f"已有结果文件第 {line_number} 行的 {field} "
                        "与当前9k输入不一致"
                    )

            raw_output = saved_result.get("模型预测输出")
            if not isinstance(raw_output, str):
                raise ValueError(
                    f"已有结果文件第 {line_number} 行缺少模型预测输出"
                )
            evaluation = evaluate_prediction(
                expected_record, raw_output, allowed_tools
            )
            for field in evaluation_fields:
                if saved_result.get(field) != evaluation[field]:
                    raise ValueError(
                        f"已有结果文件第 {line_number} 行的 {field} 校验不一致"
                    )

            update_counters(
                evaluation, result_counters, error_type_counter
            )
            completed_count = line_number

    return completed_count, result_counters, error_type_counter


def build_stats(records, result_counters, error_type_counter, args):
    """汇总本次完整推理的准确率和多标签错误类型数量。"""
    def safe_accuracy(correct_count, sample_count):
        return correct_count / sample_count if sample_count else None

    total = len(records)
    correct = result_counters["正确"]
    incorrect = result_counters["错误"]
    single_total = sum(len(item["GT工具"]) == 1 for item in records)
    multi_total = total - single_total

    return {
        "总数": total,
        "正确数": correct,
        "错误数": incorrect,
        "准确率": safe_accuracy(correct, total),
        "单意图": {
            "总数": single_total,
            "正确数": result_counters["单意图正确"],
            "错误数": result_counters["单意图错误"],
            "准确率": safe_accuracy(
                result_counters["单意图正确"], single_total
            ),
        },
        "多意图": {
            "总数": multi_total,
            "正确数": result_counters["多意图正确"],
            "错误数": result_counters["多意图错误"],
            "准确率": safe_accuracy(
                result_counters["多意图正确"], multi_total
            ),
        },
        "解析错误数": result_counters["解析错误"],
        "错误类型统计": dict(error_type_counter.most_common()),
        "错误类型说明": "错误类型允许多标签，因此各类型数量之和可能大于错误总数。",
        "metadata": {
            "model": args.model,
            "adapter": str(args.adapter),
            "finetuning": "lora_sft",
            "input": str(args.input),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "thinking": False,
            "generation": "greedy",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 Qwen3-8B LoRA SFT 模型挖掘 9k 训练集错误"
    )
    parser.add_argument(
        "--model", default=MODEL_NAME, help="ModelScope 模型 ID 或本地基础模型目录"
    )
    parser.add_argument(
        "--adapter", type=Path, default=ADAPTER_PATH, help="LoRA SFT adapter 目录"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="每批独立推理的样本数"
    )
    parser.add_argument(
        "--max-items", type=int, default=MAX_ITEMS, help="只运行前 N 条，便于小规模测试"
    )
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--tools", type=Path, default=TOOLS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--stats", type=Path, default=STATS_PATH)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="校验已有 output，并从下一条继续追加推理",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise ValueError("batch-size 和 max-new-tokens 必须大于 0")
    if args.max_items is not None and args.max_items < 1:
        raise ValueError("max-items 必须大于 0")
    if args.output.resolve() == args.stats.resolve():
        raise ValueError("output 和 stats 不能使用同一路径")
    if args.resume:
        if not args.output.is_file():
            raise FileNotFoundError(
                f"续推要求已有结果文件存在：{args.output}"
            )
        if args.stats.exists():
            raise FileExistsError(
                f"统计文件已经存在：{args.stats}。为避免覆盖，不会继续执行。"
            )
    else:
        for path in (args.output, args.stats):
            if path.exists():
                raise FileExistsError(
                    f"输出文件已存在：{path}。请更换路径，避免覆盖旧结果。"
                )

    tools_config = json.loads(args.tools.read_text(encoding="utf-8"))
    allowed_tools = set(tools_config)
    records = load_sft_data(args.input, allowed_tools)
    if args.max_items is not None:
        records = records[:args.max_items]

    print(
        f"待推理数据：{len(records)} 条；工具库：{len(allowed_tools)} 个；"
        f"每题使用其 SFT input 中原有候选；batch_size={args.batch_size}"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        completed_count, result_counters, error_type_counter = (
            restore_existing_results(args.output, records, allowed_tools)
        )
        print(
            f"断点文件校验通过：已完成 {completed_count}/{len(records)} 条；"
            f"将从第 {completed_count + 1} 条继续"
        )
    else:
        completed_count = 0
        result_counters = Counter()
        error_type_counter = Counter()

    remaining_records = records[completed_count:]
    if remaining_records:
        tokenizer, model = load_model(args.model, args.adapter)
    else:
        tokenizer = None
        model = None
        print("已有结果已覆盖全部输入，无需再次加载模型。")

    # 每批立即写入 JSONL；中途停止时，已完成的结果仍然保留。
    output_mode = "a" if args.resume else "x"
    with args.output.open(output_mode, encoding="utf-8") as output_file:
        for start in range(0, len(remaining_records), args.batch_size):
            batch = remaining_records[start:start + args.batch_size]
            raw_outputs = infer_batch(
                batch, tokenizer, model, args.max_new_tokens
            )

            for item, raw_output in zip(batch, raw_outputs, strict=True):
                evaluation = evaluate_prediction(
                    item, raw_output, allowed_tools
                )
                update_counters(
                    evaluation, result_counters, error_type_counter
                )

                result = {
                    "id": item["id"],
                    "instruction": item["instruction"],
                    "input": item["input"],
                    "候选工具": item["候选工具"],
                    "GT输出": item["GT输出"],
                    "GT工具": item["GT工具"],
                    "模型预测输出": raw_output,
                    **evaluation,
                    "metadata": {
                        "model": args.model,
                        "adapter": str(args.adapter),
                        "finetuning": "lora_sft",
                        "batch_size": args.batch_size,
                        "generated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
                output_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            output_file.flush()
            total_completed = completed_count + start + len(batch)
            print(f"已完成 {total_completed}/{len(records)} 条")

    stats = build_stats(records, result_counters, error_type_counter, args)
    with args.stats.open("x", encoding="utf-8") as stats_file:
        json.dump(stats, stats_file, ensure_ascii=False, indent=2)
        stats_file.write("\n")

    print(f"完整推理结果已保存：{args.output}")
    print(f"统计结果已保存：{args.stats}")
    print(
        f"错误挖掘完成：正确 {stats['正确数']} 条，错误 {stats['错误数']} 条，"
        f"准确率 {stats['准确率']:.2%}"
    )


if __name__ == "__main__":
    main()
